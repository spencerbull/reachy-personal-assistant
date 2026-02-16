"""
LLM-based Emotion Inference.

Uses the same vLLM endpoint as the main agent to infer Reachy's emotional
state from conversation context. Runs asynchronously at configurable intervals.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Literal, TYPE_CHECKING

import aiohttp

from agent.soul.config import SoulConfig
from agent.soul.logging_utils import SoulLogger, LogTimer
from agent.memory.emotional import EmotionalState, EMOTION_EXPRESSIONS

if TYPE_CHECKING:
    from agent.soul.personality import Personality

logger = logging.getLogger(__name__)
soul_log = SoulLogger("agent.soul.emotion")


@dataclass
class EmotionInferenceResult:
    """Result of emotion inference."""

    emotion: EmotionalState
    intensity: float  # 0.0 - 1.0
    movement_hint: Optional[str] = None
    confidence: float = 1.0
    inference_time_ms: float = 0.0

    def __str__(self) -> str:
        return f"{self.emotion}({self.intensity:.2f})"


class EmotionInference:
    """
    LLM-based emotion inference for the soul system.

    Analyzes conversation context to determine what emotional state
    Reachy should express. Uses small, focused prompts for fast inference.
    """

    # Base system prompt for emotion inference (kept small for speed)
    BASE_SYSTEM_PROMPT = """You analyze conversation context to determine emotional state for a robot assistant.

Respond with JSON only, no explanation:
{"emotion": "...", "intensity": 0.0-1.0, "movement_hint": "..."}

Available emotions: neutral, happy, curious, attentive, thinking, excited, helpful, playful

Guidelines:
- Default to "attentive" when user is engaged
- "thinking" when processing complex requests
- "happy" for positive interactions, jokes, gratitude
- "curious" for interesting topics or questions
- "excited" for celebrations, achievements
- "playful" for casual/fun conversations
- "helpful" when assisting with tasks
- Intensity 0.3-0.5 for subtle, 0.6-0.8 for clear, 0.9+ for strong

movement_hint is optional - use for specific actions like "nod", "tilt_head", "lean_forward\""""

    def __init__(self, config: SoulConfig, personality: Optional["Personality"] = None):
        self.config = config
        self._personality = personality
        self._last_inference_time: float = 0
        self._current_emotion = EmotionInferenceResult(emotion="neutral", intensity=0.5)
        self._session: Optional[aiohttp.ClientSession] = None

        # Build system prompt with personality context
        self._system_prompt = self._build_system_prompt()

        # Context tracking
        self._last_user_message: str = ""
        self._last_bot_response: str = ""
        self._user_state: str = "idle"  # idle, speaking, waiting
        self._time_since_interaction_ms: int = 0
        self._last_interaction_time: float = time.time()

        # Staleness detection: monotonic counter bumped on every context update.
        # If the version changes between when an LLM request is sent and when
        # the response arrives, the result is considered stale and its confidence
        # is reduced (or it is discarded entirely).
        self._context_version: int = 0

    def set_personality(self, personality: "Personality"):
        """Update the personality and rebuild prompts."""
        self._personality = personality
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt, optionally including personality context."""
        prompt = self.BASE_SYSTEM_PROMPT

        if self._personality and self.config.personality.use_personality_prompts:
            personality_context = self._personality.generate_emotion_inference_context()
            if personality_context:
                prompt = f"{prompt}\n\n{personality_context}"

        return prompt

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def update_context(
        self,
        user_message: Optional[str] = None,
        bot_response: Optional[str] = None,
        user_state: Optional[str] = None,
    ):
        """Update conversation context for inference.

        Each call bumps the internal context version counter used for
        staleness detection.
        """
        self._context_version += 1
        context_updates = []

        if user_message is not None:
            self._last_user_message = user_message
            self._last_interaction_time = time.time()
            context_updates.append(
                f"user_msg='{user_message[:30]}...'"
                if len(user_message) > 30
                else f"user_msg='{user_message}'"
            )

        if bot_response is not None:
            self._last_bot_response = bot_response
            self._last_interaction_time = time.time()
            context_updates.append(
                f"bot_resp='{bot_response[:30]}...'"
                if len(bot_response) > 30
                else f"bot_resp='{bot_response}'"
            )

        if user_state is not None:
            old_state = self._user_state
            self._user_state = user_state
            if user_state == "speaking":
                self._last_interaction_time = time.time()
            if old_state != user_state:
                context_updates.append(f"user_state: {old_state} → {user_state}")

        self._time_since_interaction_ms = int(
            (time.time() - self._last_interaction_time) * 1000
        )

        if context_updates and self.config.log_emotions:
            logger.debug(f"[EMOTION] context_update: {', '.join(context_updates)}")

    def _build_inference_prompt(self) -> str:
        """Build the prompt for emotion inference."""
        # Truncate messages to keep prompt small
        user_msg = (
            self._last_user_message[:200] if self._last_user_message else "(none)"
        )
        bot_msg = self._last_bot_response[:200] if self._last_bot_response else "(none)"

        return f"""Current state:
- User is: {self._user_state}
- Last user message: "{user_msg}"
- Last bot response: "{bot_msg}"
- Current emotion: {self._current_emotion.emotion}
- Time since interaction: {self._time_since_interaction_ms}ms

What should the emotional state be now?"""

    async def infer(self, force: bool = False) -> EmotionInferenceResult:
        """
        Infer emotional state from current context.

        Args:
            force: If True, run inference even if interval hasn't passed

        Returns:
            EmotionInferenceResult with inferred emotion
        """
        now = time.time()
        elapsed_ms = (now - self._last_inference_time) * 1000

        # Rate limit inference unless forced
        if not force and elapsed_ms < self.config.emotion_inference_interval_ms:
            return self._current_emotion

        self._last_inference_time = now
        start_time = time.time()
        old_emotion = self._current_emotion.emotion
        old_intensity = self._current_emotion.intensity

        # Snapshot context version before the (potentially slow) LLM call
        version_at_request = self._context_version

        try:
            result = await self._call_llm()
            result.inference_time_ms = (time.time() - start_time) * 1000

            # --- Staleness detection ---
            # If context was updated while the LLM was running, the result
            # may not match the current conversational state.
            version_drift = self._context_version - version_at_request
            if version_drift > 0:
                # Scale confidence down: 1 update → 0.6, 2 → 0.4, 3+ → 0.2
                staleness_factor = max(0.2, 1.0 - 0.4 * version_drift)
                result.confidence *= staleness_factor
                if self.config.debug_logging:
                    logger.debug(
                        f"[EMOTION] stale_result: context drifted {version_drift} versions "
                        f"during {result.inference_time_ms:.0f}ms inference, "
                        f"confidence reduced to {result.confidence:.2f}"
                    )
                # If confidence dropped below threshold, prefer rule-based
                if result.confidence < 0.3:
                    rule_result = self.infer_rule_based()
                    if self.config.debug_logging:
                        logger.debug(
                            f"[EMOTION] discarding stale LLM result "
                            f"({result.emotion}@{result.confidence:.2f}), "
                            f"using rule-based: {rule_result.emotion}"
                        )
                    result = rule_result

            self._current_emotion = result

            # Log the inference result
            if self.config.log_emotions:
                soul_log.log_emotion_inference(
                    emotion=result.emotion,
                    intensity=result.intensity,
                    confidence=result.confidence,
                    movement_hint=result.movement_hint,
                    inference_time_ms=result.inference_time_ms,
                    prompt_preview=self._last_user_message,
                )

                # Log emotion change if it changed
                if (
                    result.emotion != old_emotion
                    or abs(result.intensity - old_intensity) > 0.15
                ):
                    soul_log.log_emotion_change(
                        old_emotion=old_emotion,
                        new_emotion=result.emotion,
                        intensity=result.intensity,
                        reason=f"user: '{self._last_user_message[:40]}...'"
                        if len(self._last_user_message) > 40
                        else f"user: '{self._last_user_message}'",
                        source="LLM"
                        if version_drift == 0
                        else f"LLM(stale×{version_drift})",
                        inference_time_ms=result.inference_time_ms,
                    )

            if self.config.debug_logging:
                logger.debug(
                    f"Emotion inference: {result} ({result.inference_time_ms:.0f}ms)"
                )

            return result

        except Exception as e:
            logger.error(f"[EMOTION] inference_failed: {e}")
            # Return current emotion on failure
            return self._current_emotion

        self._last_inference_time = now
        start_time = time.time()
        old_emotion = self._current_emotion.emotion
        old_intensity = self._current_emotion.intensity

        try:
            result = await self._call_llm()
            result.inference_time_ms = (time.time() - start_time) * 1000
            self._current_emotion = result

            # Log the inference result
            if self.config.log_emotions:
                soul_log.log_emotion_inference(
                    emotion=result.emotion,
                    intensity=result.intensity,
                    confidence=result.confidence,
                    movement_hint=result.movement_hint,
                    inference_time_ms=result.inference_time_ms,
                    prompt_preview=self._last_user_message,
                )

                # Log emotion change if it changed
                if (
                    result.emotion != old_emotion
                    or abs(result.intensity - old_intensity) > 0.15
                ):
                    soul_log.log_emotion_change(
                        old_emotion=old_emotion,
                        new_emotion=result.emotion,
                        intensity=result.intensity,
                        reason=f"user: '{self._last_user_message[:40]}...'"
                        if len(self._last_user_message) > 40
                        else f"user: '{self._last_user_message}'",
                        source="LLM",
                        inference_time_ms=result.inference_time_ms,
                    )

            if self.config.debug_logging:
                logger.debug(
                    f"Emotion inference: {result} ({result.inference_time_ms:.0f}ms)"
                )

            return result

        except Exception as e:
            logger.error(f"[EMOTION] inference_failed: {e}")
            # Return current emotion on failure
            return self._current_emotion

    async def _call_llm(self) -> EmotionInferenceResult:
        """Call the LLM for emotion inference."""
        session = await self._get_session()

        prompt = self._build_inference_prompt()

        payload = {
            "model": self.config.soul_model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.soul_model_temperature,
            "max_tokens": self.config.soul_model_max_tokens,
            "response_format": {"type": "json_object"},
        }

        url = f"{self.config.soul_model_url}/chat/completions"

        try:
            async with session.post(url, json=payload, timeout=10.0) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    return self._current_emotion

                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                return self._parse_response(content)

        except asyncio.TimeoutError:
            logger.warning("Emotion inference timed out")
            return self._current_emotion
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during emotion inference: {e}")
            return self._current_emotion

    def _parse_response(self, content: str) -> EmotionInferenceResult:
        """Parse LLM response into EmotionInferenceResult."""
        try:
            # Clean up response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(content)

            emotion = data.get("emotion", "neutral").lower()
            # Validate emotion
            if emotion not in EMOTION_EXPRESSIONS:
                logger.warning(
                    f"[EMOTION] unknown_emotion: '{emotion}' → defaulting to neutral (valid: {list(EMOTION_EXPRESSIONS.keys())})"
                )
                emotion = "neutral"

            intensity = float(data.get("intensity", 0.5))
            intensity = max(0.0, min(1.0, intensity))  # Clamp to [0, 1]

            movement_hint = data.get("movement_hint")

            if self.config.debug_logging:
                logger.debug(
                    f"[EMOTION] parsed_response: emotion={emotion}, intensity={intensity:.2f}, hint={movement_hint}"
                )

            return EmotionInferenceResult(
                emotion=emotion,
                intensity=intensity,
                movement_hint=movement_hint,
            )

        except json.JSONDecodeError as e:
            logger.error(f"[EMOTION] parse_failed: JSON error - {e}")
            logger.debug(f"[EMOTION] raw_response: {content[:200]}")
            return self._current_emotion
        except Exception as e:
            logger.error(f"[EMOTION] parse_error: {e}")
            return self._current_emotion

    def infer_rule_based(self) -> EmotionInferenceResult:
        """
        Fast rule-based emotion inference (no LLM call).

        Use this for immediate reactions while waiting for LLM inference.
        """
        result = None
        rule_applied = None

        # User is speaking -> attentive
        if self._user_state == "speaking":
            result = EmotionInferenceResult(emotion="attentive", intensity=0.7)
            rule_applied = "user_speaking → attentive"

        # User just stopped speaking -> thinking (processing)
        elif self._user_state == "waiting" and self._time_since_interaction_ms < 2000:
            result = EmotionInferenceResult(emotion="thinking", intensity=0.5)
            rule_applied = "user_just_silent → thinking"

        # Recent positive interaction
        elif self._last_user_message and self._time_since_interaction_ms < 5000:
            text_lower = self._last_user_message.lower()

            if any(
                w in text_lower for w in ["thank", "thanks", "awesome", "great", "love"]
            ):
                result = EmotionInferenceResult(emotion="happy", intensity=0.7)
                rule_applied = (
                    f"positive_words in '{self._last_user_message[:20]}' → happy"
                )

            elif any(w in text_lower for w in ["?", "what", "how", "why", "tell me"]):
                result = EmotionInferenceResult(emotion="curious", intensity=0.5)
                rule_applied = "question_detected → curious"

            elif any(w in text_lower for w in ["help", "can you", "please"]):
                result = EmotionInferenceResult(emotion="helpful", intensity=0.6)
                rule_applied = "help_request → helpful"

        # Long idle -> decay to neutral
        if result is None and self._time_since_interaction_ms > 30000:
            result = EmotionInferenceResult(emotion="neutral", intensity=0.3)
            rule_applied = f"idle_{self._time_since_interaction_ms}ms → neutral_decay"

        # Default: keep current emotion
        if result is None:
            result = self._current_emotion
            rule_applied = "no_rule_match → keep_current"

        # Log rule-based decision (only if emotion changed)
        if self.config.log_emotions and result.emotion != self._current_emotion.emotion:
            soul_log.log_emotion_rule_based(
                emotion=result.emotion,
                intensity=result.intensity,
                rule=rule_applied,
            )

        return result

    @property
    def current_emotion(self) -> EmotionInferenceResult:
        """Get the current inferred emotion."""
        return self._current_emotion

    @property
    def last_inference_age_ms(self) -> float:
        """Time since last inference in milliseconds."""
        return (time.time() - self._last_inference_time) * 1000
