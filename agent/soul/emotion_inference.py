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
from typing import Optional, Literal

import aiohttp

from agent.soul.config import SoulConfig
from agent.memory.emotional import EmotionalState, EMOTION_EXPRESSIONS

logger = logging.getLogger(__name__)


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
    
    # System prompt for emotion inference (kept small for speed)
    SYSTEM_PROMPT = """You analyze conversation context to determine emotional state for a robot assistant.

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

    def __init__(self, config: SoulConfig):
        self.config = config
        self._last_inference_time: float = 0
        self._current_emotion = EmotionInferenceResult(emotion="neutral", intensity=0.5)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Context tracking
        self._last_user_message: str = ""
        self._last_bot_response: str = ""
        self._user_state: str = "idle"  # idle, speaking, waiting
        self._time_since_interaction_ms: int = 0
        self._last_interaction_time: float = time.time()
    
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
        """Update conversation context for inference."""
        if user_message is not None:
            self._last_user_message = user_message
            self._last_interaction_time = time.time()
        if bot_response is not None:
            self._last_bot_response = bot_response
            self._last_interaction_time = time.time()
        if user_state is not None:
            self._user_state = user_state
            if user_state == "speaking":
                self._last_interaction_time = time.time()
        
        self._time_since_interaction_ms = int((time.time() - self._last_interaction_time) * 1000)
    
    def _build_inference_prompt(self) -> str:
        """Build the prompt for emotion inference."""
        # Truncate messages to keep prompt small
        user_msg = self._last_user_message[:200] if self._last_user_message else "(none)"
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
        
        try:
            result = await self._call_llm()
            result.inference_time_ms = (time.time() - start_time) * 1000
            self._current_emotion = result
            
            if self.config.debug_logging:
                logger.debug(f"Emotion inference: {result} ({result.inference_time_ms:.0f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion inference failed: {e}")
            # Return current emotion on failure
            return self._current_emotion
    
    async def _call_llm(self) -> EmotionInferenceResult:
        """Call the LLM for emotion inference."""
        session = await self._get_session()
        
        prompt = self._build_inference_prompt()
        
        payload = {
            "model": self.config.soul_model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.soul_model_temperature,
            "max_tokens": self.config.soul_model_max_tokens,
            "response_format": {"type": "json_object"},
        }
        
        url = f"{self.config.soul_model_url}/chat/completions"
        
        try:
            async with session.post(url, json=payload, timeout=5.0) as response:
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
                logger.warning(f"Unknown emotion '{emotion}', defaulting to neutral")
                emotion = "neutral"
            
            intensity = float(data.get("intensity", 0.5))
            intensity = max(0.0, min(1.0, intensity))  # Clamp to [0, 1]
            
            movement_hint = data.get("movement_hint")
            
            return EmotionInferenceResult(
                emotion=emotion,
                intensity=intensity,
                movement_hint=movement_hint,
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse emotion response: {e}")
            logger.debug(f"Raw response: {content}")
            return self._current_emotion
        except Exception as e:
            logger.error(f"Error parsing emotion response: {e}")
            return self._current_emotion
    
    def infer_rule_based(self) -> EmotionInferenceResult:
        """
        Fast rule-based emotion inference (no LLM call).
        
        Use this for immediate reactions while waiting for LLM inference.
        """
        # User is speaking -> attentive
        if self._user_state == "speaking":
            return EmotionInferenceResult(emotion="attentive", intensity=0.7)
        
        # User just stopped speaking -> thinking (processing)
        if self._user_state == "waiting" and self._time_since_interaction_ms < 2000:
            return EmotionInferenceResult(emotion="thinking", intensity=0.5)
        
        # Recent positive interaction
        if self._last_user_message and self._time_since_interaction_ms < 5000:
            text_lower = self._last_user_message.lower()
            
            if any(w in text_lower for w in ["thank", "thanks", "awesome", "great", "love"]):
                return EmotionInferenceResult(emotion="happy", intensity=0.7)
            
            if any(w in text_lower for w in ["?", "what", "how", "why", "tell me"]):
                return EmotionInferenceResult(emotion="curious", intensity=0.5)
            
            if any(w in text_lower for w in ["help", "can you", "please"]):
                return EmotionInferenceResult(emotion="helpful", intensity=0.6)
        
        # Long idle -> decay to neutral
        if self._time_since_interaction_ms > 30000:
            return EmotionInferenceResult(emotion="neutral", intensity=0.3)
        
        # Default: keep current emotion
        return self._current_emotion
    
    @property
    def current_emotion(self) -> EmotionInferenceResult:
        """Get the current inferred emotion."""
        return self._current_emotion
    
    @property
    def last_inference_age_ms(self) -> float:
        """Time since last inference in milliseconds."""
        return (time.time() - self._last_inference_time) * 1000
