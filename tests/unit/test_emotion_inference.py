"""Unit tests for EmotionInference: rule-based, staleness detection, response parsing."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from agent.soul.config import SoulConfig
from agent.soul.emotion_inference import EmotionInference, EmotionInferenceResult


def _make_config(**overrides) -> SoulConfig:
    defaults = dict(
        debug_logging=False,
        log_emotions=False,
        log_events=False,
        log_decisions=False,
        emotion_inference_interval_ms=100,
        # Disable damping by default in tests so pre-existing tests
        # don't need to account for multi-cycle commit requirements
        emotion_damping_cycles=1,
        emotion_min_hold_time_s=0.0,
    )
    defaults.update(overrides)
    return SoulConfig(**defaults)


class TestEmotionInferenceResult:
    """Test the result dataclass."""

    def test_str_representation(self):
        r = EmotionInferenceResult(emotion="happy", intensity=0.8)
        assert "happy" in str(r)
        assert "0.80" in str(r)

    def test_default_confidence(self):
        r = EmotionInferenceResult(emotion="neutral", intensity=0.5)
        assert r.confidence == 1.0


class TestRuleBasedInference:
    """Test the fast rule-based emotion inference (no LLM)."""

    def test_user_speaking_returns_attentive(self):
        ei = EmotionInference(_make_config())
        ei.update_context(user_state="speaking")
        result = ei.infer_rule_based()
        assert result.emotion == "attentive"
        assert result.intensity >= 0.5

    def test_user_just_stopped_returns_thinking(self):
        ei = EmotionInference(_make_config())
        ei.update_context(user_state="waiting")
        ei._last_interaction_time = time.time()  # Just now
        ei._time_since_interaction_ms = 500
        result = ei.infer_rule_based()
        assert result.emotion == "thinking"

    def test_positive_words_return_happy(self):
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Thank you so much!")
        ei._time_since_interaction_ms = 1000  # Recent
        result = ei.infer_rule_based()
        assert result.emotion == "happy"

    def test_question_returns_curious(self):
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="What can you do?")
        ei._time_since_interaction_ms = 1000
        result = ei.infer_rule_based()
        assert result.emotion == "curious"

    def test_help_request_returns_helpful(self):
        ei = EmotionInference(_make_config())
        # Use a message with "help" but no "?" to avoid matching the question rule first
        ei.update_context(user_message="Please help me set up my email")
        ei._time_since_interaction_ms = 1000
        result = ei.infer_rule_based()
        assert result.emotion == "helpful"

    def test_long_idle_returns_neutral_decay(self):
        ei = EmotionInference(_make_config())
        ei._last_interaction_time = time.time() - 60  # 60s ago
        ei._time_since_interaction_ms = 60000
        result = ei.infer_rule_based()
        assert result.emotion == "neutral"
        assert result.intensity <= 0.3

    def test_no_match_keeps_current(self):
        ei = EmotionInference(_make_config())
        ei._current_emotion = EmotionInferenceResult(emotion="curious", intensity=0.6)
        ei._time_since_interaction_ms = 5000  # Not long enough for decay
        ei._last_user_message = "hmm"  # No keyword match
        result = ei.infer_rule_based()
        assert result.emotion == "curious"


class TestContextVersioning:
    """Test the staleness detection via context versioning."""

    def test_initial_version_is_zero(self):
        ei = EmotionInference(_make_config())
        assert ei._context_version == 0

    def test_update_context_bumps_version(self):
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="hello")
        assert ei._context_version == 1
        ei.update_context(user_state="speaking")
        assert ei._context_version == 2
        ei.update_context(bot_response="hi")
        assert ei._context_version == 3

    def test_multiple_fields_single_bump(self):
        """A single update_context call bumps version once, not per-field."""
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="hello", user_state="speaking")
        assert ei._context_version == 1


class TestStalenessDetection:
    """Test that stale LLM results get reduced confidence or discarded."""

    async def test_no_drift_full_confidence(self):
        """When context doesn't change during inference, confidence stays at 1.0."""
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Hello!")

        mock_result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, confidence=1.0
        )
        with patch.object(
            ei, "_call_llm", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await ei.infer(force=True)
        assert result.confidence == 1.0
        assert result.emotion == "happy"

    async def test_one_drift_reduces_confidence(self):
        """1 context update during inference reduces confidence to ~0.6."""
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Hello!")

        mock_result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, confidence=1.0
        )

        async def fake_llm():
            # Simulate context changing during LLM call
            ei.update_context(user_state="speaking")
            return mock_result

        with patch.object(ei, "_call_llm", side_effect=fake_llm):
            result = await ei.infer(force=True)
        # 1 drift: confidence = 1.0 * max(0.2, 1.0 - 0.4*1) = 0.6
        assert abs(result.confidence - 0.6) < 0.01

    async def test_three_drifts_discards_and_uses_rule_based(self):
        """3+ context updates during inference should discard LLM result."""
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Hello!")
        ei.update_context(user_state="speaking")

        mock_result = EmotionInferenceResult(
            emotion="excited", intensity=0.9, confidence=1.0
        )

        async def fake_llm():
            # 3 context updates during LLM call
            ei.update_context(user_state="waiting")
            ei.update_context(user_message="Goodbye")
            ei.update_context(user_state="idle")
            return mock_result

        with patch.object(ei, "_call_llm", side_effect=fake_llm):
            result = await ei.infer(force=True)
        # 3 drifts: confidence = 1.0 * max(0.2, 1.0-1.2) = 0.2 < 0.3
        # Should fall back to rule-based
        assert result.emotion != "excited"  # LLM result discarded

    async def test_rate_limiting(self):
        """Inference should be rate-limited by interval."""
        ei = EmotionInference(_make_config(emotion_inference_interval_ms=5000))
        ei._last_inference_time = time.time()  # Just inferred

        mock_result = EmotionInferenceResult(emotion="happy", intensity=0.7)
        with patch.object(
            ei, "_call_llm", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await ei.infer(force=False)
        # Should return cached, not call LLM
        assert result.emotion == "neutral"  # Default cached emotion

    async def test_force_bypasses_rate_limit(self):
        ei = EmotionInference(_make_config(emotion_inference_interval_ms=5000))
        ei._last_inference_time = time.time()

        mock_result = EmotionInferenceResult(emotion="happy", intensity=0.7)
        with patch.object(
            ei, "_call_llm", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await ei.infer(force=True)
        assert result.emotion == "happy"


class TestResponseParsing:
    """Test LLM response parsing."""

    def test_parse_valid_json(self):
        ei = EmotionInference(_make_config())
        result = ei._parse_response('{"emotion": "happy", "intensity": 0.8}')
        assert result.emotion == "happy"
        assert result.intensity == 0.8

    def test_parse_with_movement_hint(self):
        ei = EmotionInference(_make_config())
        result = ei._parse_response(
            '{"emotion": "curious", "intensity": 0.5, "movement_hint": "tilt_head"}'
        )
        assert result.movement_hint == "tilt_head"

    def test_parse_unknown_emotion_defaults_to_neutral(self):
        ei = EmotionInference(_make_config())
        result = ei._parse_response('{"emotion": "rage", "intensity": 1.0}')
        assert result.emotion == "neutral"

    def test_parse_clamps_intensity(self):
        ei = EmotionInference(_make_config())
        result = ei._parse_response('{"emotion": "happy", "intensity": 5.0}')
        assert result.intensity == 1.0
        result = ei._parse_response('{"emotion": "happy", "intensity": -1.0}')
        assert result.intensity == 0.0

    def test_parse_markdown_code_block(self):
        ei = EmotionInference(_make_config())
        response = '```json\n{"emotion": "attentive", "intensity": 0.6}\n```'
        result = ei._parse_response(response)
        assert result.emotion == "attentive"

    def test_parse_invalid_json_returns_current(self):
        ei = EmotionInference(_make_config())
        ei._current_emotion = EmotionInferenceResult(emotion="curious", intensity=0.5)
        result = ei._parse_response("not json at all")
        assert result.emotion == "curious"

    def test_build_inference_prompt_contains_context(self):
        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Tell me a joke", user_state="speaking")
        prompt = ei._build_inference_prompt()
        assert "Tell me a joke" in prompt
        assert "speaking" in prompt


class TestEmotionDamping:
    """Test emotion damping to prevent flickering."""

    def test_same_emotion_passes_through(self):
        """When result matches committed emotion, return as-is."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=2, emotion_min_hold_time_s=1.0)
        )
        ei._last_committed_emotion = "happy"
        result = EmotionInferenceResult(
            emotion="happy", intensity=0.8, inference_time_ms=50.0
        )
        damped = ei.apply_damping(result, min_hold_s=1.0, damping_cycles=2)
        assert damped.emotion == "happy"
        assert damped.intensity == 0.8

    def test_new_emotion_not_committed_immediately(self):
        """A new emotion shouldn't commit on first cycle with damping_cycles=2."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=2, emotion_min_hold_time_s=0.0)
        )
        ei._last_committed_emotion = "neutral"
        result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, inference_time_ms=50.0
        )
        damped = ei.apply_damping(result, min_hold_s=0.0, damping_cycles=2)
        # Should still return neutral (not committed yet)
        assert damped.emotion == "neutral"
        assert ei._pending_emotion == "happy"
        assert ei._pending_emotion_count == 1

    def test_new_emotion_commits_after_enough_cycles(self):
        """After damping_cycles consecutive occurrences, emotion commits."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=2, emotion_min_hold_time_s=0.0)
        )
        ei._last_committed_emotion = "neutral"

        result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, inference_time_ms=50.0
        )

        # First cycle: sets pending
        damped = ei.apply_damping(result, min_hold_s=0.0, damping_cycles=2)
        assert damped.emotion == "neutral"

        # Second cycle: pending count reaches 2, commits
        damped = ei.apply_damping(result, min_hold_s=0.0, damping_cycles=2)
        assert damped.emotion == "happy"
        assert ei._last_committed_emotion == "happy"

    def test_different_emotion_resets_pending(self):
        """If a different emotion appears mid-damping, pending resets."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=3, emotion_min_hold_time_s=0.0)
        )
        ei._last_committed_emotion = "neutral"

        happy_result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, inference_time_ms=50.0
        )
        curious_result = EmotionInferenceResult(
            emotion="curious", intensity=0.5, inference_time_ms=50.0
        )

        # First: happy pending
        ei.apply_damping(happy_result, min_hold_s=0.0, damping_cycles=3)
        assert ei._pending_emotion == "happy"
        assert ei._pending_emotion_count == 1

        # Second: curious replaces pending
        ei.apply_damping(curious_result, min_hold_s=0.0, damping_cycles=3)
        assert ei._pending_emotion == "curious"
        assert ei._pending_emotion_count == 1

    def test_high_confidence_skips_damping(self):
        """Very high confidence+intensity emotions bypass damping."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=5, emotion_min_hold_time_s=10.0)
        )
        ei._last_committed_emotion = "neutral"

        result = EmotionInferenceResult(
            emotion="excited", intensity=0.9, confidence=0.95, inference_time_ms=50.0
        )
        damped = ei.apply_damping(result, min_hold_s=10.0, damping_cycles=5)
        # Should bypass damping entirely
        assert damped.emotion == "excited"
        assert ei._last_committed_emotion == "excited"

    def test_min_hold_time_prevents_rapid_changes(self):
        """Even with enough cycles, min_hold_time_s must pass."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=1, emotion_min_hold_time_s=5.0)
        )
        ei._last_committed_emotion = "neutral"
        ei._last_emotion_change_time = time.time()  # Just changed

        result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, inference_time_ms=50.0
        )
        damped = ei.apply_damping(result, min_hold_s=5.0, damping_cycles=1)
        # count=1 >= cycles=1, but hold time not met
        assert damped.emotion == "neutral"

    def test_damping_preserves_inference_time(self):
        """Damped results should preserve the original inference_time_ms."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=3, emotion_min_hold_time_s=0.0)
        )
        ei._last_committed_emotion = "neutral"

        result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, inference_time_ms=123.4
        )
        damped = ei.apply_damping(result, min_hold_s=0.0, damping_cycles=3)
        assert damped.inference_time_ms == 123.4

    def test_single_cycle_damping_commits_immediately(self):
        """With damping_cycles=1 and min_hold_s=0, a new emotion commits on first call."""
        ei = EmotionInference(
            _make_config(emotion_damping_cycles=1, emotion_min_hold_time_s=0.0)
        )
        ei._last_committed_emotion = "neutral"

        result = EmotionInferenceResult(
            emotion="happy", intensity=0.7, inference_time_ms=50.0
        )
        damped = ei.apply_damping(result, min_hold_s=0.0, damping_cycles=1)
        assert damped.emotion == "happy"
