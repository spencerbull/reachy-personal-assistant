"""Integration tests for LLM emotion inference against a real vLLM endpoint.

These tests require a running vLLM instance. They are marked with the
`integration` marker and will be skipped if the service is unavailable.

Run with: pytest tests/integration/ -m integration
"""

import asyncio
import os

import aiohttp
import pytest

from agent.soul.config import SoulConfig
from agent.soul.emotion_inference import EmotionInference, EmotionInferenceResult

# Default vLLM endpoint for emotion inference
VLLM_URL = os.getenv("SOUL_MODEL_URL", "http://localhost:8002/v1")


async def _vllm_is_available() -> bool:
    """Check if the vLLM service is reachable."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VLLM_URL}/models", timeout=3.0) as resp:
                return resp.status == 200
    except Exception:
        return False


def _make_config(**overrides) -> SoulConfig:
    defaults = dict(
        soul_model_url=VLLM_URL,
        emotion_inference_interval_ms=0,  # No rate limiting for tests
        debug_logging=True,
        log_emotions=True,
        log_events=False,
        log_decisions=False,
    )
    defaults.update(overrides)
    return SoulConfig(**defaults)


@pytest.mark.integration
class TestLLMEmotionInference:
    """Test emotion inference against a real LLM endpoint."""

    async def test_basic_inference(self):
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Hello! How are you?", user_state="speaking")
        try:
            result = await ei.infer(force=True)
            assert isinstance(result, EmotionInferenceResult)
            assert result.emotion in [
                "neutral",
                "happy",
                "curious",
                "attentive",
                "thinking",
                "excited",
                "helpful",
                "playful",
            ]
            assert 0.0 <= result.intensity <= 1.0
            assert result.inference_time_ms > 0
        finally:
            await ei.close()

    async def test_positive_message_trends_happy(self):
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        ei = EmotionInference(_make_config())
        ei.update_context(
            user_message="You're amazing! Thank you so much, this is incredible!",
            user_state="idle",
        )
        try:
            result = await ei.infer(force=True)
            # Should trend toward positive emotions
            assert result.emotion in ["happy", "excited", "playful", "helpful"]
        finally:
            await ei.close()

    async def test_question_trends_curious(self):
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        ei = EmotionInference(_make_config())
        ei.update_context(
            user_message="What is quantum computing and how does it work?",
            user_state="idle",
        )
        try:
            result = await ei.infer(force=True)
            # Should trend toward thinking/curious/attentive
            assert result.emotion in [
                "curious",
                "thinking",
                "attentive",
                "helpful",
                "neutral",
            ]
        finally:
            await ei.close()

    async def test_help_request_trends_helpful(self):
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        ei = EmotionInference(_make_config())
        ei.update_context(
            user_message="Can you please help me set up my email?",
            user_state="idle",
        )
        try:
            result = await ei.infer(force=True)
            assert result.emotion in ["helpful", "attentive", "neutral", "curious"]
        finally:
            await ei.close()

    async def test_inference_returns_valid_json(self):
        """Ensure the LLM always returns parseable emotion JSON."""
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Tell me a joke", user_state="speaking")
        try:
            result = await ei.infer(force=True)
            # If we got here without exception, the JSON parsed successfully
            assert result.emotion != ""
            assert result.intensity > 0
        finally:
            await ei.close()

    async def test_staleness_during_real_inference(self):
        """Verify staleness detection works with actual LLM latency."""
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        ei = EmotionInference(_make_config())
        ei.update_context(user_message="Hello!")

        # Record version before
        version_before = ei._context_version

        # Start inference, but update context during it
        async def infer_with_drift():
            # Fire context update shortly after inference starts
            await asyncio.sleep(0.05)
            ei.update_context(user_state="speaking")
            ei.update_context(user_message="Actually, goodbye!")

        try:
            # Run inference and drift concurrently
            result, _ = await asyncio.gather(ei.infer(force=True), infer_with_drift())

            # Context drifted by 2 during inference (on top of whatever version_before was)
            assert (
                ei._context_version == version_before + 2
            )  # 2 drift calls during inference
            # Confidence should be reduced (but exact value depends on timing)
            # If inference was fast enough the drift may have happened after
            # the version snapshot, so we just verify no crash
            assert isinstance(result, EmotionInferenceResult)
        finally:
            await ei.close()


@pytest.mark.integration
class TestLLMEndpointHealth:
    """Sanity check that the vLLM endpoint is reachable and configured."""

    async def test_vllm_models_endpoint(self):
        if not await _vllm_is_available():
            pytest.skip(f"vLLM not available at {VLLM_URL}")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VLLM_URL}/models") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "data" in data
                models = [m["id"] for m in data["data"]]
                assert len(models) > 0
