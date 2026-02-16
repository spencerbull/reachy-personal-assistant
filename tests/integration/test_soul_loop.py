"""Integration tests for SoulLoop lifecycle and event handling.

These tests create a real SoulLoop with mocked ReachyService (no robot needed)
but exercise the full async event processing pipeline.
"""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from agent.soul.config import SoulConfig
from agent.soul.loop import SoulLoop, SoulState
from agent.soul.events import EventType


def _make_config(**overrides) -> SoulConfig:
    defaults = dict(
        poll_interval_ms=50,  # Fast polling for tests
        emotion_inference_interval_ms=100,
        debug_logging=False,
        log_emotions=False,
        log_events=False,
        log_decisions=False,
        log_movements=False,
        idle_scanning_enabled=False,
        idle_micro_movements_enabled=False,
        # Use a model URL that won't connect (tests mock the LLM)
        soul_model_url="http://localhost:99999/v1",
    )
    defaults.update(overrides)
    return SoulConfig(**defaults)


def _make_mock_reachy_service():
    """Create a mock reachy service that accepts soul poses."""
    service = MagicMock()
    service.apply_soul_pose = MagicMock()
    service.set_face_tracking = MagicMock()
    service.antenna_wave = MagicMock()
    service.look_around = MagicMock()
    return service


class TestSoulLoopLifecycle:
    """Test start/stop and basic state."""

    async def test_start_and_stop(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        assert not soul.is_running

        await soul.start()
        assert soul.is_running

        await soul.stop()
        assert not soul.is_running

    async def test_double_start_is_safe(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()
        await soul.start()  # Should not crash
        assert soul.is_running
        await soul.stop()

    async def test_stop_without_start_is_safe(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.stop()  # Should not crash

    async def test_get_status_returns_dict(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        status = soul.get_status()
        assert isinstance(status, dict)


class TestSoulLoopEventHandling:
    """Test that events are processed correctly by the soul loop."""

    async def test_user_speaking_event(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_user_speaking()
        await asyncio.sleep(0.2)  # Let the loop process

        assert soul._state.user_state == "speaking"
        await soul.stop()

    async def test_user_silent_event(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_user_speaking()
        await asyncio.sleep(0.1)
        soul.on_user_silent(silence_duration_ms=500)
        await asyncio.sleep(0.2)

        # User state should have changed from speaking
        assert soul._state.user_state != "speaking"
        await soul.stop()

    async def test_user_message_updates_emotion_context(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_user_message("Thank you!")
        await asyncio.sleep(0.2)

        # The emotion inference should have received the message
        assert soul._emotion_inference._last_user_message == "Thank you!"
        await soul.stop()

    async def test_bot_speaking_started_sets_flag(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_bot_speaking_started()
        await asyncio.sleep(0.2)

        assert soul._state.bot_is_speaking is True
        await soul.stop()

    async def test_bot_speaking_stopped_clears_flag(self):
        config = _make_config()
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_bot_speaking_started()
        await asyncio.sleep(0.1)
        soul.on_bot_speaking_stopped()
        await asyncio.sleep(0.2)

        assert soul._state.bot_is_speaking is False
        await soul.stop()

    async def test_face_detected_suppresses_scanning(self):
        config = _make_config(idle_scanning_enabled=True)
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_face_detected(position=(0.5, 0.5), is_new=True)
        await asyncio.sleep(0.2)

        assert soul._state.face_detected is True
        assert soul._idle_generator._scanning_suppressed is True
        await soul.stop()

    async def test_face_lost_unsuppresses_scanning(self):
        config = _make_config(idle_scanning_enabled=True)
        soul = SoulLoop(config=config, reachy_service=_make_mock_reachy_service())
        await soul.start()

        soul.on_face_detected(position=(0.5, 0.5), is_new=True)
        await asyncio.sleep(0.1)
        soul.on_face_lost()
        await asyncio.sleep(0.2)

        assert soul._state.face_detected is False
        assert soul._idle_generator._scanning_suppressed is False
        await soul.stop()


class TestSoulLoopModeTransitionChain:
    """Test the full mode transition chain that matches real conversation flow.

    Simulates: idle -> user speaks -> user stops -> processing -> bot speaks -> idle
    """

    async def test_conversation_event_chain(self):
        """Fire events in the order they occur during a real conversation and
        verify soul state at each step."""
        config = _make_config(poll_interval_ms=30)
        service = _make_mock_reachy_service()
        soul = SoulLoop(config=config, reachy_service=service)
        await soul.start()
        await asyncio.sleep(0.1)

        # 1. User starts speaking
        soul.on_user_speaking()
        await asyncio.sleep(0.15)
        assert soul._state.user_state == "speaking"

        # 2. User message transcribed
        soul.on_user_message("What can you do?")
        await asyncio.sleep(0.1)

        # 3. User stops speaking
        soul.on_user_silent(silence_duration_ms=200)
        await asyncio.sleep(0.15)

        # 4. Bot responds (LLM done)
        soul.on_bot_response("I can help you with lots of things!")
        await asyncio.sleep(0.1)

        # 5. TTS starts playing
        soul.on_bot_speaking_started()
        await asyncio.sleep(0.15)
        assert soul._state.bot_is_speaking is True

        # 6. TTS finishes
        soul.on_bot_speaking_stopped()
        await asyncio.sleep(0.15)
        assert soul._state.bot_is_speaking is False

        await soul.stop()
