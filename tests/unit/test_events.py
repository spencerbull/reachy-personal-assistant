"""Unit tests for EventQueue, event dispatch, and all event types."""

import asyncio

import pytest

from agent.soul.events import (
    EventQueue,
    EventType,
    SoulEvent,
    UserSpeakingEvent,
    UserSilentEvent,
    UserMessageEvent,
    BotResponseEvent,
    BotStreamingEvent,
    BotSpeakingStartedEvent,
    BotSpeakingStoppedEvent,
    FaceDetectedEvent,
    FaceLostEvent,
    SoundDetectedEvent,
    IdleTickEvent,
)


class TestEventTypes:
    """Verify all event types are correctly defined."""

    def test_all_event_types_exist(self):
        expected = [
            "USER_SPEAKING",
            "USER_SILENT",
            "USER_MESSAGE",
            "BOT_RESPONSE",
            "BOT_STREAMING",
            "BOT_SPEAKING_STARTED",
            "BOT_SPEAKING_STOPPED",
            "FACE_DETECTED",
            "FACE_LOST",
            "SOUND_DETECTED",
            "IDLE_TICK",
        ]
        for name in expected:
            assert hasattr(EventType, name), f"EventType.{name} missing"

    def test_bot_speaking_events_have_correct_types(self):
        started = BotSpeakingStartedEvent()
        stopped = BotSpeakingStoppedEvent()
        assert started.event_type == EventType.BOT_SPEAKING_STARTED
        assert stopped.event_type == EventType.BOT_SPEAKING_STOPPED


class TestEventDataclasses:
    """Test event construction and fields."""

    def test_user_speaking_event(self):
        e = UserSpeakingEvent()
        assert e.event_type == EventType.USER_SPEAKING
        assert e.timestamp > 0

    def test_user_silent_event_with_duration(self):
        e = UserSilentEvent(silence_duration_ms=500)
        assert e.silence_duration_ms == 500

    def test_user_message_event(self):
        e = UserMessageEvent(message="Hello!")
        assert e.message == "Hello!"
        assert "Hello!" in str(e)

    def test_bot_response_event(self):
        e = BotResponseEvent(response="Hi there")
        assert e.response == "Hi there"

    def test_bot_streaming_event(self):
        e = BotStreamingEvent(token="Hi", accumulated_text="Hi there")
        assert e.token == "Hi"
        assert e.accumulated_text == "Hi there"

    def test_face_detected_event(self):
        e = FaceDetectedEvent(position=(0.3, 0.7), is_new_face=True)
        assert e.is_new_face is True
        assert e.position == (0.3, 0.7)
        assert "NEW" in str(e)

    def test_face_lost_event(self):
        e = FaceLostEvent(face_id="face_1")
        assert e.face_id == "face_1"

    def test_idle_tick_event(self):
        e = IdleTickEvent(idle_duration_s=10.5)
        assert e.idle_duration_s == 10.5


class TestEventQueue:
    """Test async event queue operations."""

    async def test_push_and_pop(self):
        q = EventQueue(max_size=10)
        event = UserSpeakingEvent()
        await q.push(event)
        assert q.size == 1
        got = await q.pop(timeout=0.1)
        assert got is event
        assert q.size == 0

    async def test_pop_timeout_returns_none(self):
        q = EventQueue(max_size=10)
        got = await q.pop(timeout=0.05)
        assert got is None

    async def test_overflow_drops_oldest(self):
        q = EventQueue(max_size=2)
        e1 = UserSpeakingEvent()
        e2 = UserSilentEvent()
        e3 = UserMessageEvent(message="overflow")
        await q.push(e1)
        await q.push(e2)
        # Queue full, pushing e3 should drop e1
        await q.push(e3)
        assert q.size == 2
        first = await q.pop(timeout=0.1)
        # e1 was dropped, so first should be e2
        assert first.event_type == EventType.USER_SILENT

    async def test_pop_nowait(self):
        q = EventQueue(max_size=10)
        assert q.pop_nowait() is None
        await q.push(UserSpeakingEvent())
        got = q.pop_nowait()
        assert got is not None

    async def test_clear(self):
        q = EventQueue(max_size=10)
        for _ in range(5):
            await q.push(IdleTickEvent())
        assert q.size == 5
        q.clear()
        assert q.size == 0

    async def test_dispatch_calls_handler(self):
        q = EventQueue()
        received = []

        async def handler(event):
            received.append(event)

        q.register_handler(EventType.USER_SPEAKING, handler)
        event = UserSpeakingEvent()
        await q.dispatch(event)
        assert len(received) == 1
        assert received[0] is event

    async def test_dispatch_only_matching_type(self):
        q = EventQueue()
        received = []

        async def handler(event):
            received.append(event)

        q.register_handler(EventType.USER_SPEAKING, handler)
        await q.dispatch(UserSilentEvent())  # Wrong type
        assert len(received) == 0

    async def test_dispatch_multiple_handlers(self):
        q = EventQueue()
        results = []

        async def h1(event):
            results.append("h1")

        async def h2(event):
            results.append("h2")

        q.register_handler(EventType.FACE_DETECTED, h1)
        q.register_handler(EventType.FACE_DETECTED, h2)
        await q.dispatch(FaceDetectedEvent())
        assert results == ["h1", "h2"]

    async def test_dispatch_handler_error_doesnt_crash(self):
        q = EventQueue()

        async def bad_handler(event):
            raise ValueError("boom")

        q.register_handler(EventType.USER_MESSAGE, bad_handler)
        # Should not raise
        await q.dispatch(UserMessageEvent(message="test"))
