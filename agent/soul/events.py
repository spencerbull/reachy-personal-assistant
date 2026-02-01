"""
Soul System Events.

Defines event types for conversation state changes that the soul loop responds to.
These events are fired by the Pipecat pipeline and consumed by the soul loop.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from enum import Enum


class EventType(Enum):
    """Types of events the soul loop can receive."""
    USER_SPEAKING = "user_speaking"
    USER_SILENT = "user_silent"
    USER_MESSAGE = "user_message"
    BOT_RESPONSE = "bot_response"
    BOT_STREAMING = "bot_streaming"
    FACE_DETECTED = "face_detected"
    FACE_LOST = "face_lost"
    SOUND_DETECTED = "sound_detected"
    IDLE_TICK = "idle_tick"


@dataclass
class SoulEvent:
    """Base class for soul events."""
    event_type: EventType
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __str__(self) -> str:
        return f"{self.event_type.value}@{self.timestamp:.3f}"


@dataclass
class UserSpeakingEvent(SoulEvent):
    """User has started speaking (VAD detected voice)."""
    event_type: EventType = field(default=EventType.USER_SPEAKING)


@dataclass
class UserSilentEvent(SoulEvent):
    """User has stopped speaking."""
    event_type: EventType = field(default=EventType.USER_SILENT)
    silence_duration_ms: int = 0  # How long they've been silent


@dataclass
class UserMessageEvent(SoulEvent):
    """User message has been transcribed (STT complete)."""
    event_type: EventType = field(default=EventType.USER_MESSAGE)
    message: str = ""
    
    def __str__(self) -> str:
        preview = self.message[:50] + "..." if len(self.message) > 50 else self.message
        return f"UserMessage: '{preview}'"


@dataclass
class BotResponseEvent(SoulEvent):
    """Bot has completed a response."""
    event_type: EventType = field(default=EventType.BOT_RESPONSE)
    response: str = ""
    
    def __str__(self) -> str:
        preview = self.response[:50] + "..." if len(self.response) > 50 else self.response
        return f"BotResponse: '{preview}'"


@dataclass
class BotStreamingEvent(SoulEvent):
    """Bot is streaming response tokens (for reactive expressions)."""
    event_type: EventType = field(default=EventType.BOT_STREAMING)
    token: str = ""
    accumulated_text: str = ""  # Full text so far
    
    def __str__(self) -> str:
        return f"BotStreaming: +'{self.token}' (total: {len(self.accumulated_text)} chars)"


@dataclass
class FaceDetectedEvent(SoulEvent):
    """A face has been detected in camera view."""
    event_type: EventType = field(default=EventType.FACE_DETECTED)
    face_id: Optional[str] = None  # For tracking specific faces
    position: tuple[float, float] = (0.5, 0.5)  # Normalized (0-1) position in frame
    is_new_face: bool = False  # True if this face wasn't tracked before
    
    def __str__(self) -> str:
        new_str = " (NEW)" if self.is_new_face else ""
        return f"FaceDetected{new_str}: pos={self.position}"


@dataclass
class FaceLostEvent(SoulEvent):
    """A tracked face has left the camera view."""
    event_type: EventType = field(default=EventType.FACE_LOST)
    face_id: Optional[str] = None
    
    def __str__(self) -> str:
        return f"FaceLost: {self.face_id or 'unknown'}"


@dataclass
class SoundDetectedEvent(SoulEvent):
    """A loud or sudden sound was detected."""
    event_type: EventType = field(default=EventType.SOUND_DETECTED)
    direction: Optional[str] = None  # "left", "right", "front", "behind" if determinable
    intensity: float = 0.5  # 0-1 loudness
    
    def __str__(self) -> str:
        dir_str = f" from {self.direction}" if self.direction else ""
        return f"SoundDetected{dir_str}: intensity={self.intensity:.2f}"


@dataclass
class IdleTickEvent(SoulEvent):
    """Periodic tick when nothing else is happening."""
    event_type: EventType = field(default=EventType.IDLE_TICK)
    idle_duration_s: float = 0.0  # How long we've been idle
    
    def __str__(self) -> str:
        return f"IdleTick: {self.idle_duration_s:.1f}s"


# Convenience type for event handlers
EventHandler = callable  # Callable[[SoulEvent], None]


class EventQueue:
    """
    Thread-safe event queue for the soul loop.
    
    Events are pushed by Pipecat processors and consumed by the soul loop.
    """
    
    def __init__(self, max_size: int = 100):
        import asyncio
        self._queue: asyncio.Queue[SoulEvent] = asyncio.Queue(maxsize=max_size)
        self._handlers: dict[EventType, list[EventHandler]] = {}
    
    async def push(self, event: SoulEvent):
        """Add an event to the queue."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass
    
    async def pop(self, timeout: Optional[float] = None) -> Optional[SoulEvent]:
        """Get the next event from the queue."""
        import asyncio
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout)
            else:
                return await self._queue.get()
        except asyncio.TimeoutError:
            return None
    
    def pop_nowait(self) -> Optional[SoulEvent]:
        """Get the next event without waiting."""
        try:
            return self._queue.get_nowait()
        except:
            return None
    
    def register_handler(self, event_type: EventType, handler: EventHandler):
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def dispatch(self, event: SoulEvent):
        """Dispatch an event to all registered handlers."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if hasattr(result, '__await__'):
                    await result
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Event handler error: {e}")
    
    @property
    def size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()
    
    def clear(self):
        """Clear all pending events."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except:
                break
