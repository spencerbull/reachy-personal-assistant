"""
Soul Loop - Main Embodiment Loop.

The soul loop runs continuously alongside the main agent, managing:
- Emotion inference from conversation
- Movement blending for smooth transitions
- Idle behaviors for lifelike presence
- Event-driven reactions

The loop runs at a configurable poll interval (default 200ms) and
coordinates all soul subsystems.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass

from agent.soul.config import SoulConfig
from agent.soul.events import (
    EventQueue,
    SoulEvent,
    EventType,
    UserSpeakingEvent,
    UserSilentEvent,
    UserMessageEvent,
    BotResponseEvent,
    BotStreamingEvent,
    FaceDetectedEvent,
    FaceLostEvent,
    IdleTickEvent,
)
from agent.soul.emotion_inference import EmotionInference, EmotionInferenceResult
from agent.soul.movement_blender import MovementBlender, Pose
from agent.soul.idle_generator import IdleGenerator
from agent.memory.emotional import EmotionalState

logger = logging.getLogger(__name__)


@dataclass
class SoulState:
    """Current state of the soul loop."""
    is_running: bool = False
    current_emotion: EmotionalState = "neutral"
    emotion_intensity: float = 0.5
    user_state: str = "idle"  # idle, speaking, waiting
    last_interaction_time: float = 0.0
    face_detected: bool = False
    is_responding: bool = False


class SoulLoop:
    """
    Main embodiment loop for Reachy's continuous presence.
    
    Runs as an async task alongside the main agent pipeline,
    coordinating emotion inference, movement blending, and idle behaviors.
    
    Usage:
        config = SoulConfig()
        soul = SoulLoop(config, reachy_service)
        
        await soul.start()
        
        # Events are fed from Pipecat pipeline
        soul.on_user_speaking()
        soul.on_user_message("Hello!")
        
        await soul.stop()
    """
    
    def __init__(
        self,
        config: SoulConfig,
        reachy_service: Optional[Any] = None,
        on_movement: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize the soul loop.
        
        Args:
            config: Soul configuration
            reachy_service: ReachyService instance for sending commands
            on_movement: Callback for movement updates (receives pose dict)
        """
        self.config = config
        self._reachy_service = reachy_service
        self._on_movement = on_movement
        
        # Subsystems
        self._emotion_inference = EmotionInference(config)
        self._movement_blender = MovementBlender(config)
        self._idle_generator = IdleGenerator(config)
        
        # Event queue
        self._event_queue = EventQueue()
        
        # State
        self._state = SoulState()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Time tracking
        self._loop_start_time = 0.0
        self._last_emotion_inference_time = 0.0
        
        logger.info("SoulLoop initialized")
    
    async def start(self):
        """Start the soul loop as a background task."""
        if self._state.is_running:
            logger.warning("Soul loop already running")
            return
        
        self._stop_event.clear()
        self._state.is_running = True
        self._loop_start_time = time.time()
        
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Soul loop started")
    
    async def stop(self):
        """Stop the soul loop."""
        if not self._state.is_running:
            return
        
        logger.info("Stopping soul loop...")
        self._stop_event.set()
        
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        
        self._state.is_running = False
        
        # Clean up
        await self._emotion_inference.close()
        
        logger.info("Soul loop stopped")
    
    async def _run_loop(self):
        """Main loop that runs continuously."""
        poll_interval = self.config.poll_interval_ms / 1000.0
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Process any pending events
                await self._process_events()
                
                # Run emotion inference (rate limited internally)
                await self._update_emotion()
                
                # Update movement blender
                dt = time.time() - self._movement_blender._last_update_time
                pose = self._movement_blender.update(dt)
                
                # Generate and apply idle behaviors
                if not self._state.is_responding:
                    idle_overlay = self._idle_generator.update(dt)
                    self._movement_blender.add_overlay(idle_overlay)
                
                # Check for idle scanning
                if self._idle_generator.should_scan():
                    self._idle_generator.start_scan()
                
                # Handle active scanning
                if self._idle_generator.is_scanning:
                    scan_target = self._idle_generator.get_scan_target()
                    if scan_target:
                        yaw, pitch = scan_target
                        scan_overlay = Pose(head_yaw=yaw, head_pitch=pitch)
                        self._movement_blender.add_overlay(scan_overlay, weight=0.5)
                
                # Send movement command
                await self._send_movement()
                
                # Generate idle tick event if appropriate
                idle_duration = self._idle_generator.idle_duration_s
                if idle_duration > 5.0:
                    await self._event_queue.push(IdleTickEvent(idle_duration_s=idle_duration))
                
            except Exception as e:
                logger.error(f"Soul loop error: {e}", exc_info=True)
            
            # Sleep for remainder of poll interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, poll_interval - elapsed)
            
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=sleep_time
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Normal loop continuation
    
    async def _process_events(self):
        """Process all pending events from the queue."""
        while True:
            event = self._event_queue.pop_nowait()
            if event is None:
                break
            
            await self._handle_event(event)
    
    async def _handle_event(self, event: SoulEvent):
        """Handle a single event."""
        if self.config.debug_logging:
            logger.debug(f"Soul event: {event}")
        
        if event.event_type == EventType.USER_SPEAKING:
            self._state.user_state = "speaking"
            self._state.last_interaction_time = time.time()
            self._idle_generator.reset_idle_timer()
            self._emotion_inference.update_context(user_state="speaking")
            
            # Immediate reaction: become attentive
            if self.config.user_speaking_triggers_attention:
                self._movement_blender.set_emotion("attentive", intensity=0.7)
        
        elif event.event_type == EventType.USER_SILENT:
            self._state.user_state = "waiting"
            self._emotion_inference.update_context(user_state="waiting")
            
            # Show thinking pose after short delay
            if self.config.user_silence_triggers_processing_look:
                self._movement_blender.set_emotion("thinking", intensity=0.5)
        
        elif event.event_type == EventType.USER_MESSAGE:
            msg_event = event  # type: UserMessageEvent
            self._emotion_inference.update_context(user_message=msg_event.message)
        
        elif event.event_type == EventType.BOT_RESPONSE:
            resp_event = event  # type: BotResponseEvent
            self._state.is_responding = False
            self._emotion_inference.update_context(bot_response=resp_event.response)
        
        elif event.event_type == EventType.BOT_STREAMING:
            self._state.is_responding = True
            stream_event = event  # type: BotStreamingEvent
            # Could update emotion based on streaming content here
        
        elif event.event_type == EventType.FACE_DETECTED:
            face_event = event  # type: FaceDetectedEvent
            was_detected = self._state.face_detected
            self._state.face_detected = True
            
            # Wave at new faces
            if face_event.is_new_face and self.config.antenna_wave_on_face_detected:
                await self._do_antenna_wave()
            
            # Enable face tracking if configured
            if self.config.face_tracking_on_attention and not was_detected:
                await self._set_face_tracking(True)
        
        elif event.event_type == EventType.FACE_LOST:
            self._state.face_detected = False
            
            if self.config.face_tracking_on_attention:
                await self._set_face_tracking(False)
        
        # Dispatch to registered handlers
        await self._event_queue.dispatch(event)
    
    async def _update_emotion(self):
        """Update emotion through inference."""
        now = time.time()
        
        # Check if we should run LLM inference
        should_infer = (
            now - self._last_emotion_inference_time
        ) * 1000 >= self.config.emotion_inference_interval_ms
        
        if should_infer:
            # Run async emotion inference
            result = await self._emotion_inference.infer()
            self._last_emotion_inference_time = now
            
            # Update movement blender with new emotion
            if result.emotion != self._state.current_emotion or \
               abs(result.intensity - self._state.emotion_intensity) > 0.1:
                self._movement_blender.set_emotion(result.emotion, result.intensity)
                self._state.current_emotion = result.emotion
                self._state.emotion_intensity = result.intensity
        else:
            # Use fast rule-based inference for immediate reactions
            result = self._emotion_inference.infer_rule_based()
            
            # Only apply if significantly different
            if result.emotion != self._state.current_emotion:
                self._movement_blender.set_emotion(result.emotion, result.intensity)
                self._state.current_emotion = result.emotion
                self._state.emotion_intensity = result.intensity
    
    async def _send_movement(self):
        """Send current pose to Reachy."""
        pose_dict = self._movement_blender.to_reachy_command()
        
        # Callback for external handling
        if self._on_movement:
            try:
                self._on_movement(pose_dict)
            except Exception as e:
                logger.error(f"Movement callback error: {e}")
        
        # Direct service call if available
        if self._reachy_service:
            try:
                # The service should have a method to apply pose
                if hasattr(self._reachy_service, 'apply_soul_pose'):
                    self._reachy_service.apply_soul_pose(pose_dict)
            except Exception as e:
                if self.config.debug_logging:
                    logger.debug(f"Could not send pose to Reachy: {e}")
    
    async def _do_antenna_wave(self):
        """Perform an antenna wave greeting."""
        if self._reachy_service and hasattr(self._reachy_service, 'antenna_wave'):
            try:
                self._reachy_service.antenna_wave()
            except Exception as e:
                logger.debug(f"Antenna wave failed: {e}")
    
    async def _set_face_tracking(self, enabled: bool):
        """Enable or disable face tracking."""
        if self._reachy_service and hasattr(self._reachy_service, 'set_face_tracking'):
            try:
                self._reachy_service.set_face_tracking(enabled)
            except Exception as e:
                logger.debug(f"Set face tracking failed: {e}")
    
    # Public API for feeding events from Pipecat
    
    def on_user_speaking(self):
        """Call when user starts speaking (VAD triggers)."""
        asyncio.create_task(self._event_queue.push(UserSpeakingEvent()))
    
    def on_user_silent(self, silence_duration_ms: int = 0):
        """Call when user stops speaking."""
        asyncio.create_task(self._event_queue.push(
            UserSilentEvent(silence_duration_ms=silence_duration_ms)
        ))
    
    def on_user_message(self, message: str):
        """Call when user message is transcribed."""
        asyncio.create_task(self._event_queue.push(
            UserMessageEvent(message=message)
        ))
    
    def on_bot_response(self, response: str):
        """Call when bot completes a response."""
        asyncio.create_task(self._event_queue.push(
            BotResponseEvent(response=response)
        ))
    
    def on_bot_streaming(self, token: str, accumulated: str = ""):
        """Call for each streaming token from bot response."""
        asyncio.create_task(self._event_queue.push(
            BotStreamingEvent(token=token, accumulated_text=accumulated)
        ))
    
    def on_face_detected(
        self,
        position: tuple[float, float] = (0.5, 0.5),
        is_new: bool = False,
        face_id: Optional[str] = None,
    ):
        """Call when a face is detected."""
        asyncio.create_task(self._event_queue.push(
            FaceDetectedEvent(
                position=position,
                is_new_face=is_new,
                face_id=face_id,
            )
        ))
    
    def on_face_lost(self, face_id: Optional[str] = None):
        """Call when a tracked face is lost."""
        asyncio.create_task(self._event_queue.push(
            FaceLostEvent(face_id=face_id)
        ))
    
    # Properties
    
    @property
    def is_running(self) -> bool:
        """Check if the soul loop is running."""
        return self._state.is_running
    
    @property
    def current_emotion(self) -> EmotionalState:
        """Get the current emotional state."""
        return self._state.current_emotion
    
    @property
    def emotion_intensity(self) -> float:
        """Get the current emotion intensity."""
        return self._state.emotion_intensity
    
    @property
    def state(self) -> SoulState:
        """Get the full soul state."""
        return self._state
    
    def get_status(self) -> dict:
        """Get soul loop status for debugging."""
        return {
            "is_running": self._state.is_running,
            "current_emotion": self._state.current_emotion,
            "emotion_intensity": self._state.emotion_intensity,
            "user_state": self._state.user_state,
            "face_detected": self._state.face_detected,
            "is_responding": self._state.is_responding,
            "idle_duration_s": self._idle_generator.idle_duration_s,
            "is_scanning": self._idle_generator.is_scanning,
            "event_queue_size": self._event_queue.size,
        }
