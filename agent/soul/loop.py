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
from agent.soul.personality import Personality, get_personality
from agent.soul.logging_utils import SoulLogger, configure_soul_logging
from agent.memory.emotional import EmotionalState

logger = logging.getLogger(__name__)
soul_log = SoulLogger("agent.soul.loop")


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
        personality: Optional[Personality] = None,
    ):
        """
        Initialize the soul loop.
        
        Args:
            config: Soul configuration
            reachy_service: ReachyService instance for sending commands
            on_movement: Callback for movement updates (receives pose dict)
            personality: Optional personality to use (loads from file if not provided)
        """
        self.config = config
        self._reachy_service = reachy_service
        self._on_movement = on_movement
        
        # Configure soul logging based on config
        configure_soul_logging(
            level=config.log_level if hasattr(config, 'log_level') else "INFO",
        )
        
        # Load personality
        if personality is not None:
            self._personality = personality
        else:
            self._personality = Personality.load(config.personality.soul_file_path)
        
        if self._personality.is_loaded():
            # Log personality loading with details
            active_traits = []
            if self._personality.traits.curious:
                active_traits.append("curious")
            if self._personality.traits.helpful:
                active_traits.append("helpful")
            if self._personality.traits.playful:
                active_traits.append("playful")
            if self._personality.traits.attentive:
                active_traits.append("attentive")
            if self._personality.traits.patient:
                active_traits.append("patient")
            
            soul_log.log_personality_loaded(
                name=self._personality.name,
                file_path=self._personality.file_path,
                traits=active_traits,
            )
            
            # Apply embodiment principles to config
            self._apply_personality_to_config()
        else:
            logger.warning("[PERSONALITY] load_failed: using defaults")
        
        # Subsystems (pass personality to emotion inference)
        self._emotion_inference = EmotionInference(config, self._personality)
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
        self._last_personality_reload = 0.0
        self._last_status_log_time = 0.0
        
        logger.info(f"[STATE] SoulLoop initialized: poll={config.poll_interval_ms}ms, emotion_inference={config.emotion_inference_interval_ms}ms")
    
    def _apply_personality_to_config(self):
        """Apply personality embodiment principles to config."""
        if not self._personality.is_loaded():
            return
        
        idle_config = self._personality.get_idle_behavior_config()
        
        # Override config with personality preferences
        if "breathing_enabled" in idle_config:
            old_val = self.config.idle_breathing_enabled
            self.config.idle_breathing_enabled = idle_config["breathing_enabled"]
            if self.config.log_decisions:
                soul_log.log_personality_applied("idle_breathing_enabled", idle_config["breathing_enabled"])
        
        if "micro_movements_enabled" in idle_config:
            self.config.idle_micro_movements_enabled = idle_config["micro_movements_enabled"]
            if self.config.log_decisions:
                soul_log.log_personality_applied("idle_micro_movements_enabled", idle_config["micro_movements_enabled"])
        
        if "scanning_enabled" in idle_config:
            self.config.idle_scanning_enabled = idle_config["scanning_enabled"]
            if self.config.log_decisions:
                soul_log.log_personality_applied("idle_scanning_enabled", idle_config["scanning_enabled"])
        
        logger.debug(f"[PERSONALITY] applied_config: {idle_config}")
    
    @property
    def personality(self) -> Optional[Personality]:
        """Get the loaded personality."""
        return self._personality
    
    def reload_personality(self) -> bool:
        """Reload personality from file."""
        if self._personality.reload():
            self._apply_personality_to_config()
            self._emotion_inference.set_personality(self._personality)
            soul_log.log_personality_reloaded()
            return True
        logger.warning("[PERSONALITY] reload_failed")
        return False
    
    async def start(self):
        """Start the soul loop as a background task."""
        if self._state.is_running:
            logger.warning("[STATE] start_rejected: soul loop already running")
            return
        
        self._stop_event.clear()
        self._state.is_running = True
        self._loop_start_time = time.time()
        self._last_status_log_time = time.time()
        
        self._task = asyncio.create_task(self._run_loop())
        
        # Log startup with config summary
        soul_log.log_loop_start(config_summary={
            "poll_ms": self.config.poll_interval_ms,
            "emotion_ms": self.config.emotion_inference_interval_ms,
            "personality": self._personality.name if self._personality.is_loaded() else "none",
            "breathing": self.config.idle_breathing_enabled,
            "scanning": self.config.idle_scanning_enabled,
        })
    
    async def stop(self):
        """Stop the soul loop."""
        if not self._state.is_running and self._task is None:
            return
        
        logger.info("[STATE] stopping: initiating graceful shutdown...")
        self._stop_event.set()
        
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("[STATE] stop_timeout: force cancelling task")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    logger.debug("[STATE] task_cancelled: cleanup complete")
            except asyncio.CancelledError:
                # If we're being cancelled ourselves, still try to cancel the task
                if self._task and not self._task.done():
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        pass
                raise  # Re-raise to propagate cancellation
        
        self._state.is_running = False
        self._task = None
        
        # Clean up
        try:
            await self._emotion_inference.close()
        except Exception as e:
            logger.debug(f"[STATE] cleanup_error: {e}")
        
        # Log shutdown with uptime
        uptime = time.time() - self._loop_start_time if self._loop_start_time > 0 else 0
        soul_log.log_loop_stop(uptime_s=uptime)
    
    async def _run_loop(self):
        """Main loop that runs continuously."""
        poll_interval = self.config.poll_interval_ms / 1000.0
        
        try:
            await self._run_loop_inner(poll_interval)
        except asyncio.CancelledError:
            logger.info("[STATE] loop_cancelled: received cancellation signal")
            raise  # Re-raise to properly propagate cancellation
        finally:
            # Ensure cleanup happens even on cancellation
            self._state.is_running = False
            logger.debug("[STATE] loop_exited: cleanup complete")
    
    async def _run_loop_inner(self, poll_interval: float):
        """Inner loop logic, separated for clean cancellation handling."""
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Check for personality reload if auto-reload enabled
                if self.config.personality.auto_reload:
                    await self._check_personality_reload()
                
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
                    if self.config.log_decisions:
                        soul_log.log_scan_start()
                
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
                
                # Periodic status logging
                if self.config.log_status_interval_s > 0:
                    now = time.time()
                    if now - self._last_status_log_time >= self.config.log_status_interval_s:
                        self._last_status_log_time = now
                        soul_log.log_status_summary(
                            emotion=self._state.current_emotion,
                            intensity=self._state.emotion_intensity,
                            user_state=self._state.user_state,
                            face_detected=self._state.face_detected,
                            is_responding=self._state.is_responding,
                            idle_duration_s=self._idle_generator.idle_duration_s,
                            event_queue_size=self._event_queue.size,
                            throttle_ms=0,  # Don't throttle, we manage interval ourselves
                        )
                
            except Exception as e:
                soul_log.log_loop_error(e, context="main_loop")
            
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
    
    async def _check_personality_reload(self):
        """Check if personality file needs reloading."""
        now = time.time()
        if now - self._last_personality_reload < self.config.personality.reload_interval_s:
            return
        
        self._last_personality_reload = now
        
        # Check file modification time
        import os
        try:
            mtime = os.path.getmtime(self._personality.file_path)
            if not hasattr(self, '_personality_mtime'):
                self._personality_mtime = mtime
            elif mtime > self._personality_mtime:
                logger.info("Personality file changed, reloading...")
                self.reload_personality()
                self._personality_mtime = mtime
        except OSError:
            pass
    
    async def _process_events(self):
        """Process all pending events from the queue."""
        while True:
            event = self._event_queue.pop_nowait()
            if event is None:
                break
            
            await self._handle_event(event)
    
    async def _handle_event(self, event: SoulEvent):
        """Handle a single event."""
        # Log event receipt
        if self.config.log_events:
            event_details = {}
            preview = None
            
            if hasattr(event, 'message'):
                preview = event.message
            elif hasattr(event, 'response'):
                preview = event.response
            elif hasattr(event, 'silence_duration_ms'):
                event_details['silence_ms'] = event.silence_duration_ms
            elif hasattr(event, 'position'):
                event_details['position'] = event.position
            elif hasattr(event, 'is_new_face'):
                event_details['is_new'] = event.is_new_face
            
            soul_log.log_event(
                event_type=event.event_type.value,
                details=event_details if event_details else None,
                preview=preview,
            )
        
        if event.event_type == EventType.USER_SPEAKING:
            old_state = self._state.user_state
            self._state.user_state = "speaking"
            self._state.last_interaction_time = time.time()
            self._idle_generator.reset_idle_timer()
            self._emotion_inference.update_context(user_state="speaking")
            
            # Immediate reaction: become attentive
            if self.config.user_speaking_triggers_attention:
                self._movement_blender.set_emotion("attentive", intensity=0.7)
                if self.config.log_decisions:
                    soul_log.log_decision(
                        decision_point="user_speaking_response",
                        decision="set attentive(0.7)",
                        reason="user_speaking_triggers_attention=True",
                    )
            
            if self.config.log_events and old_state != "speaking":
                soul_log.log_state_change("user_state", old_state, "speaking", reason="USER_SPEAKING event")
        
        elif event.event_type == EventType.USER_SILENT:
            old_state = self._state.user_state
            self._state.user_state = "waiting"
            self._emotion_inference.update_context(user_state="waiting")
            
            # Show thinking pose after short delay
            if self.config.user_silence_triggers_processing_look:
                self._movement_blender.set_emotion("thinking", intensity=0.5)
                if self.config.log_decisions:
                    soul_log.log_decision(
                        decision_point="user_silent_response",
                        decision="set thinking(0.5)",
                        reason="user_silence_triggers_processing_look=True",
                    )
            
            if self.config.log_events and old_state != "waiting":
                soul_log.log_state_change("user_state", old_state, "waiting", reason="USER_SILENT event")
        
        elif event.event_type == EventType.USER_MESSAGE:
            msg_event = event  # type: UserMessageEvent
            self._emotion_inference.update_context(user_message=msg_event.message)
            
            if self.config.log_events:
                soul_log.log_event_response(
                    event_type="USER_MESSAGE",
                    action_taken="updated emotion context",
                    details={"msg_len": len(msg_event.message)},
                )
        
        elif event.event_type == EventType.BOT_RESPONSE:
            resp_event = event  # type: BotResponseEvent
            was_responding = self._state.is_responding
            self._state.is_responding = False
            self._emotion_inference.update_context(bot_response=resp_event.response)
            
            if self.config.log_events and was_responding:
                soul_log.log_state_change("is_responding", True, False, reason="BOT_RESPONSE complete")
        
        elif event.event_type == EventType.BOT_STREAMING:
            was_responding = self._state.is_responding
            self._state.is_responding = True
            stream_event = event  # type: BotStreamingEvent
            
            if self.config.log_events and not was_responding:
                soul_log.log_state_change("is_responding", False, True, reason="BOT_STREAMING started")
        
        elif event.event_type == EventType.FACE_DETECTED:
            face_event = event  # type: FaceDetectedEvent
            was_detected = self._state.face_detected
            self._state.face_detected = True
            
            # Wave at new faces
            if face_event.is_new_face and self.config.antenna_wave_on_face_detected:
                await self._do_antenna_wave()
                if self.config.log_decisions:
                    soul_log.log_decision(
                        decision_point="face_greeting",
                        decision="antenna_wave",
                        reason="new_face + antenna_wave_on_face_detected=True",
                    )
            
            # Enable face tracking if configured
            if self.config.face_tracking_on_attention and not was_detected:
                await self._set_face_tracking(True)
                if self.config.log_decisions:
                    soul_log.log_decision(
                        decision_point="face_tracking",
                        decision="enable",
                        reason="face_detected + face_tracking_on_attention=True",
                    )
            
            if self.config.log_events and not was_detected:
                soul_log.log_state_change("face_detected", False, True, reason="FACE_DETECTED event")
        
        elif event.event_type == EventType.FACE_LOST:
            was_detected = self._state.face_detected
            self._state.face_detected = False
            
            if self.config.face_tracking_on_attention:
                await self._set_face_tracking(False)
                if self.config.log_decisions:
                    soul_log.log_decision(
                        decision_point="face_tracking",
                        decision="disable",
                        reason="face_lost + face_tracking_on_attention=True",
                    )
            
            if self.config.log_events and was_detected:
                soul_log.log_state_change("face_detected", True, False, reason="FACE_LOST event")
        
        # Dispatch to registered handlers
        await self._event_queue.dispatch(event)
    
    async def _update_emotion(self):
        """Update emotion through inference."""
        now = time.time()
        
        # Check if we should run LLM inference
        should_infer = (
            now - self._last_emotion_inference_time
        ) * 1000 >= self.config.emotion_inference_interval_ms
        
        old_emotion = self._state.current_emotion
        old_intensity = self._state.emotion_intensity
        
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
                
                if self.config.log_emotions and result.emotion != old_emotion:
                    soul_log.log_decision(
                        decision_point="emotion_update",
                        decision=f"{result.emotion}({result.intensity:.2f})",
                        reason="LLM inference",
                        context={"prev": f"{old_emotion}({old_intensity:.2f})"},
                    )
        else:
            # Use fast rule-based inference for immediate reactions
            result = self._emotion_inference.infer_rule_based()
            
            # Only apply if significantly different
            if result.emotion != self._state.current_emotion:
                self._movement_blender.set_emotion(result.emotion, result.intensity)
                self._state.current_emotion = result.emotion
                self._state.emotion_intensity = result.intensity
                
                if self.config.log_emotions:
                    soul_log.log_decision(
                        decision_point="emotion_update",
                        decision=f"{result.emotion}({result.intensity:.2f})",
                        reason="rule-based inference",
                        context={"prev": f"{old_emotion}({old_intensity:.2f})"},
                    )
    
    async def _send_movement(self):
        """Send current pose to Reachy."""
        pose_dict = self._movement_blender.to_reachy_command()
        
        # Log movement if enabled (throttled by default)
        if self.config.log_movements:
            soul_log.log_movement_command(pose_dict, throttle_ms=5000)
        
        # Callback for external handling
        if self._on_movement:
            try:
                self._on_movement(pose_dict)
            except Exception as e:
                logger.error(f"[MOVEMENT] callback_error: {e}")
        
        # Direct service call if available
        if self._reachy_service:
            try:
                # The service should have a method to apply pose
                if hasattr(self._reachy_service, 'apply_soul_pose'):
                    self._reachy_service.apply_soul_pose(pose_dict)
            except Exception as e:
                if self.config.debug_logging:
                    logger.debug(f"[MOVEMENT] send_failed: {e}")
    
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
