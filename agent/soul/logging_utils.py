"""
Soul System Logging Utilities.

Provides structured, readable logging for the soul system to help
understand what decisions are being made and why.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
from functools import wraps


class SoulLogCategory(Enum):
    """Categories for soul log messages."""
    EMOTION = "EMOTION"
    EVENT = "EVENT"
    MOVEMENT = "MOVEMENT"
    PERSONALITY = "PERSONALITY"
    IDLE = "IDLE"
    DECISION = "DECISION"
    STATE = "STATE"
    LLM = "LLM"


@dataclass
class SoulLogContext:
    """Context for a soul log message."""
    category: SoulLogCategory
    action: str
    details: Optional[dict] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: Optional[str] = None
    duration_ms: Optional[float] = None


class SoulLogger:
    """
    Structured logger for the soul system.
    
    Provides readable, categorized logging that makes it easy to understand
    what the soul is doing and why.
    
    Usage:
        soul_logger = SoulLogger("soul.loop")
        
        soul_logger.log_emotion_change("neutral", "happy", 
            reason="User said 'thanks'", intensity=0.7)
        
        soul_logger.log_event("USER_SPEAKING", 
            details={"user_state": "speaking"})
        
        soul_logger.log_decision("set_pose", 
            decision="attentive",
            reason="User speaking triggers attention")
    """
    
    def __init__(self, name: str, config: Optional[Any] = None):
        self.logger = logging.getLogger(name)
        self.config = config
        self._last_log_time: dict[str, float] = {}
        self._start_time = time.time()
    
    def _format_context(self, ctx: SoulLogContext) -> str:
        """Format a log context into a readable string."""
        parts = [f"[{ctx.category.value}]", ctx.action]
        
        if ctx.old_value is not None and ctx.new_value is not None:
            parts.append(f"({ctx.old_value} → {ctx.new_value})")
        elif ctx.new_value is not None:
            parts.append(f"= {ctx.new_value}")
        
        if ctx.reason:
            parts.append(f"| reason: {ctx.reason}")
        
        if ctx.duration_ms is not None:
            parts.append(f"[{ctx.duration_ms:.1f}ms]")
        
        if ctx.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in ctx.details.items())
            parts.append(f"{{ {detail_str} }}")
        
        return " ".join(parts)
    
    def _should_log(self, category: SoulLogCategory, throttle_key: Optional[str] = None, 
                    throttle_ms: float = 0) -> bool:
        """Check if we should log (with optional throttling)."""
        if throttle_key and throttle_ms > 0:
            now = time.time()
            last = self._last_log_time.get(throttle_key, 0)
            if (now - last) * 1000 < throttle_ms:
                return False
            self._last_log_time[throttle_key] = now
        return True
    
    def _uptime(self) -> str:
        """Get formatted uptime string."""
        elapsed = time.time() - self._start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}m"
        else:
            return f"{elapsed/3600:.1f}h"
    
    # -------------------------------------------------------------------------
    # Emotion Logging
    # -------------------------------------------------------------------------
    
    def log_emotion_change(
        self,
        old_emotion: str,
        new_emotion: str,
        intensity: float = 0.5,
        reason: Optional[str] = None,
        source: str = "inferred",
        inference_time_ms: Optional[float] = None,
    ):
        """Log an emotion state change."""
        if old_emotion == new_emotion:
            return
        
        ctx = SoulLogContext(
            category=SoulLogCategory.EMOTION,
            action=f"emotion_change ({source})",
            old_value=old_emotion,
            new_value=f"{new_emotion}({intensity:.2f})",
            reason=reason,
            duration_ms=inference_time_ms,
        )
        self.logger.info(self._format_context(ctx))
    
    def log_emotion_inference(
        self,
        emotion: str,
        intensity: float,
        confidence: float = 1.0,
        movement_hint: Optional[str] = None,
        inference_time_ms: float = 0,
        prompt_preview: Optional[str] = None,
    ):
        """Log LLM emotion inference result."""
        details = {
            "intensity": f"{intensity:.2f}",
            "confidence": f"{confidence:.2f}",
        }
        if movement_hint:
            details["hint"] = movement_hint
        if prompt_preview:
            details["input"] = prompt_preview[:50] + "..." if len(prompt_preview) > 50 else prompt_preview
        
        ctx = SoulLogContext(
            category=SoulLogCategory.LLM,
            action="emotion_inference",
            new_value=emotion,
            details=details,
            duration_ms=inference_time_ms,
        )
        self.logger.info(self._format_context(ctx))
    
    def log_emotion_rule_based(
        self,
        emotion: str,
        intensity: float,
        rule: str,
    ):
        """Log rule-based emotion determination."""
        ctx = SoulLogContext(
            category=SoulLogCategory.EMOTION,
            action="rule_based",
            new_value=f"{emotion}({intensity:.2f})",
            reason=rule,
        )
        self.logger.debug(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # Event Logging
    # -------------------------------------------------------------------------
    
    def log_event(
        self,
        event_type: str,
        details: Optional[dict] = None,
        preview: Optional[str] = None,
    ):
        """Log a soul event being processed."""
        if preview:
            details = details or {}
            details["preview"] = preview[:40] + "..." if len(preview) > 40 else preview
        
        ctx = SoulLogContext(
            category=SoulLogCategory.EVENT,
            action=f"received: {event_type}",
            details=details,
        )
        self.logger.info(self._format_context(ctx))
    
    def log_event_response(
        self,
        event_type: str,
        action_taken: str,
        details: Optional[dict] = None,
    ):
        """Log the response to an event."""
        ctx = SoulLogContext(
            category=SoulLogCategory.EVENT,
            action=f"response: {event_type}",
            new_value=action_taken,
            details=details,
        )
        self.logger.debug(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # State Logging
    # -------------------------------------------------------------------------
    
    def log_state_change(
        self,
        field: str,
        old_value: Any,
        new_value: Any,
        reason: Optional[str] = None,
    ):
        """Log a state field change."""
        if old_value == new_value:
            return
        
        ctx = SoulLogContext(
            category=SoulLogCategory.STATE,
            action=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        self.logger.info(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # Movement Logging
    # -------------------------------------------------------------------------
    
    def log_pose_blend_start(
        self,
        target_emotion: str,
        intensity: float,
        duration_s: float,
    ):
        """Log the start of a pose blend."""
        ctx = SoulLogContext(
            category=SoulLogCategory.MOVEMENT,
            action="blend_start",
            new_value=f"{target_emotion}({intensity:.2f})",
            details={"duration": f"{duration_s:.2f}s"},
        )
        self.logger.debug(self._format_context(ctx))
    
    def log_pose_blend_complete(
        self,
        emotion: str,
    ):
        """Log completion of a pose blend."""
        ctx = SoulLogContext(
            category=SoulLogCategory.MOVEMENT,
            action="blend_complete",
            new_value=emotion,
        )
        self.logger.debug(self._format_context(ctx))
    
    def log_movement_command(
        self,
        pose_dict: dict,
        throttle_ms: float = 5000,  # Only log every 5s by default
    ):
        """Log a movement command being sent (throttled)."""
        if not self._should_log(SoulLogCategory.MOVEMENT, "movement_cmd", throttle_ms):
            return
        
        # Summarize pose
        summary = {
            "pitch": f"{pose_dict.get('head_pitch', 0):.1f}°",
            "yaw": f"{pose_dict.get('head_yaw', 0):.1f}°",
            "ant": f"[{pose_dict.get('left_antenna', 0):.2f}, {pose_dict.get('right_antenna', 0):.2f}]",
        }
        
        ctx = SoulLogContext(
            category=SoulLogCategory.MOVEMENT,
            action="send_pose",
            details=summary,
        )
        self.logger.debug(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # Idle Behavior Logging
    # -------------------------------------------------------------------------
    
    def log_idle_action(
        self,
        action: str,
        details: Optional[dict] = None,
    ):
        """Log idle behavior actions."""
        ctx = SoulLogContext(
            category=SoulLogCategory.IDLE,
            action=action,
            details=details,
        )
        self.logger.debug(self._format_context(ctx))
    
    def log_scan_start(self):
        """Log start of idle room scan."""
        ctx = SoulLogContext(
            category=SoulLogCategory.IDLE,
            action="scan_start",
            reason="idle timeout reached",
        )
        self.logger.info(self._format_context(ctx))
    
    def log_scan_complete(self):
        """Log completion of idle room scan."""
        ctx = SoulLogContext(
            category=SoulLogCategory.IDLE,
            action="scan_complete",
        )
        self.logger.debug(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # Personality Logging
    # -------------------------------------------------------------------------
    
    def log_personality_loaded(
        self,
        name: str,
        file_path: str,
        traits: list[str],
    ):
        """Log personality loading."""
        ctx = SoulLogContext(
            category=SoulLogCategory.PERSONALITY,
            action="loaded",
            new_value=name,
            details={
                "path": file_path,
                "traits": ", ".join(traits),
            },
        )
        self.logger.info(self._format_context(ctx))
    
    def log_personality_applied(
        self,
        setting: str,
        value: Any,
    ):
        """Log personality setting being applied."""
        ctx = SoulLogContext(
            category=SoulLogCategory.PERSONALITY,
            action=f"apply: {setting}",
            new_value=value,
        )
        self.logger.debug(self._format_context(ctx))
    
    def log_personality_reloaded(self):
        """Log personality file reload."""
        ctx = SoulLogContext(
            category=SoulLogCategory.PERSONALITY,
            action="reloaded",
            reason="file changed",
        )
        self.logger.info(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # Decision Logging
    # -------------------------------------------------------------------------
    
    def log_decision(
        self,
        decision_point: str,
        decision: str,
        reason: Optional[str] = None,
        alternatives: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ):
        """Log a decision made by the soul system."""
        details = context or {}
        if alternatives:
            details["alternatives"] = alternatives
        
        ctx = SoulLogContext(
            category=SoulLogCategory.DECISION,
            action=decision_point,
            new_value=decision,
            reason=reason,
            details=details if details else None,
        )
        self.logger.info(self._format_context(ctx))
    
    # -------------------------------------------------------------------------
    # Lifecycle Logging
    # -------------------------------------------------------------------------
    
    def log_loop_start(self, config_summary: Optional[dict] = None):
        """Log soul loop starting."""
        self._start_time = time.time()
        details = config_summary or {}
        ctx = SoulLogContext(
            category=SoulLogCategory.STATE,
            action="loop_start",
            details=details,
        )
        self.logger.info(self._format_context(ctx))
    
    def log_loop_stop(self, uptime_s: Optional[float] = None):
        """Log soul loop stopping."""
        uptime = uptime_s or (time.time() - self._start_time)
        ctx = SoulLogContext(
            category=SoulLogCategory.STATE,
            action="loop_stop",
            details={"uptime": f"{uptime:.1f}s"},
        )
        self.logger.info(self._format_context(ctx))
    
    def log_loop_error(self, error: Exception, context: Optional[str] = None):
        """Log a soul loop error."""
        ctx = SoulLogContext(
            category=SoulLogCategory.STATE,
            action="error",
            new_value=str(error),
            reason=context,
        )
        self.logger.error(self._format_context(ctx), exc_info=True)
    
    # -------------------------------------------------------------------------
    # Status Summary
    # -------------------------------------------------------------------------
    
    def log_status_summary(
        self,
        emotion: str,
        intensity: float,
        user_state: str,
        face_detected: bool,
        is_responding: bool,
        idle_duration_s: float,
        event_queue_size: int,
        throttle_ms: float = 10000,  # Log status every 10s max
    ):
        """Log a periodic status summary (throttled)."""
        if not self._should_log(SoulLogCategory.STATE, "status_summary", throttle_ms):
            return
        
        details = {
            "emotion": f"{emotion}({intensity:.2f})",
            "user": user_state,
            "face": "yes" if face_detected else "no",
            "responding": "yes" if is_responding else "no",
            "idle": f"{idle_duration_s:.1f}s",
            "queue": event_queue_size,
            "uptime": self._uptime(),
        }
        
        ctx = SoulLogContext(
            category=SoulLogCategory.STATE,
            action="status",
            details=details,
        )
        self.logger.info(self._format_context(ctx))


# -----------------------------------------------------------------------------
# Convenience function for timing blocks
# -----------------------------------------------------------------------------

class LogTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: SoulLogger, action: str, category: SoulLogCategory = SoulLogCategory.LLM):
        self.logger = logger
        self.action = action
        self.category = category
        self.start_time = 0.0
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        return False


# -----------------------------------------------------------------------------
# Global soul logger instance
# -----------------------------------------------------------------------------

_soul_logger: Optional[SoulLogger] = None


def get_soul_logger(name: str = "agent.soul") -> SoulLogger:
    """Get the global soul logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = SoulLogger(name)
    return _soul_logger


def configure_soul_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    enable_file_logging: bool = False,
    log_file_path: str = "soul.log",
):
    """
    Configure soul system logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string (uses default if None)
        enable_file_logging: Whether to also log to a file
        log_file_path: Path to log file if file logging enabled
    """
    # Default format that's readable but informative
    if format_string is None:
        format_string = "%(asctime)s.%(msecs)03d [%(name)s] %(message)s"
    
    # Configure the soul logger
    soul_logger = logging.getLogger("agent.soul")
    soul_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    if not soul_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(format_string, datefmt="%H:%M:%S"))
        soul_logger.addHandler(console_handler)
    
    # File handler (optional)
    if enable_file_logging:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s"
        ))
        soul_logger.addHandler(file_handler)
