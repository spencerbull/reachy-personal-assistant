"""
Emotional State Manager for Reachy.

Manages Reachy's emotional state and triggers appropriate physical expressions
through movements, antenna positions, and behaviors.

Emotional States:
- neutral: Default calm state
- happy: Positive, upbeat (antenna wiggle, slight bounce)
- curious: Interested, engaged (head tilt, raised antennas)
- attentive: Focused, listening (forward lean, steady gaze)
- thinking: Processing, contemplating (look up, pause)
- excited: Very positive, energetic (bouncy, fast antenna)
- playful: Fun, casual (quick movements, dance-like)
- helpful: Ready to assist (forward, engaged)
"""

import logging
from typing import Optional, Callable, Literal
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Emotional state type
EmotionalState = Literal[
    "neutral",
    "happy",
    "curious", 
    "attentive",
    "thinking",
    "excited",
    "playful",
    "helpful",
]


@dataclass
class EmotionExpression:
    """Physical expression parameters for an emotion."""
    
    # Antenna positions (radians)
    left_antenna: float = 0.0
    right_antenna: float = 0.0
    
    # Head adjustments (radians)
    head_pitch_offset: float = 0.0  # Positive = look down
    head_yaw_offset: float = 0.0    # Positive = look right
    head_roll_offset: float = 0.0   # Head tilt
    
    # Animation
    antenna_wiggle: bool = False
    antenna_wiggle_speed: float = 1.0
    bounce: bool = False
    bounce_amplitude: float = 0.005  # meters
    
    # Face tracking behavior
    enable_face_tracking: bool = False
    
    # Duration modifiers
    speaking_speed_factor: float = 1.0


# Emotion to expression mappings
EMOTION_EXPRESSIONS: dict[EmotionalState, EmotionExpression] = {
    "neutral": EmotionExpression(
        left_antenna=0.0,
        right_antenna=0.0,
    ),
    
    "happy": EmotionExpression(
        left_antenna=0.15,   # ~8 degrees up
        right_antenna=0.15,
        antenna_wiggle=True,
        antenna_wiggle_speed=2.0,
        bounce=True,
        bounce_amplitude=0.008,
        speaking_speed_factor=1.1,
    ),
    
    "curious": EmotionExpression(
        left_antenna=0.2,    # ~11 degrees up
        right_antenna=0.2,
        head_roll_offset=0.1,  # Slight head tilt
        head_pitch_offset=-0.05,  # Slight look up
    ),
    
    "attentive": EmotionExpression(
        left_antenna=0.1,
        right_antenna=0.1,
        enable_face_tracking=True,
        head_pitch_offset=-0.02,  # Very slight forward lean
    ),
    
    "thinking": EmotionExpression(
        left_antenna=0.05,
        right_antenna=0.05,
        head_pitch_offset=-0.15,  # Look up
        head_yaw_offset=0.1,      # Look slightly to side
        speaking_speed_factor=0.9,
    ),
    
    "excited": EmotionExpression(
        left_antenna=0.25,
        right_antenna=0.25,
        antenna_wiggle=True,
        antenna_wiggle_speed=3.0,
        bounce=True,
        bounce_amplitude=0.012,
        speaking_speed_factor=1.2,
    ),
    
    "playful": EmotionExpression(
        left_antenna=0.2,
        right_antenna=-0.1,  # Asymmetric for playful look
        antenna_wiggle=True,
        antenna_wiggle_speed=1.5,
        head_roll_offset=0.08,
    ),
    
    "helpful": EmotionExpression(
        left_antenna=0.1,
        right_antenna=0.1,
        head_pitch_offset=-0.03,  # Slight forward
        enable_face_tracking=True,
    ),
}


class EmotionalStateManager:
    """
    Manages Reachy's emotional state and expressions.
    
    Tracks the current emotional state and provides parameters for
    physical expression. Can be connected to the movement manager
    for automatic expression updates.
    """
    
    def __init__(
        self,
        on_state_change: Optional[Callable[[EmotionalState, EmotionExpression], None]] = None
    ):
        """
        Initialize emotional state manager.
        
        Args:
            on_state_change: Callback when emotional state changes
        """
        self._current_state: EmotionalState = "neutral"
        self._current_expression = EMOTION_EXPRESSIONS["neutral"]
        self._on_state_change = on_state_change
        
        self._last_change_time = time.time()
        self._state_history: list[tuple[EmotionalState, float]] = []
        self._max_history = 10
        
        # Auto-decay settings
        self._auto_decay = True
        self._decay_to = "neutral"
        self._decay_after_seconds = 30.0
    
    @property
    def current_state(self) -> EmotionalState:
        """Get the current emotional state."""
        return self._current_state
    
    @property
    def current_expression(self) -> EmotionExpression:
        """Get the current expression parameters."""
        return self._current_expression
    
    def set_state(self, state: EmotionalState):
        """
        Set the emotional state.
        
        Args:
            state: The new emotional state
        """
        if state not in EMOTION_EXPRESSIONS:
            logger.warning(f"Unknown emotional state: {state}, using neutral")
            state = "neutral"
        
        if state != self._current_state:
            old_state = self._current_state
            self._current_state = state
            self._current_expression = EMOTION_EXPRESSIONS[state]
            self._last_change_time = time.time()
            
            # Track history
            self._state_history.append((state, self._last_change_time))
            if len(self._state_history) > self._max_history:
                self._state_history.pop(0)
            
            logger.info(f"Emotional state changed: {old_state} -> {state}")
            
            # Notify callback
            if self._on_state_change:
                try:
                    self._on_state_change(state, self._current_expression)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
    
    def get_expression_offsets(self) -> tuple[float, float, float, float, float, float]:
        """
        Get the head pose offsets for current emotional expression.
        
        Returns:
            Tuple of (x, y, z, roll, pitch, yaw) offsets
        """
        expr = self._current_expression
        return (
            0.0,  # x
            0.0,  # y
            expr.bounce_amplitude if expr.bounce else 0.0,  # z (bounce)
            expr.head_roll_offset,
            expr.head_pitch_offset,
            expr.head_yaw_offset,
        )
    
    def get_antenna_positions(self, time_offset: float = 0.0) -> tuple[float, float]:
        """
        Get antenna positions for current emotional expression.
        
        Args:
            time_offset: Time for wiggle animation
            
        Returns:
            Tuple of (left_antenna, right_antenna) in radians
        """
        expr = self._current_expression
        left = expr.left_antenna
        right = expr.right_antenna
        
        if expr.antenna_wiggle:
            import math
            wiggle = math.sin(time_offset * expr.antenna_wiggle_speed * 2 * math.pi) * 0.1
            left += wiggle
            right -= wiggle  # Opposite for natural look
        
        return (left, right)
    
    def should_enable_face_tracking(self) -> bool:
        """Check if face tracking should be enabled for current emotion."""
        return self._current_expression.enable_face_tracking
    
    def update(self):
        """
        Update emotional state (call periodically for auto-decay).
        """
        if not self._auto_decay:
            return
        
        elapsed = time.time() - self._last_change_time
        
        if elapsed > self._decay_after_seconds and self._current_state != self._decay_to:
            logger.debug(f"Emotional state decaying to {self._decay_to}")
            self.set_state(self._decay_to)
    
    def infer_from_text(self, text: str) -> EmotionalState:
        """
        Infer appropriate emotional state from response text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Suggested emotional state
        """
        text_lower = text.lower()
        
        # Check for excitement markers
        if any(w in text_lower for w in ["exciting", "amazing", "awesome", "fantastic", "love"]):
            return "excited"
        
        # Check for happiness
        if any(w in text_lower for w in ["happy", "glad", "great", "wonderful", "pleased"]):
            return "happy"
        
        # Check for curiosity
        if any(w in text_lower for w in ["interesting", "curious", "wonder", "fascinating"]):
            return "curious"
        
        # Check for thinking
        if any(w in text_lower for w in ["think", "consider", "perhaps", "maybe", "hmm"]):
            return "thinking"
        
        # Check for helpfulness
        if any(w in text_lower for w in ["help", "assist", "here's", "let me"]):
            return "helpful"
        
        # Check for playfulness
        if any(w in text_lower for w in ["joke", "fun", "play", "haha", "silly"]):
            return "playful"
        
        # Default to attentive
        return "attentive"
    
    def get_state_summary(self) -> dict:
        """Get a summary of current emotional state."""
        return {
            "state": self._current_state,
            "since": self._last_change_time,
            "duration_seconds": time.time() - self._last_change_time,
            "face_tracking_recommended": self.should_enable_face_tracking(),
            "recent_history": [s[0] for s in self._state_history[-5:]],
        }


# Global emotional state manager instance
_emotional_manager: Optional[EmotionalStateManager] = None


def get_emotional_manager() -> EmotionalStateManager:
    """Get the global emotional state manager."""
    global _emotional_manager
    if _emotional_manager is None:
        _emotional_manager = EmotionalStateManager()
    return _emotional_manager


def set_emotional_state(state: EmotionalState):
    """Convenience function to set emotional state."""
    get_emotional_manager().set_state(state)


def get_emotional_state() -> EmotionalState:
    """Convenience function to get current emotional state."""
    return get_emotional_manager().current_state
