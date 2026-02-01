"""
Movement Blender for Smooth Pose Transitions.

Smoothly interpolates between emotional poses to prevent jarring movements.
Uses exponential smoothing and configurable blend durations.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from agent.soul.config import SoulConfig
from agent.memory.emotional import (
    EmotionalState,
    EmotionExpression,
    EMOTION_EXPRESSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """
    Robot pose representation.
    
    All angles in radians, positions in meters.
    """
    # Head pose (4x4 transformation matrix or simplified 6DOF)
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    head_x: float = 0.0
    head_y: float = 0.0
    head_z: float = 0.0
    
    # Antennas (radians)
    left_antenna: float = 0.0
    right_antenna: float = 0.0
    
    # Body yaw (radians)
    body_yaw: float = 0.0
    
    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array for interpolation."""
        return np.array([
            self.head_pitch, self.head_yaw, self.head_roll,
            self.head_x, self.head_y, self.head_z,
            self.left_antenna, self.right_antenna,
            self.body_yaw,
        ], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Pose":
        """Create from numpy array."""
        return cls(
            head_pitch=float(arr[0]),
            head_yaw=float(arr[1]),
            head_roll=float(arr[2]),
            head_x=float(arr[3]),
            head_y=float(arr[4]),
            head_z=float(arr[5]),
            left_antenna=float(arr[6]),
            right_antenna=float(arr[7]),
            body_yaw=float(arr[8]),
        )
    
    def copy(self) -> "Pose":
        """Create a copy of this pose."""
        return Pose(
            head_pitch=self.head_pitch,
            head_yaw=self.head_yaw,
            head_roll=self.head_roll,
            head_x=self.head_x,
            head_y=self.head_y,
            head_z=self.head_z,
            left_antenna=self.left_antenna,
            right_antenna=self.right_antenna,
            body_yaw=self.body_yaw,
        )
    
    def __str__(self) -> str:
        return (
            f"Pose(pitch={math.degrees(self.head_pitch):.1f}°, "
            f"yaw={math.degrees(self.head_yaw):.1f}°, "
            f"ant=[{math.degrees(self.left_antenna):.1f}°, {math.degrees(self.right_antenna):.1f}°])"
        )


@dataclass
class BlendTarget:
    """A target pose to blend towards."""
    target_pose: Pose
    duration_s: float
    start_time: float = field(default_factory=time.time)
    start_pose: Optional[Pose] = None
    
    @property
    def progress(self) -> float:
        """Get blend progress (0.0 to 1.0)."""
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.duration_s) if self.duration_s > 0 else 1.0
    
    @property
    def is_complete(self) -> bool:
        """Check if blend is complete."""
        return self.progress >= 1.0


class MovementBlender:
    """
    Smoothly blends between poses for natural robot movement.
    
    Uses a combination of:
    1. Linear interpolation for pose transitions
    2. Exponential smoothing for continuous adjustment
    3. Easing functions for natural acceleration/deceleration
    """
    
    def __init__(self, config: SoulConfig):
        self.config = config
        
        # Current actual pose
        self._current_pose = Pose()
        
        # Smoothed pose (for continuous smoothing)
        self._smoothed_pose = Pose()
        
        # Active blend target
        self._blend_target: Optional[BlendTarget] = None
        
        # Base pose from emotion (before idle overlays)
        self._emotion_pose = Pose()
        
        # Current emotion for reference
        self._current_emotion: EmotionalState = "neutral"
        self._emotion_intensity: float = 0.5
        
        # Last update time
        self._last_update_time = time.time()
    
    def set_emotion(
        self,
        emotion: EmotionalState,
        intensity: float = 0.5,
        blend_duration: Optional[float] = None,
    ):
        """
        Set target emotion with smooth blend transition.
        
        Args:
            emotion: Target emotional state
            intensity: Emotion intensity (0-1)
            blend_duration: Override blend duration (uses config default if None)
        """
        if emotion not in EMOTION_EXPRESSIONS:
            logger.warning(f"Unknown emotion '{emotion}', using neutral")
            emotion = "neutral"
        
        expression = EMOTION_EXPRESSIONS[emotion]
        intensity = max(0.0, min(1.0, intensity))
        
        # Calculate target pose from emotion expression
        target_pose = self._expression_to_pose(expression, intensity)
        
        # Set up blend
        duration = blend_duration or self.config.movement_blend_duration_s
        self._blend_target = BlendTarget(
            target_pose=target_pose,
            duration_s=duration,
            start_pose=self._emotion_pose.copy(),
        )
        
        self._current_emotion = emotion
        self._emotion_intensity = intensity
        
        if self.config.debug_logging:
            logger.debug(f"Blending to emotion: {emotion} ({intensity:.2f}) over {duration:.2f}s")
    
    def _expression_to_pose(self, expression: EmotionExpression, intensity: float) -> Pose:
        """Convert emotion expression to pose, scaled by intensity."""
        return Pose(
            head_pitch=expression.head_pitch_offset * intensity,
            head_yaw=expression.head_yaw_offset * intensity,
            head_roll=expression.head_roll_offset * intensity,
            head_x=0.0,
            head_y=0.0,
            head_z=expression.bounce_amplitude * intensity if expression.bounce else 0.0,
            left_antenna=expression.left_antenna * intensity,
            right_antenna=expression.right_antenna * intensity,
            body_yaw=0.0,
        )
    
    def update(self, dt: Optional[float] = None) -> Pose:
        """
        Update pose blending and return current pose.
        
        Args:
            dt: Time delta since last update (auto-calculated if None)
            
        Returns:
            Current blended pose
        """
        now = time.time()
        if dt is None:
            dt = now - self._last_update_time
        self._last_update_time = now
        
        # Update blend target if active
        if self._blend_target is not None:
            progress = self._blend_target.progress
            
            if progress < 1.0:
                # Apply easing
                eased_progress = self._ease_in_out(progress)
                
                # Interpolate between start and target poses
                start = self._blend_target.start_pose or Pose()
                target = self._blend_target.target_pose
                
                self._emotion_pose = self._interpolate_poses(start, target, eased_progress)
            else:
                # Blend complete
                self._emotion_pose = self._blend_target.target_pose.copy()
                self._blend_target = None
        
        # Apply exponential smoothing to current pose
        self._current_pose = self._smooth_pose(
            self._current_pose,
            self._emotion_pose,
            self.config.movement_smooth_factor,
            dt,
        )
        
        return self._current_pose
    
    def add_overlay(self, overlay: Pose, weight: float = 1.0):
        """
        Add an overlay pose (e.g., breathing, micro-movements).
        
        The overlay is added on top of the emotion pose.
        
        Args:
            overlay: Pose offset to add
            weight: Blend weight (0-1)
        """
        weight = max(0.0, min(1.0, weight))
        
        self._current_pose.head_pitch += overlay.head_pitch * weight
        self._current_pose.head_yaw += overlay.head_yaw * weight
        self._current_pose.head_roll += overlay.head_roll * weight
        self._current_pose.head_x += overlay.head_x * weight
        self._current_pose.head_y += overlay.head_y * weight
        self._current_pose.head_z += overlay.head_z * weight
        self._current_pose.left_antenna += overlay.left_antenna * weight
        self._current_pose.right_antenna += overlay.right_antenna * weight
    
    def _interpolate_poses(self, start: Pose, end: Pose, t: float) -> Pose:
        """Linear interpolation between two poses."""
        start_arr = start.to_array()
        end_arr = end.to_array()
        interp_arr = start_arr + (end_arr - start_arr) * t
        return Pose.from_array(interp_arr)
    
    def _smooth_pose(
        self,
        current: Pose,
        target: Pose,
        factor: float,
        dt: float,
    ) -> Pose:
        """
        Apply exponential smoothing to pose.
        
        Uses framerate-independent smoothing based on dt.
        """
        # Framerate-independent smoothing factor
        # Reference: https://www.gamedeveloper.com/programming/improved-lerp-smoothing-
        smooth = 1.0 - math.pow(1.0 - factor, dt * 60.0)  # Normalize to 60fps
        
        current_arr = current.to_array()
        target_arr = target.to_array()
        
        smoothed_arr = current_arr + (target_arr - current_arr) * smooth
        return Pose.from_array(smoothed_arr)
    
    def _ease_in_out(self, t: float) -> float:
        """Smoothstep easing function for natural acceleration/deceleration."""
        # Smoothstep: 3t² - 2t³
        return t * t * (3.0 - 2.0 * t)
    
    def get_current_pose(self) -> Pose:
        """Get the current blended pose."""
        return self._current_pose.copy()
    
    def get_emotion_pose(self) -> Pose:
        """Get the base emotion pose (before smoothing/overlays)."""
        return self._emotion_pose.copy()
    
    def is_blending(self) -> bool:
        """Check if currently blending between poses."""
        return self._blend_target is not None and not self._blend_target.is_complete
    
    def get_antenna_positions(self, time_offset: float = 0.0) -> Tuple[float, float]:
        """
        Get current antenna positions with optional wiggle animation.
        
        Args:
            time_offset: Time for wiggle animation phase
            
        Returns:
            Tuple of (left_antenna, right_antenna) in radians
        """
        expression = EMOTION_EXPRESSIONS.get(self._current_emotion, EMOTION_EXPRESSIONS["neutral"])
        
        left = self._current_pose.left_antenna
        right = self._current_pose.right_antenna
        
        # Add wiggle if enabled for this emotion
        if expression.antenna_wiggle:
            wiggle = math.sin(time_offset * expression.antenna_wiggle_speed * 2 * math.pi)
            wiggle *= 0.1 * self._emotion_intensity
            left += wiggle
            right -= wiggle  # Opposite for natural look
        
        return (left, right)
    
    def to_reachy_command(self) -> dict:
        """
        Convert current pose to Reachy movement command.
        
        Returns:
            Dictionary with movement parameters for ReachyService
        """
        pose = self._current_pose
        
        return {
            "head_pitch": math.degrees(pose.head_pitch),
            "head_yaw": math.degrees(pose.head_yaw),
            "head_roll": math.degrees(pose.head_roll),
            "head_z_offset": pose.head_z,
            "left_antenna": pose.left_antenna,
            "right_antenna": pose.right_antenna,
            "body_yaw": math.degrees(pose.body_yaw),
        }
