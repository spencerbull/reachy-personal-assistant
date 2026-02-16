"""
Idle Behavior Generator.

Generates subtle idle movements to give Reachy a sense of life:
- Micro-movements: Tiny random head adjustments (disabled by default)
- Scanning: Occasional room scanning when idle for long periods

NOTE: Breathing is handled exclusively by MovementManager.BreathingMove
(the primary-layer idle move). This generator only produces secondary
overlays for the soul system.
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from agent.soul.config import SoulConfig
from agent.soul.movement_blender import Pose
from agent.soul.logging_utils import SoulLogger

logger = logging.getLogger(__name__)
soul_log = SoulLogger("agent.soul.idle")


@dataclass
class IdleState:
    """Current state of idle behaviors."""

    micro_movement_target: Pose = None  # Current micro-movement target
    micro_movement_progress: float = 1.0  # Progress to target (0-1)
    time_since_last_scan: float = 0.0
    is_scanning: bool = False
    scan_start_time: float = 0.0

    def __post_init__(self):
        if self.micro_movement_target is None:
            self.micro_movement_target = Pose()


class IdleGenerator:
    """
    Generates idle behaviors for lifelike presence.

    All outputs are pose overlays that get added on top of the
    emotion pose by the movement blender.
    """

    def __init__(self, config: SoulConfig):
        self.config = config
        self._state = IdleState()
        self._last_update_time = time.time()
        self._idle_start_time = time.time()

        # Random seed for reproducible but varied behavior
        self._rng = random.Random()

        # Perlin-like noise state for smooth micro-movements
        self._noise_time = 0.0

        # Scanning suppression (e.g., during active face tracking)
        self._scanning_suppressed: bool = False

    def update(self, dt: Optional[float] = None) -> Pose:
        """
        Update idle behaviors and return overlay pose.

        Args:
            dt: Time delta since last update (auto-calculated if None)

        Returns:
            Pose overlay to add to emotion pose
        """
        now = time.time()
        if dt is None:
            dt = now - self._last_update_time
        self._last_update_time = now
        self._state.time_since_last_scan += dt
        self._noise_time += dt

        overlay = Pose()

        # Generate micro-movements (disabled by default)
        if self.config.idle_micro_movements_enabled:
            micro = self._generate_micro_movements(dt)
            overlay.head_yaw += micro.head_yaw
            overlay.head_roll += micro.head_roll
            overlay.head_pitch += micro.head_pitch

        return overlay

    def _generate_micro_movements(self, dt: float) -> Pose:
        """
        Generate subtle random micro-movements.

        Uses smooth noise-like interpolation for natural movement.
        """
        # Use perlin-like smooth noise
        # Simple approximation using sum of sines at different frequencies
        freq = self.config.idle_micro_movement_frequency_hz
        amp = self.config.idle_micro_movement_amplitude
        t = self._noise_time

        # Multiple octaves for natural movement
        yaw = (
            math.sin(t * freq * 0.7) * 0.5
            + math.sin(t * freq * 1.3 + 1.5) * 0.3
            + math.sin(t * freq * 2.1 + 3.0) * 0.2
        ) * amp

        roll = (
            (
                math.sin(t * freq * 0.5 + 2.0) * 0.5
                + math.sin(t * freq * 1.1 + 0.5) * 0.3
                + math.sin(t * freq * 1.9 + 1.0) * 0.2
            )
            * amp
            * 0.5
        )  # Less roll

        pitch = (
            (
                math.sin(t * freq * 0.6 + 1.0) * 0.5
                + math.sin(t * freq * 1.2 + 2.5) * 0.3
                + math.sin(t * freq * 2.0 + 0.3) * 0.2
            )
            * amp
            * 0.3
        )  # Even less pitch

        return Pose(
            head_yaw=yaw,
            head_roll=roll,
            head_pitch=pitch,
        )

    def suppress_scanning(self, suppressed: bool) -> None:
        """Suppress or allow room scanning.

        Call with True when face tracking is active to prevent scanning
        from conflicting with face-following head movements.
        """
        if self._scanning_suppressed != suppressed:
            self._scanning_suppressed = suppressed
            if suppressed and self._state.is_scanning:
                soul_log.log_idle_action(
                    "scan_interrupted", {"reason": "scanning_suppressed"}
                )
                self.end_scan()

    def should_scan(self) -> bool:
        """Check if it's time for an idle room scan."""
        if not self.config.idle_scanning_enabled:
            return False

        if self._scanning_suppressed:
            return False

        if self._state.is_scanning:
            return False

        return self._state.time_since_last_scan >= self.config.idle_scan_interval_s

    def start_scan(self):
        """Start an idle room scan."""
        self._state.is_scanning = True
        self._state.scan_start_time = time.time()
        soul_log.log_idle_action(
            "scan_started", {"idle_duration": f"{self.idle_duration_s:.1f}s"}
        )

    def end_scan(self):
        """End an idle room scan."""
        scan_duration = time.time() - self._state.scan_start_time
        self._state.is_scanning = False
        self._state.time_since_last_scan = 0.0
        soul_log.log_idle_action(
            "scan_completed", {"duration": f"{scan_duration:.1f}s"}
        )

    def get_scan_target(self) -> Optional[Tuple[float, float]]:
        """
        Get the current scan look direction.

        Returns:
            Tuple of (yaw, pitch) in radians, or None if not scanning
        """
        if not self._state.is_scanning:
            return None

        elapsed = time.time() - self._state.scan_start_time
        scan_duration = 4.0  # Full scan takes 4 seconds

        if elapsed >= scan_duration:
            self.end_scan()
            return None

        # Scan pattern: left -> center -> right -> center
        progress = elapsed / scan_duration

        if progress < 0.25:
            # Look left
            t = progress / 0.25
            yaw = t * 0.5  # ~30 degrees left
        elif progress < 0.5:
            # Return to center
            t = (progress - 0.25) / 0.25
            yaw = 0.5 * (1 - t)
        elif progress < 0.75:
            # Look right
            t = (progress - 0.5) / 0.25
            yaw = -t * 0.5  # ~30 degrees right
        else:
            # Return to center
            t = (progress - 0.75) / 0.25
            yaw = -0.5 * (1 - t)

        # Add slight vertical variation
        pitch = math.sin(progress * math.pi * 2) * 0.1

        return (yaw, pitch)

    def reset_idle_timer(self):
        """Reset the idle timer (call when user interacts)."""
        was_idle_for = self.idle_duration_s
        self._idle_start_time = time.time()
        self._state.time_since_last_scan = 0.0

        # Stop any active scan
        if self._state.is_scanning:
            soul_log.log_idle_action("scan_interrupted", {"reason": "user_interaction"})
            self.end_scan()

        if was_idle_for > 5.0:
            soul_log.log_idle_action(
                "idle_reset", {"was_idle_for": f"{was_idle_for:.1f}s"}
            )

    @property
    def idle_duration_s(self) -> float:
        """How long we've been idle."""
        return time.time() - self._idle_start_time

    @property
    def is_scanning(self) -> bool:
        """Check if currently scanning."""
        return self._state.is_scanning
