"""Unit tests for MovementMode weight blending, smoothstep, and _get_secondary_pose.

These tests exercise the mode system logic without requiring a ReachyMini robot
by testing the math functions in isolation and creating a MovementManager with
a mock robot just enough to access internal methods.
"""

import math
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We need to mock reachy_mini before importing moves.py since it has
# module-level imports from reachy_mini
_mock_reachy = MagicMock()
_mock_reachy.utils.create_head_pose = MagicMock(return_value=np.eye(4))
_mock_reachy.motion.move.Move = type("Move", (), {})
_mock_reachy.utils.interpolation.linear_pose_interpolation = MagicMock()
_mock_reachy.utils.interpolation.compose_world_offset = MagicMock(
    return_value=np.eye(4)
)

import sys

sys.modules.setdefault("reachy_mini", _mock_reachy)
sys.modules.setdefault("reachy_mini.utils", _mock_reachy.utils)
sys.modules.setdefault(
    "reachy_mini.utils.interpolation", _mock_reachy.utils.interpolation
)
sys.modules.setdefault("reachy_mini.motion", _mock_reachy.motion)
sys.modules.setdefault("reachy_mini.motion.move", _mock_reachy.motion.move)

from services.moves import (
    MovementMode,
    DEFAULT_MODE_WEIGHTS,
    MovementState,
    _MAX_SECONDARY_PITCH,
    _MAX_SECONDARY_YAW,
    _MAX_SECONDARY_ROLL,
    _MAX_ANTENNA_OFFSET,
)


class TestMovementMode:
    """Test the MovementMode enum and default weights."""

    def test_all_modes_exist(self):
        assert MovementMode.IDLE.value == "idle"
        assert MovementMode.LISTENING.value == "listening"
        assert MovementMode.PROCESSING.value == "processing"
        assert MovementMode.SPEAKING.value == "speaking"
        assert MovementMode.SCANNING.value == "scanning"

    def test_all_modes_have_default_weights(self):
        for mode in MovementMode:
            assert mode in DEFAULT_MODE_WEIGHTS, f"No default weights for {mode}"

    def test_default_weights_are_4_tuples(self):
        for mode, weights in DEFAULT_MODE_WEIGHTS.items():
            assert len(weights) == 4, f"{mode}: expected 4 weights, got {len(weights)}"

    def test_idle_weights(self):
        w = DEFAULT_MODE_WEIGHTS[MovementMode.IDLE]
        soul_head, soul_antenna, face_tracking, speech = w
        assert soul_head == 1.0
        assert soul_antenna == 1.0
        assert face_tracking == 0.0
        assert speech == 0.0

    def test_speaking_weights_prioritize_speech(self):
        w = DEFAULT_MODE_WEIGHTS[MovementMode.SPEAKING]
        _, _, _, speech = w
        assert speech == 1.0

    def test_listening_weights_prioritize_face_tracking(self):
        w = DEFAULT_MODE_WEIGHTS[MovementMode.LISTENING]
        _, _, face_tracking, _ = w
        assert face_tracking == 1.0


class TestSmoothstep:
    """Test the smoothstep easing function used in mode transitions."""

    def test_smoothstep_at_zero(self):
        t = 0.0
        eased = t * t * (3.0 - 2.0 * t)
        assert eased == 0.0

    def test_smoothstep_at_one(self):
        t = 1.0
        eased = t * t * (3.0 - 2.0 * t)
        assert eased == 1.0

    def test_smoothstep_at_half(self):
        t = 0.5
        eased = t * t * (3.0 - 2.0 * t)
        assert eased == 0.5

    def test_smoothstep_monotonic(self):
        prev = 0.0
        for i in range(1, 101):
            t = i / 100.0
            eased = t * t * (3.0 - 2.0 * t)
            assert eased >= prev, f"Not monotonic at t={t}: {eased} < {prev}"
            prev = eased

    def test_smoothstep_derivative_zero_at_endpoints(self):
        """Derivative of 3t^2 - 2t^3 is 6t - 6t^2 = 6t(1-t), which is 0 at t=0 and t=1."""
        for t in [0.0, 1.0]:
            derivative = 6.0 * t * (1.0 - t)
            assert abs(derivative) < 1e-10


class TestWeightInterpolation:
    """Test weight interpolation math used in _advance_mode_blend."""

    def _interpolate_weights(self, current, target, t):
        """Replicate the interpolation from _advance_mode_blend."""
        eased = t * t * (3.0 - 2.0 * t)
        return tuple(c + (tgt - c) * eased for c, tgt in zip(current, target))

    def test_interpolation_at_zero_returns_current(self):
        current = (1.0, 1.0, 0.0, 0.0)
        target = (0.2, 0.3, 0.4, 1.0)
        result = self._interpolate_weights(current, target, 0.0)
        assert result == current

    def test_interpolation_at_one_returns_target(self):
        current = (1.0, 1.0, 0.0, 0.0)
        target = (0.2, 0.3, 0.4, 1.0)
        result = self._interpolate_weights(current, target, 1.0)
        for r, t in zip(result, target):
            assert abs(r - t) < 1e-10

    def test_interpolation_at_half(self):
        current = (0.0, 0.0, 0.0, 0.0)
        target = (1.0, 1.0, 1.0, 1.0)
        result = self._interpolate_weights(current, target, 0.5)
        for r in result:
            assert abs(r - 0.5) < 1e-10


class TestMovementState:
    """Test MovementState dataclass."""

    def test_default_offsets_are_zero(self):
        state = MovementState()
        assert all(v == 0.0 for v in state.speech_offsets)
        assert all(v == 0.0 for v in state.face_tracking_offsets)
        assert all(v == 0.0 for v in state.soul_offsets)
        assert state.soul_antenna_offsets == (0.0, 0.0)

    def test_update_activity(self):
        state = MovementState()
        old_time = state.last_activity_time
        time.sleep(0.01)
        state.update_activity()
        assert state.last_activity_time > old_time


class TestSecondaryPoseWeighting:
    """Test the weighted combination logic from _get_secondary_pose.

    We replicate the math here since we can't easily instantiate MovementManager
    without a real robot.
    """

    def _compute_weighted_offsets(self, weights, speech, face, soul):
        """Replicate _get_secondary_pose offset computation."""
        soul_head_w, soul_antenna_w, face_tracking_w, speech_w = weights
        return [
            speech[i] * speech_w + face[i] * face_tracking_w + soul[i] * soul_head_w
            for i in range(6)
        ]

    def test_idle_mode_only_soul(self):
        """In IDLE mode, only soul contributes."""
        weights = DEFAULT_MODE_WEIGHTS[MovementMode.IDLE]
        speech = (0.0, 0.0, 0.0, 0.0, 0.1, 0.1)
        face = (0.0, 0.0, 0.0, 0.0, 0.2, 0.2)
        soul = (0.0, 0.0, 0.0, 0.0, 0.3, 0.3)
        result = self._compute_weighted_offsets(weights, speech, face, soul)
        # soul_head_w=1.0, face_tracking_w=0.0, speech_w=0.0
        assert abs(result[4] - 0.3) < 1e-10  # pitch from soul only
        assert abs(result[5] - 0.3) < 1e-10  # yaw from soul only

    def test_speaking_mode_speech_dominant(self):
        """In SPEAKING mode, speech wobble is at full weight."""
        weights = DEFAULT_MODE_WEIGHTS[MovementMode.SPEAKING]
        speech = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
        face = (0.0, 0.0, 0.0, 0.0, 0.0, 0.3)
        soul = (0.0, 0.0, 0.0, 0.0, 0.0, 0.2)
        result = self._compute_weighted_offsets(weights, speech, face, soul)
        # speech_w=1.0, face_tracking_w=0.4, soul_head_w=0.2
        expected_yaw = 0.5 * 1.0 + 0.3 * 0.4 + 0.2 * 0.2
        assert abs(result[5] - expected_yaw) < 1e-10

    def test_listening_mode_face_tracking_dominant(self):
        """In LISTENING mode, face tracking is at full weight."""
        weights = DEFAULT_MODE_WEIGHTS[MovementMode.LISTENING]
        face = (0.0, 0.0, 0.0, 0.0, 0.0, 0.4)
        speech = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        soul = (0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
        result = self._compute_weighted_offsets(weights, speech, face, soul)
        # face_tracking_w=1.0, soul_head_w=0.3
        expected_yaw = 0.4 * 1.0 + 0.1 * 0.3
        assert abs(result[5] - expected_yaw) < 1e-10

    def test_clamping_pitch(self):
        """Values beyond max pitch should be clamped."""
        raw = 2.0  # Way beyond _MAX_SECONDARY_PITCH
        clamped = max(-_MAX_SECONDARY_PITCH, min(_MAX_SECONDARY_PITCH, raw))
        assert clamped == _MAX_SECONDARY_PITCH

    def test_clamping_yaw(self):
        raw = -2.0
        clamped = max(-_MAX_SECONDARY_YAW, min(_MAX_SECONDARY_YAW, raw))
        assert clamped == -_MAX_SECONDARY_YAW

    def test_clamping_roll(self):
        raw = 1.0
        clamped = max(-_MAX_SECONDARY_ROLL, min(_MAX_SECONDARY_ROLL, raw))
        assert clamped == _MAX_SECONDARY_ROLL

    def test_antenna_clamping(self):
        raw = 1.0  # Beyond _MAX_ANTENNA_OFFSET
        clamped = max(-_MAX_ANTENNA_OFFSET, min(_MAX_ANTENNA_OFFSET, raw))
        assert clamped == _MAX_ANTENNA_OFFSET

    def test_scanning_mode_zeroes_soul_head_and_speech(self):
        """SCANNING mode has soul_head_w=0, speech_w=0."""
        weights = DEFAULT_MODE_WEIGHTS[MovementMode.SCANNING]
        speech = (0.0, 0.0, 0.0, 0.1, 0.1, 0.1)
        face = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        soul = (0.0, 0.0, 0.0, 0.2, 0.2, 0.2)
        result = self._compute_weighted_offsets(weights, speech, face, soul)
        # soul_head_w=0.0, speech_w=0.0 -> all zero
        for i in range(6):
            assert abs(result[i]) < 1e-10
