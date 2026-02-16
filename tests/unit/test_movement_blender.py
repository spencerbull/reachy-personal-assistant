"""Unit tests for MovementBlender, Pose, and BlendTarget."""

import math
import time
from unittest.mock import patch

import numpy as np
import pytest

from agent.soul.config import SoulConfig
from agent.soul.movement_blender import MovementBlender, Pose, BlendTarget


class TestPose:
    """Test the Pose dataclass."""

    def test_default_pose_is_zero(self):
        p = Pose()
        arr = p.to_array()
        np.testing.assert_array_equal(arr, np.zeros(9))

    def test_to_array_and_from_array_roundtrip(self):
        p = Pose(
            head_pitch=0.1,
            head_yaw=0.2,
            head_roll=0.3,
            head_x=0.01,
            head_y=0.02,
            head_z=0.03,
            left_antenna=0.4,
            right_antenna=-0.4,
            body_yaw=0.5,
        )
        arr = p.to_array()
        p2 = Pose.from_array(arr)
        assert abs(p2.head_pitch - 0.1) < 1e-10
        assert abs(p2.head_yaw - 0.2) < 1e-10
        assert abs(p2.left_antenna - 0.4) < 1e-10
        assert abs(p2.right_antenna - (-0.4)) < 1e-10

    def test_copy_creates_independent_instance(self):
        p = Pose(head_pitch=0.5)
        p2 = p.copy()
        p2.head_pitch = 1.0
        assert p.head_pitch == 0.5

    def test_str_shows_degrees(self):
        p = Pose(head_pitch=math.radians(10), head_yaw=math.radians(-20))
        s = str(p)
        assert "10.0" in s
        assert "-20.0" in s


class TestBlendTarget:
    """Test BlendTarget progress calculation."""

    def test_progress_starts_at_zero_or_near(self):
        target = BlendTarget(target_pose=Pose(), duration_s=1.0)
        # Just created, progress should be near 0
        assert target.progress < 0.1

    def test_progress_reaches_one(self):
        target = BlendTarget(
            target_pose=Pose(),
            duration_s=0.1,
            start_time=time.time() - 0.2,  # Already past duration
        )
        assert target.progress >= 1.0
        assert target.is_complete

    def test_zero_duration_immediately_complete(self):
        target = BlendTarget(target_pose=Pose(), duration_s=0.0)
        assert target.progress == 1.0
        assert target.is_complete


class TestMovementBlender:
    """Test the MovementBlender blending logic."""

    def _make_blender(self, **config_overrides) -> MovementBlender:
        cfg = SoulConfig(
            debug_logging=False,
            log_emotions=False,
            log_events=False,
            log_decisions=False,
            **config_overrides,
        )
        return MovementBlender(cfg)

    def test_initial_pose_is_zero(self):
        blender = self._make_blender()
        pose = blender.get_current_pose()
        np.testing.assert_array_almost_equal(pose.to_array(), np.zeros(9))

    def test_set_emotion_creates_blend_target(self):
        blender = self._make_blender()
        blender.set_emotion("happy", intensity=0.8)
        assert blender.is_blending()

    def test_set_unknown_emotion_falls_back_to_neutral(self):
        blender = self._make_blender()
        blender.set_emotion("nonexistent_emotion", intensity=0.5)
        # Should not crash, should use neutral
        assert blender._current_emotion == "neutral"

    def test_update_moves_toward_target(self):
        blender = self._make_blender(movement_blend_duration_s=0.01)
        blender.set_emotion("happy", intensity=0.8)
        # Force blend to complete by setting start_time in the past
        if blender._blend_target:
            blender._blend_target.start_time = time.time() - 1.0
        pose = blender.update(dt=0.1)
        # After completion, pose should be non-zero for happy emotion
        # (happy has antenna offsets at least)
        arr = pose.to_array()
        # At least some values should be non-zero
        assert np.any(np.abs(arr) > 0.001)

    def test_blend_completes(self):
        blender = self._make_blender(movement_blend_duration_s=0.01)
        blender.set_emotion("curious", intensity=0.5)
        if blender._blend_target:
            blender._blend_target.start_time = time.time() - 1.0
        blender.update(dt=0.1)
        blender.update(dt=0.1)
        assert not blender.is_blending()

    def test_add_overlay(self):
        blender = self._make_blender()
        blender.update(dt=0.01)  # Ensure current_pose exists
        overlay = Pose(head_pitch=0.1, left_antenna=0.2)
        blender.add_overlay(overlay, weight=0.5)
        pose = blender.get_current_pose()
        assert abs(pose.head_pitch - 0.05) < 1e-6
        assert abs(pose.left_antenna - 0.1) < 1e-6

    def test_add_overlay_weight_clamped(self):
        blender = self._make_blender()
        blender.update(dt=0.01)
        overlay = Pose(head_yaw=1.0)
        blender.add_overlay(overlay, weight=2.0)  # Should be clamped to 1.0
        pose = blender.get_current_pose()
        assert abs(pose.head_yaw - 1.0) < 1e-6

    def test_ease_in_out_boundaries(self):
        blender = self._make_blender()
        assert blender._ease_in_out(0.0) == 0.0
        assert blender._ease_in_out(1.0) == 1.0
        # Midpoint should be 0.5 for smoothstep
        assert abs(blender._ease_in_out(0.5) - 0.5) < 1e-10

    def test_ease_in_out_monotonic(self):
        blender = self._make_blender()
        prev = 0.0
        for i in range(1, 100):
            t = i / 100.0
            val = blender._ease_in_out(t)
            assert val >= prev, f"Not monotonic at t={t}"
            prev = val

    def test_interpolate_poses_at_zero(self):
        blender = self._make_blender()
        start = Pose(head_pitch=0.0)
        end = Pose(head_pitch=1.0)
        result = blender._interpolate_poses(start, end, 0.0)
        assert abs(result.head_pitch) < 1e-10

    def test_interpolate_poses_at_one(self):
        blender = self._make_blender()
        start = Pose(head_pitch=0.0)
        end = Pose(head_pitch=1.0)
        result = blender._interpolate_poses(start, end, 1.0)
        assert abs(result.head_pitch - 1.0) < 1e-10

    def test_interpolate_poses_at_half(self):
        blender = self._make_blender()
        start = Pose(head_yaw=-0.5)
        end = Pose(head_yaw=0.5)
        result = blender._interpolate_poses(start, end, 0.5)
        assert abs(result.head_yaw) < 1e-10

    def test_to_reachy_command_returns_dict(self):
        blender = self._make_blender()
        blender.set_emotion("happy", intensity=0.5)
        if blender._blend_target:
            blender._blend_target.start_time = time.time() - 1.0
        blender.update(dt=0.1)
        cmd = blender.to_reachy_command()
        assert isinstance(cmd, dict)
        expected_keys = [
            "head_pitch",
            "head_yaw",
            "head_roll",
            "head_z_offset",
            "left_antenna",
            "right_antenna",
            "body_yaw",
        ]
        for key in expected_keys:
            assert key in cmd, f"Missing key: {key}"

    def test_to_reachy_command_angles_in_degrees(self):
        blender = self._make_blender()
        # Set a known pose directly
        blender._current_pose = Pose(
            head_pitch=math.radians(10),
            head_yaw=math.radians(-20),
        )
        cmd = blender.to_reachy_command()
        assert abs(cmd["head_pitch"] - 10.0) < 0.1
        assert abs(cmd["head_yaw"] - (-20.0)) < 0.1

    def test_get_antenna_positions_basic(self):
        blender = self._make_blender()
        blender._current_pose = Pose(left_antenna=0.1, right_antenna=-0.1)
        left, right = blender.get_antenna_positions()
        assert abs(left - 0.1) < 0.01
        assert abs(right - (-0.1)) < 0.01

    def test_smooth_pose_converges(self):
        blender = self._make_blender(movement_smooth_factor=0.5)
        current = Pose(head_pitch=0.0)
        target = Pose(head_pitch=1.0)
        # Repeated smoothing should converge to target
        for _ in range(100):
            current = blender._smooth_pose(current, target, 0.5, 1.0 / 60.0)
        assert abs(current.head_pitch - 1.0) < 0.01
