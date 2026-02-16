"""Unit tests for IdleGenerator: micro-movements, scan suppression, scanning."""

import time
from unittest.mock import patch

import pytest

from agent.soul.config import SoulConfig
from agent.soul.idle_generator import IdleGenerator, IdleState


def _make_config(**overrides) -> SoulConfig:
    defaults = dict(
        debug_logging=False,
        log_emotions=False,
        log_events=False,
        log_decisions=False,
    )
    defaults.update(overrides)
    return SoulConfig(**defaults)


class TestIdleGeneratorMicroMovements:
    """Test micro-movement generation."""

    def test_micro_movements_disabled_returns_zero_overlay(self):
        cfg = _make_config(idle_micro_movements_enabled=False)
        gen = IdleGenerator(cfg)
        overlay = gen.update(dt=0.1)
        assert overlay.head_yaw == 0.0
        assert overlay.head_roll == 0.0
        assert overlay.head_pitch == 0.0

    def test_micro_movements_enabled_returns_nonzero(self):
        cfg = _make_config(
            idle_micro_movements_enabled=True,
            idle_micro_movement_amplitude=0.1,
            idle_micro_movement_frequency_hz=1.0,
        )
        gen = IdleGenerator(cfg)
        # Advance time a bit so sine functions produce non-zero
        gen._noise_time = 1.0
        overlay = gen.update(dt=0.1)
        # At least one axis should be non-zero
        any_nonzero = (
            abs(overlay.head_yaw) > 1e-6
            or abs(overlay.head_roll) > 1e-6
            or abs(overlay.head_pitch) > 1e-6
        )
        assert any_nonzero

    def test_micro_movements_bounded_by_amplitude(self):
        amp = 0.05
        cfg = _make_config(
            idle_micro_movements_enabled=True,
            idle_micro_movement_amplitude=amp,
            idle_micro_movement_frequency_hz=1.0,
        )
        gen = IdleGenerator(cfg)
        # Sample many time steps
        for t in range(100):
            gen._noise_time = t * 0.1
            overlay = gen.update(dt=0.1)
            # The multi-octave sum has max magnitude of 1.0 * amp
            assert abs(overlay.head_yaw) <= amp + 1e-6
            assert abs(overlay.head_roll) <= amp * 0.5 + 1e-6
            assert abs(overlay.head_pitch) <= amp * 0.3 + 1e-6


class TestIdleGeneratorScanning:
    """Test scan scheduling and suppression."""

    def test_should_scan_respects_interval(self):
        cfg = _make_config(idle_scanning_enabled=True, idle_scan_interval_s=10.0)
        gen = IdleGenerator(cfg)
        # Just created, not enough time passed
        assert gen.should_scan() is False
        # Advance past interval
        gen._state.time_since_last_scan = 11.0
        assert gen.should_scan() is True

    def test_should_scan_disabled(self):
        cfg = _make_config(idle_scanning_enabled=False)
        gen = IdleGenerator(cfg)
        gen._state.time_since_last_scan = 9999.0
        assert gen.should_scan() is False

    def test_should_scan_suppressed(self):
        cfg = _make_config(idle_scanning_enabled=True, idle_scan_interval_s=1.0)
        gen = IdleGenerator(cfg)
        gen._state.time_since_last_scan = 100.0
        gen.suppress_scanning(True)
        assert gen.should_scan() is False

    def test_suppress_scanning_interrupts_active_scan(self):
        cfg = _make_config(idle_scanning_enabled=True, idle_scan_interval_s=1.0)
        gen = IdleGenerator(cfg)
        gen.start_scan()
        assert gen.is_scanning is True
        gen.suppress_scanning(True)
        assert gen.is_scanning is False

    def test_scan_lifecycle(self):
        cfg = _make_config(idle_scanning_enabled=True, idle_scan_interval_s=1.0)
        gen = IdleGenerator(cfg)
        gen._state.time_since_last_scan = 2.0
        assert gen.should_scan() is True

        gen.start_scan()
        assert gen.is_scanning is True
        # During scan, should_scan returns False
        assert gen.should_scan() is False

        gen.end_scan()
        assert gen.is_scanning is False
        # Timer resets
        assert gen._state.time_since_last_scan == 0.0

    def test_get_scan_target_returns_none_when_not_scanning(self):
        gen = IdleGenerator(_make_config())
        assert gen.get_scan_target() is None

    def test_get_scan_target_returns_tuple_when_scanning(self):
        gen = IdleGenerator(_make_config())
        gen.start_scan()
        target = gen.get_scan_target()
        assert target is not None
        yaw, pitch = target
        assert isinstance(yaw, float)
        assert isinstance(pitch, float)

    def test_scan_auto_ends_after_duration(self):
        gen = IdleGenerator(_make_config())
        gen.start_scan()
        # Set scan start time far enough in the past (>4s)
        gen._state.scan_start_time = time.time() - 5.0
        target = gen.get_scan_target()
        # Should have ended the scan and returned None
        assert target is None
        assert gen.is_scanning is False

    def test_reset_idle_timer(self):
        gen = IdleGenerator(_make_config(idle_scanning_enabled=True))
        gen._state.time_since_last_scan = 100.0
        gen.start_scan()
        gen.reset_idle_timer()
        assert gen.is_scanning is False
        assert gen._state.time_since_last_scan == 0.0

    def test_unsuppress_allows_scanning_again(self):
        cfg = _make_config(idle_scanning_enabled=True, idle_scan_interval_s=1.0)
        gen = IdleGenerator(cfg)
        gen._state.time_since_last_scan = 10.0
        gen.suppress_scanning(True)
        assert gen.should_scan() is False
        gen.suppress_scanning(False)
        assert gen.should_scan() is True

    def test_idle_duration_increases(self):
        gen = IdleGenerator(_make_config())
        start = gen.idle_duration_s
        time.sleep(0.05)
        assert gen.idle_duration_s > start
