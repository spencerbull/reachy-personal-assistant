"""Unit tests for SoulConfig parsing, from_env(), and mode weight helpers."""

import os
from unittest.mock import patch

from agent.soul.config import SoulConfig, PersonalityConfig


class TestSoulConfigDefaults:
    """Verify dataclass defaults are correct."""

    def test_default_mode_weights_idle(self):
        cfg = SoulConfig()
        assert cfg.mode_weights_idle == (1.0, 1.0, 0.0, 0.0)

    def test_default_mode_weights_listening(self):
        cfg = SoulConfig()
        assert cfg.mode_weights_listening == (0.3, 0.5, 1.0, 0.0)

    def test_default_mode_weights_speaking(self):
        cfg = SoulConfig()
        assert cfg.mode_weights_speaking == (0.2, 0.3, 0.4, 1.0)

    def test_default_mode_weights_processing(self):
        cfg = SoulConfig()
        assert cfg.mode_weights_processing == (0.8, 0.8, 0.6, 0.0)

    def test_default_mode_weights_scanning(self):
        cfg = SoulConfig()
        assert cfg.mode_weights_scanning == (0.0, 0.5, 0.0, 0.0)

    def test_default_mode_transition_blend_s(self):
        cfg = SoulConfig()
        assert cfg.mode_transition_blend_s == 0.3

    def test_idle_micro_movements_disabled_by_default(self):
        cfg = SoulConfig()
        assert cfg.idle_micro_movements_enabled is False

    def test_idle_scanning_enabled_by_default(self):
        cfg = SoulConfig()
        assert cfg.idle_scanning_enabled is True

    def test_personality_defaults(self):
        cfg = SoulConfig()
        assert cfg.personality.soul_file_path == "REACHY_SOUL.md"
        assert cfg.personality.use_personality_prompts is True


class TestSoulConfigFromEnv:
    """Test from_env() parses environment variables correctly."""

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        cfg = SoulConfig.from_env()
        assert cfg.poll_interval_ms == 200
        assert cfg.mode_transition_blend_s == 0.3
        # Fixed bug: these should default to False, not True
        assert cfg.idle_micro_movements_enabled is False

    @patch.dict(
        os.environ,
        {
            "SOUL_MODE_TRANSITION_BLEND_S": "0.6",
            "SOUL_MODE_WEIGHTS_IDLE": "0.9,0.9,0.1,0.0",
            "SOUL_IDLE_MICRO_MOVEMENTS": "true",
        },
        clear=True,
    )
    def test_from_env_custom_values(self):
        cfg = SoulConfig.from_env()
        assert cfg.mode_transition_blend_s == 0.6
        assert cfg.mode_weights_idle == (0.9, 0.9, 0.1, 0.0)
        assert cfg.idle_micro_movements_enabled is True

    @patch.dict(
        os.environ,
        {"SOUL_MODE_WEIGHTS_SPEAKING": "bad,data"},
        clear=True,
    )
    def test_from_env_bad_weights_uses_default(self):
        cfg = SoulConfig.from_env()
        # Should fall back to default since parse fails (wrong count)
        assert cfg.mode_weights_speaking == (0.2, 0.3, 0.4, 1.0)

    @patch.dict(
        os.environ,
        {"SOUL_MODE_WEIGHTS_LISTENING": "not_numbers"},
        clear=True,
    )
    def test_from_env_non_numeric_weights_uses_default(self):
        cfg = SoulConfig.from_env()
        assert cfg.mode_weights_listening == (0.3, 0.5, 1.0, 0.0)


class TestSoulConfigGetModeWeights:
    """Test the get_mode_weights() helper."""

    def test_get_known_modes(self):
        cfg = SoulConfig()
        assert cfg.get_mode_weights("idle") == cfg.mode_weights_idle
        assert cfg.get_mode_weights("listening") == cfg.mode_weights_listening
        assert cfg.get_mode_weights("processing") == cfg.mode_weights_processing
        assert cfg.get_mode_weights("speaking") == cfg.mode_weights_speaking
        assert cfg.get_mode_weights("scanning") == cfg.mode_weights_scanning

    def test_get_unknown_mode_returns_idle(self):
        cfg = SoulConfig()
        assert cfg.get_mode_weights("unknown") == cfg.mode_weights_idle

    def test_custom_weights_are_returned(self):
        cfg = SoulConfig(mode_weights_idle=(0.5, 0.5, 0.5, 0.5))
        assert cfg.get_mode_weights("idle") == (0.5, 0.5, 0.5, 0.5)


class TestSoulConfigToDict:
    """Test serialization."""

    def test_to_dict_contains_mode_weights(self):
        cfg = SoulConfig()
        d = cfg.to_dict()
        assert "mode_weights" in d
        assert "idle" in d["mode_weights"]
        assert d["mode_weights"]["idle"] == (1.0, 1.0, 0.0, 0.0)

    def test_to_dict_contains_mode_transition(self):
        cfg = SoulConfig()
        d = cfg.to_dict()
        assert d["mode_transition_blend_s"] == 0.3

    def test_to_dict_roundtrip_mode_weights(self):
        """Custom weights survive serialization."""
        cfg = SoulConfig(mode_weights_speaking=(0.1, 0.2, 0.3, 0.4))
        d = cfg.to_dict()
        assert d["mode_weights"]["speaking"] == (0.1, 0.2, 0.3, 0.4)
