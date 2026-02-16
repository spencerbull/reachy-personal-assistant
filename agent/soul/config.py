"""
Soul System Configuration.

Defines all configurable parameters for the Reachy Soul embodiment loop.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class PersonalityConfig:
    """Configuration for personality loading."""

    soul_file_path: str = "REACHY_SOUL.md"
    auto_reload: bool = False  # Reload personality on file change
    reload_interval_s: float = 60.0  # How often to check for changes
    use_personality_expressions: bool = True  # Use SOUL.md physical expressions
    use_personality_prompts: bool = True  # Use SOUL.md for system prompts


@dataclass
class SoulConfig:
    """
    Configuration for the Reachy Soul System.

    Attributes:
        poll_interval_ms: How often the soul loop runs (milliseconds)
        emotion_inference_interval_ms: How often to run LLM emotion inference

        soul_model_url: OpenAI-compatible API endpoint for emotion inference
        soul_model_name: Model name for emotion inference
        soul_model_temperature: Temperature for emotion inference (lower = more consistent)
        soul_model_max_tokens: Max tokens for emotion inference response

        idle_micro_movements_enabled: Whether to add subtle random movements
        idle_micro_movement_amplitude: How much micro-movement (radians)
        idle_micro_movement_frequency_hz: How often micro-movements occur

        idle_scanning_enabled: Whether to occasionally look around when idle
        idle_scan_interval_s: How often to scan when no interaction

        movement_blend_duration_s: Duration to blend between poses (seconds)
        movement_smooth_factor: Smoothing factor for movement (0-1, higher = smoother)

        mode_transition_blend_s: Duration to smoothly transition between movement modes
        mode_weights: Per-mode weight tuples (soul_head, soul_antenna, face_tracking, speech)

        user_speaking_triggers_attention: Look at user when they speak
        user_silence_triggers_processing_look: Show thinking pose when user stops
        processing_look_delay_ms: Delay before showing thinking pose

        emotion_decay_enabled: Whether emotions decay back to neutral
        emotion_decay_to: Default emotion to decay to
        emotion_decay_after_s: Seconds before emotion decays

        antenna_wave_on_face_detected: Wave antenna when new face detected
    """

    # Polling / Loop timing
    # NOTE: Poll interval affects responsiveness of emotion/behavior changes
    # Keep fast (200ms) for quick reactions, emotion inference runs separately
    poll_interval_ms: int = 200
    emotion_inference_interval_ms: int = 2000  # LLM inference every 2s (expensive)

    # Model configuration (use VLM for emotion inference)
    soul_model_url: str = "http://localhost:8002/v1"
    soul_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    soul_model_temperature: float = 0.3  # Lower for consistent emotion inference
    soul_model_max_tokens: int = 100  # Small response for emotion JSON

    # Idle behavior - Micro movements
    # NOTE: Disabled by default. MovementManager handles primary idle behaviors
    # (BreathingMove). Enable for additional subtle variation on top.
    idle_micro_movements_enabled: bool = False
    idle_micro_movement_amplitude: float = 0.02  # ~1 degree subtle shifts
    idle_micro_movement_frequency_hz: float = 0.1  # Every ~10 seconds

    # Idle behavior - Scanning
    # Soul-driven room scanning when idle
    idle_scanning_enabled: bool = True
    idle_scan_interval_s: float = 45.0  # Scan room every 45s when idle

    # Movement blending
    movement_blend_duration_s: float = 0.5  # 500ms transitions
    movement_smooth_factor: float = 0.15  # Exponential smoothing factor

    # Movement mode transition blending
    # Controls how smoothly weights transition between movement modes
    # (e.g., IDLE -> LISTENING -> PROCESSING -> SPEAKING)
    # Higher values = smoother but slower transitions. Tune this to allow
    # enough time for LLM processing while keeping transitions natural.
    mode_transition_blend_s: float = 0.3  # Seconds to blend between modes

    # Movement mode weights: (soul_head, soul_antenna, face_tracking, speech)
    # Each mode defines how much each movement source contributes.
    # Values 0.0 to 1.0. These are the primary tuning knobs for movement conflict resolution.
    mode_weights_idle: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)
    mode_weights_listening: Tuple[float, float, float, float] = (0.3, 0.5, 1.0, 0.0)
    mode_weights_processing: Tuple[float, float, float, float] = (0.8, 0.8, 0.6, 0.0)
    mode_weights_speaking: Tuple[float, float, float, float] = (0.2, 0.3, 0.4, 1.0)
    mode_weights_scanning: Tuple[float, float, float, float] = (0.0, 0.5, 0.0, 0.0)

    # Event triggers (rule-based, no LLM)
    user_speaking_triggers_attention: bool = True
    user_silence_triggers_processing_look: bool = True
    processing_look_delay_ms: int = 300  # Short delay before "thinking" pose

    # Emotion decay
    emotion_decay_enabled: bool = True
    emotion_decay_to: str = "neutral"
    emotion_decay_after_s: float = 30.0

    # Face tracking
    # NOTE: Face tracking is now controlled by movement mode weights, not toggled on/off.
    # The camera always runs face detection; mode weights control how much the
    # face tracking offsets affect the head pose.
    antenna_wave_on_face_detected: bool = True

    # Personality
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)

    # Logging Configuration
    # debug_logging: Enable verbose debug output for all soul components
    # log_level: Overall log level ("DEBUG", "INFO", "WARNING", "ERROR")
    # log_emotions: Log emotion inference and changes
    # log_events: Log soul event processing
    # log_decisions: Log decision points and reasoning
    # log_movements: Log pose/movement updates (can be noisy)
    # log_status_interval_s: How often to log status summary (0 to disable)
    debug_logging: bool = False
    log_level: str = "INFO"
    log_emotions: bool = True
    log_events: bool = True
    log_decisions: bool = True
    log_movements: bool = False  # Can be very noisy
    log_status_interval_s: float = 30.0  # Status summary every 30s

    @classmethod
    def from_env(cls) -> "SoulConfig":
        """Create config from environment variables."""
        import os

        personality = PersonalityConfig(
            soul_file_path=os.getenv("SOUL_FILE_PATH", "REACHY_SOUL.md"),
            auto_reload=os.getenv("SOUL_AUTO_RELOAD", "false").lower() == "true",
            use_personality_expressions=os.getenv(
                "SOUL_USE_PERSONALITY_EXPRESSIONS", "true"
            ).lower()
            == "true",
            use_personality_prompts=os.getenv(
                "SOUL_USE_PERSONALITY_PROMPTS", "true"
            ).lower()
            == "true",
        )

        def _parse_weights(
            env_key: str, default: Tuple[float, float, float, float]
        ) -> Tuple[float, float, float, float]:
            """Parse a comma-separated weight tuple from env var."""
            raw = os.getenv(env_key)
            if raw:
                try:
                    parts = [float(x.strip()) for x in raw.split(",")]
                    if len(parts) == 4:
                        return (parts[0], parts[1], parts[2], parts[3])
                except (ValueError, IndexError):
                    pass
            return default

        return cls(
            poll_interval_ms=int(os.getenv("SOUL_POLL_INTERVAL_MS", "200")),
            emotion_inference_interval_ms=int(
                os.getenv("SOUL_EMOTION_INFERENCE_INTERVAL_MS", "2000")
            ),
            soul_model_url=os.getenv("SOUL_MODEL_URL", "http://localhost:8002/v1"),
            soul_model_name=os.getenv(
                "SOUL_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
            ),
            idle_micro_movements_enabled=os.getenv(
                "SOUL_IDLE_MICRO_MOVEMENTS", "false"
            ).lower()
            == "true",
            idle_scanning_enabled=os.getenv("SOUL_IDLE_SCANNING", "true").lower()
            == "true",
            mode_transition_blend_s=float(
                os.getenv("SOUL_MODE_TRANSITION_BLEND_S", "0.3")
            ),
            mode_weights_idle=_parse_weights(
                "SOUL_MODE_WEIGHTS_IDLE", (1.0, 1.0, 0.0, 0.0)
            ),
            mode_weights_listening=_parse_weights(
                "SOUL_MODE_WEIGHTS_LISTENING", (0.3, 0.5, 1.0, 0.0)
            ),
            mode_weights_processing=_parse_weights(
                "SOUL_MODE_WEIGHTS_PROCESSING", (0.8, 0.8, 0.6, 0.0)
            ),
            mode_weights_speaking=_parse_weights(
                "SOUL_MODE_WEIGHTS_SPEAKING", (0.2, 0.3, 0.4, 1.0)
            ),
            mode_weights_scanning=_parse_weights(
                "SOUL_MODE_WEIGHTS_SCANNING", (0.0, 0.5, 0.0, 0.0)
            ),
            personality=personality,
            debug_logging=os.getenv("SOUL_DEBUG", "false").lower() == "true",
            log_level=os.getenv("SOUL_LOG_LEVEL", "INFO").upper(),
            log_emotions=os.getenv("SOUL_LOG_EMOTIONS", "true").lower() == "true",
            log_events=os.getenv("SOUL_LOG_EVENTS", "true").lower() == "true",
            log_decisions=os.getenv("SOUL_LOG_DECISIONS", "true").lower() == "true",
            log_movements=os.getenv("SOUL_LOG_MOVEMENTS", "false").lower() == "true",
            log_status_interval_s=float(
                os.getenv("SOUL_LOG_STATUS_INTERVAL_S", "30.0")
            ),
        )

    def get_mode_weights(self, mode_name: str) -> Tuple[float, float, float, float]:
        """Get weight tuple for a movement mode by name.

        Returns:
            Tuple of (soul_head, soul_antenna, face_tracking, speech) weights.
        """
        weights_map = {
            "idle": self.mode_weights_idle,
            "listening": self.mode_weights_listening,
            "processing": self.mode_weights_processing,
            "speaking": self.mode_weights_speaking,
            "scanning": self.mode_weights_scanning,
        }
        return weights_map.get(mode_name, self.mode_weights_idle)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "poll_interval_ms": self.poll_interval_ms,
            "emotion_inference_interval_ms": self.emotion_inference_interval_ms,
            "soul_model_url": self.soul_model_url,
            "soul_model_name": self.soul_model_name,
            "idle_micro_movements_enabled": self.idle_micro_movements_enabled,
            "idle_scanning_enabled": self.idle_scanning_enabled,
            "movement_blend_duration_s": self.movement_blend_duration_s,
            "mode_transition_blend_s": self.mode_transition_blend_s,
            "mode_weights": {
                "idle": self.mode_weights_idle,
                "listening": self.mode_weights_listening,
                "processing": self.mode_weights_processing,
                "speaking": self.mode_weights_speaking,
                "scanning": self.mode_weights_scanning,
            },
            "personality": {
                "soul_file_path": self.personality.soul_file_path,
                "auto_reload": self.personality.auto_reload,
                "use_personality_expressions": self.personality.use_personality_expressions,
                "use_personality_prompts": self.personality.use_personality_prompts,
            },
            "logging": {
                "debug_logging": self.debug_logging,
                "log_level": self.log_level,
                "log_emotions": self.log_emotions,
                "log_events": self.log_events,
                "log_decisions": self.log_decisions,
                "log_movements": self.log_movements,
                "log_status_interval_s": self.log_status_interval_s,
            },
        }
