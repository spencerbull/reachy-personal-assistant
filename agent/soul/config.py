"""
Soul System Configuration.

Defines all configurable parameters for the Reachy Soul embodiment loop.
"""

from dataclasses import dataclass, field
from typing import Optional


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
        
        idle_breathing_enabled: Whether to animate breathing
        idle_breathing_frequency_hz: Breathing rate (cycles per second)
        idle_breathing_amplitude: Amplitude of breathing movement (meters)
        
        idle_micro_movements_enabled: Whether to add subtle random movements
        idle_micro_movement_amplitude: How much micro-movement (radians)
        idle_micro_movement_frequency_hz: How often micro-movements occur
        
        idle_scanning_enabled: Whether to occasionally look around when idle
        idle_scan_interval_s: How often to scan when no interaction
        
        movement_blend_duration_s: Duration to blend between poses (seconds)
        movement_smooth_factor: Smoothing factor for movement (0-1, higher = smoother)
        
        user_speaking_triggers_attention: Look at user when they speak
        user_silence_triggers_processing_look: Show thinking pose when user stops
        processing_look_delay_ms: Delay before showing thinking pose
        
        emotion_decay_enabled: Whether emotions decay back to neutral
        emotion_decay_to: Default emotion to decay to
        emotion_decay_after_s: Seconds before emotion decays
        
        face_tracking_on_attention: Enable face tracking when attentive
        antenna_wave_on_face_detected: Wave antenna when new face detected
    """
    
    # Polling / Loop timing
    poll_interval_ms: int = 200
    emotion_inference_interval_ms: int = 1000  # Don't run LLM every loop
    
    # Model configuration (use router model for fast emotion inference)
    soul_model_url: str = "http://localhost:8003/v1"
    soul_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    soul_model_temperature: float = 0.3  # Lower for consistent emotion inference
    soul_model_max_tokens: int = 100  # Small response for emotion JSON
    
    # Idle behavior - Breathing
    idle_breathing_enabled: bool = True
    idle_breathing_frequency_hz: float = 0.15  # ~9 breaths per minute (relaxed)
    idle_breathing_amplitude: float = 0.003  # 3mm subtle movement
    
    # Idle behavior - Micro movements
    idle_micro_movements_enabled: bool = True
    idle_micro_movement_amplitude: float = 0.02  # ~1 degree subtle shifts
    idle_micro_movement_frequency_hz: float = 0.1  # Every ~10 seconds
    
    # Idle behavior - Scanning
    idle_scanning_enabled: bool = True
    idle_scan_interval_s: float = 30.0  # Scan room every 30s when idle
    
    # Movement blending
    movement_blend_duration_s: float = 0.5  # 500ms transitions
    movement_smooth_factor: float = 0.15  # Exponential smoothing factor
    
    # Event triggers (rule-based, no LLM)
    user_speaking_triggers_attention: bool = True
    user_silence_triggers_processing_look: bool = True
    processing_look_delay_ms: int = 300  # Short delay before "thinking" pose
    
    # Emotion decay
    emotion_decay_enabled: bool = True
    emotion_decay_to: str = "neutral"
    emotion_decay_after_s: float = 30.0
    
    # Face tracking
    face_tracking_on_attention: bool = True
    antenna_wave_on_face_detected: bool = True
    
    # Personality
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    
    # Debug
    debug_logging: bool = False
    
    @classmethod
    def from_env(cls) -> "SoulConfig":
        """Create config from environment variables."""
        import os
        
        personality = PersonalityConfig(
            soul_file_path=os.getenv("SOUL_FILE_PATH", "REACHY_SOUL.md"),
            auto_reload=os.getenv("SOUL_AUTO_RELOAD", "false").lower() == "true",
            use_personality_expressions=os.getenv("SOUL_USE_PERSONALITY_EXPRESSIONS", "true").lower() == "true",
            use_personality_prompts=os.getenv("SOUL_USE_PERSONALITY_PROMPTS", "true").lower() == "true",
        )
        
        return cls(
            poll_interval_ms=int(os.getenv("SOUL_POLL_INTERVAL_MS", "200")),
            emotion_inference_interval_ms=int(os.getenv("SOUL_EMOTION_INFERENCE_INTERVAL_MS", "1000")),
            soul_model_url=os.getenv("SOUL_MODEL_URL", "http://localhost:8003/v1"),
            soul_model_name=os.getenv("SOUL_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct"),
            idle_breathing_enabled=os.getenv("SOUL_IDLE_BREATHING", "true").lower() == "true",
            idle_micro_movements_enabled=os.getenv("SOUL_IDLE_MICRO_MOVEMENTS", "true").lower() == "true",
            idle_scanning_enabled=os.getenv("SOUL_IDLE_SCANNING", "true").lower() == "true",
            personality=personality,
            debug_logging=os.getenv("SOUL_DEBUG", "false").lower() == "true",
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "poll_interval_ms": self.poll_interval_ms,
            "emotion_inference_interval_ms": self.emotion_inference_interval_ms,
            "soul_model_url": self.soul_model_url,
            "soul_model_name": self.soul_model_name,
            "idle_breathing_enabled": self.idle_breathing_enabled,
            "idle_micro_movements_enabled": self.idle_micro_movements_enabled,
            "idle_scanning_enabled": self.idle_scanning_enabled,
            "movement_blend_duration_s": self.movement_blend_duration_s,
            "personality": {
                "soul_file_path": self.personality.soul_file_path,
                "auto_reload": self.personality.auto_reload,
                "use_personality_expressions": self.personality.use_personality_expressions,
                "use_personality_prompts": self.personality.use_personality_prompts,
            },
        }
