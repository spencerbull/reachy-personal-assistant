"""
Reachy Soul System - Continuous Embodiment Loop.

The Soul System provides Reachy with continuous physical presence by:
1. Running an async embodiment loop alongside the main agent
2. Inferring emotional state from conversation context
3. Generating smooth movement transitions between emotional states
4. Adding idle behaviors (breathing, micro-movements) for lifelike presence
5. Loading personality from REACHY_SOUL.md for consistent character

Key Components:
- SoulLoop: Main async loop that coordinates everything
- EmotionInference: LLM-based emotion detection from conversation
- MovementBlender: Smooth interpolation between poses
- IdleGenerator: Breathing, micro-movements, subtle life
- Personality: Character traits and expression guidelines from SOUL.md

Usage:
    from agent.soul import SoulLoop, SoulConfig, Personality

    config = SoulConfig()
    personality = Personality.load("REACHY_SOUL.md")
    soul = SoulLoop(config, reachy_service, personality=personality)

    # Start the soul (runs in background)
    await soul.start()

    # Feed conversation events
    soul.on_user_speaking()
    soul.on_user_message("Hello!")
    soul.on_bot_response("Hi there!")

    # Stop when done
    await soul.stop()
"""

from agent.soul.config import SoulConfig, PersonalityConfig
from agent.soul.loop import SoulLoop
from agent.soul.events import (
    SoulEvent,
    UserSpeakingEvent,
    UserSilentEvent,
    UserMessageEvent,
    BotResponseEvent,
    BotStreamingEvent,
    BotSpeakingStartedEvent,
    BotSpeakingStoppedEvent,
    FaceDetectedEvent,
    FaceLostEvent,
)
from agent.soul.emotion_inference import EmotionInference
from agent.soul.movement_blender import MovementBlender
from agent.soul.idle_generator import IdleGenerator
from agent.soul.personality import (
    Personality,
    PersonalityTraits,
    PhysicalExpression,
    ConversationalStyle,
    EmbodimentPrinciples,
    get_personality,
    reload_personality,
    set_personality_file,
)
from agent.soul.logging_utils import (
    SoulLogger,
    SoulLogCategory,
    configure_soul_logging,
    get_soul_logger,
)

__all__ = [
    # Main components
    "SoulConfig",
    "PersonalityConfig",
    "SoulLoop",
    "EmotionInference",
    "MovementBlender",
    "IdleGenerator",
    # Personality
    "Personality",
    "PersonalityTraits",
    "PhysicalExpression",
    "ConversationalStyle",
    "EmbodimentPrinciples",
    "get_personality",
    "reload_personality",
    "set_personality_file",
    # Events
    "SoulEvent",
    "UserSpeakingEvent",
    "UserSilentEvent",
    "UserMessageEvent",
    "BotResponseEvent",
    "BotStreamingEvent",
    "BotSpeakingStartedEvent",
    "BotSpeakingStoppedEvent",
    "FaceDetectedEvent",
    "FaceLostEvent",
    # Logging
    "SoulLogger",
    "SoulLogCategory",
    "configure_soul_logging",
    "get_soul_logger",
]
