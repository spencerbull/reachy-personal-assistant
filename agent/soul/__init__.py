"""
Reachy Soul System - Continuous Embodiment Loop.

The Soul System provides Reachy with continuous physical presence by:
1. Running an async embodiment loop alongside the main agent
2. Inferring emotional state from conversation context
3. Generating smooth movement transitions between emotional states
4. Adding idle behaviors (breathing, micro-movements) for lifelike presence

Key Components:
- SoulLoop: Main async loop that coordinates everything
- EmotionInference: LLM-based emotion detection from conversation
- MovementBlender: Smooth interpolation between poses
- IdleGenerator: Breathing, micro-movements, subtle life

Usage:
    from agent.soul import SoulLoop, SoulConfig
    
    config = SoulConfig()
    soul = SoulLoop(config, reachy_service)
    
    # Start the soul (runs in background)
    await soul.start()
    
    # Feed conversation events
    soul.on_user_speaking()
    soul.on_user_message("Hello!")
    soul.on_bot_response("Hi there!")
    
    # Stop when done
    await soul.stop()
"""

from agent.soul.config import SoulConfig
from agent.soul.loop import SoulLoop
from agent.soul.events import (
    SoulEvent,
    UserSpeakingEvent,
    UserSilentEvent,
    UserMessageEvent,
    BotResponseEvent,
    BotStreamingEvent,
    FaceDetectedEvent,
    FaceLostEvent,
)
from agent.soul.emotion_inference import EmotionInference
from agent.soul.movement_blender import MovementBlender
from agent.soul.idle_generator import IdleGenerator

__all__ = [
    # Main components
    "SoulConfig",
    "SoulLoop",
    "EmotionInference",
    "MovementBlender",
    "IdleGenerator",
    # Events
    "SoulEvent",
    "UserSpeakingEvent",
    "UserSilentEvent",
    "UserMessageEvent",
    "BotResponseEvent",
    "BotStreamingEvent",
    "FaceDetectedEvent",
    "FaceLostEvent",
]
