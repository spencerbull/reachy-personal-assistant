"""
Memory systems for the Reachy Personal Assistant.

Provides:
- Short-term memory: Conversation history via LangGraph checkpointers
- Long-term memory: User preferences and facts via LangGraph Store
- Spatial memory: Object locations with room/location context
- Emotional state: Current emotional state and expression management
"""

from agent.memory.spatial import SpatialMemoryManager
from agent.memory.long_term import LongTermMemoryManager
from agent.memory.emotional import (
    EmotionalStateManager,
    EmotionExpression,
    get_emotional_manager,
    set_emotional_state,
    get_emotional_state,
    EmotionalState,
)

__all__ = [
    "SpatialMemoryManager",
    "LongTermMemoryManager",
    "EmotionalStateManager",
    "EmotionExpression",
    "get_emotional_manager",
    "set_emotional_state",
    "get_emotional_state",
    "EmotionalState",
]
