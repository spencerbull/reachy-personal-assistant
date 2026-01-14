"""
State schema for the Reachy Personal Assistant Agent.

Defines the shared state that flows through all nodes in the LangGraph.
"""

from typing import Annotated, Any, Optional, Literal
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph.message import add_messages


# Emotional states that Reachy can express
EmotionalState = Literal[
    "neutral",
    "happy", 
    "curious",
    "attentive",
    "thinking",
    "excited",
    "helpful",
    "playful"
]


@dataclass
class ReachyState:
    """Current physical and emotional state of the Reachy robot."""
    
    # Head pose (pitch, yaw in degrees)
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    
    # Body orientation (yaw in degrees)
    body_yaw: float = 0.0
    
    # Antenna positions (radians)
    left_antenna: float = 0.0
    right_antenna: float = 0.0
    
    # Current emotional state
    emotion: EmotionalState = "neutral"
    
    # Face tracking state
    face_tracking_enabled: bool = False
    
    # Last known face position (if tracking)
    face_position: Optional[tuple[float, float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for state serialization."""
        return {
            "head_pitch": self.head_pitch,
            "head_yaw": self.head_yaw,
            "body_yaw": self.body_yaw,
            "left_antenna": self.left_antenna,
            "right_antenna": self.right_antenna,
            "emotion": self.emotion,
            "face_tracking_enabled": self.face_tracking_enabled,
            "face_position": self.face_position,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReachyState":
        """Create from dictionary."""
        return cls(
            head_pitch=data.get("head_pitch", 0.0),
            head_yaw=data.get("head_yaw", 0.0),
            body_yaw=data.get("body_yaw", 0.0),
            left_antenna=data.get("left_antenna", 0.0),
            right_antenna=data.get("right_antenna", 0.0),
            emotion=data.get("emotion", "neutral"),
            face_tracking_enabled=data.get("face_tracking_enabled", False),
            face_position=data.get("face_position"),
        )


@dataclass
class SpatialMemoryEntry:
    """An entry in spatial memory tracking an object's location."""
    
    object_name: str
    location_description: str
    room_context: str = ""  # e.g., "hotel_tokyo", "home_office"
    head_direction: Optional[str] = None  # Direction Reachy was looking
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "object_name": self.object_name,
            "location_description": self.location_description,
            "room_context": self.room_context,
            "head_direction": self.head_direction,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SpatialMemoryEntry":
        return cls(**data)


class ReachyAgentState(TypedDict):
    """
    Main state schema for the Reachy Personal Assistant.
    
    This state flows through all nodes in the LangGraph and maintains:
    - Conversation history (messages)
    - Current camera image for vision
    - Robot physical/emotional state
    - Memory systems (spatial, preferences)
    - Routing decisions
    """
    
    # Conversation messages - uses add_messages reducer for proper merging
    messages: Annotated[list, add_messages]
    
    # Current image from Reachy's camera (base64 encoded)
    current_image: Optional[str]
    
    # Robot physical and emotional state
    reachy_state: dict  # Serialized ReachyState
    
    # Spatial memory for object locations
    # Structure: {"object_name": SpatialMemoryEntry.to_dict()}
    spatial_memory: dict
    
    # User preferences learned over time
    user_preferences: dict
    
    # Current emotional state (convenience accessor)
    emotional_state: EmotionalState
    
    # Current location context (for spatial memory)
    current_location: str
    
    # Routing decision from the router node
    route: Optional[str]
    
    # Tool results from the last tool execution
    tool_results: Optional[list[dict]]
    
    # Pending Reachy commands to execute
    pending_reachy_commands: list[dict]
    
    # Image generation state
    # Contains: phase, previous_messages, style_details, etc.
    image_gen_context: Optional[dict]
    
    # Base64-encoded source image captured when user presents it
    # This is stored separately from current_image to persist through follow-up turns
    captured_source_image: Optional[str]
    
    # Base64-encoded generated image from ComfyUI
    generated_image: Optional[str]
    
    # Vision description of the original input image
    original_image_description: Optional[str]


def create_initial_state() -> ReachyAgentState:
    """Create the initial state for a new conversation."""
    return ReachyAgentState(
        messages=[],
        current_image=None,
        reachy_state=ReachyState().to_dict(),
        spatial_memory={},
        user_preferences={},
        emotional_state="neutral",
        current_location="unknown",
        route=None,
        tool_results=None,
        pending_reachy_commands=[],
        image_gen_context=None,
        captured_source_image=None,
        generated_image=None,
        original_image_description=None,
    )


# Convenience type for node return values
StateUpdate = dict[str, Any]
