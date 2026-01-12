"""
Memory tools for spatial and long-term memory.

These tools allow Reachy to remember and recall information about:
- Object locations (spatial memory)
- User preferences
- Learned facts
"""

import logging
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# In-memory storage (will be replaced with persistent storage via LangGraph Store)
_spatial_memory: dict[str, dict] = {}
_current_location: str = "unknown"


def set_current_location(location: str):
    """Set the current location context."""
    global _current_location
    _current_location = location
    logger.info(f"Current location set to: {location}")


def get_current_location() -> str:
    """Get the current location context."""
    return _current_location


@tool
def remember_location_tool(
    object_name: str,
    location_description: str,
    direction_looking: Optional[str] = None,
) -> str:
    """
    Remember where an object is located.
    
    Use this when the user tells you where they're putting something,
    or when you see where something is located. This creates spatial
    memory that can be recalled later.
    
    Args:
        object_name: Name of the object (e.g., "keys", "passport", "wallet")
        location_description: Description of where it is (e.g., "on the nightstand", "in the top drawer")
        direction_looking: Optional direction you were looking when you saw it
        
    Returns:
        Confirmation that the location was remembered
    """
    object_key = object_name.lower().strip()
    
    memory_entry = {
        "object_name": object_name,
        "location_description": location_description,
        "room_context": _current_location,
        "direction_looking": direction_looking,
        "timestamp": datetime.now().isoformat(),
        "confidence": 1.0,
    }
    
    _spatial_memory[object_key] = memory_entry
    
    logger.info(f"remember_location_tool: Stored '{object_name}' at '{location_description}'")
    
    location_context = ""
    if _current_location != "unknown":
        location_context = f" here in {_current_location}"
    
    return f"Got it! I'll remember that your {object_name} is {location_description}{location_context}."


@tool
def recall_location_tool(object_name: str) -> str:
    """
    Recall where an object was last seen or placed.
    
    Use this when the user asks where something is. If you remember
    where it is, you can also look in that direction to help them find it.
    
    Args:
        object_name: Name of the object to find
        
    Returns:
        Location information if known, or indication that it's not remembered
    """
    object_key = object_name.lower().strip()
    
    # Check exact match first
    if object_key in _spatial_memory:
        entry = _spatial_memory[object_key]
        location = entry["location_description"]
        room = entry.get("room_context", "")
        direction = entry.get("direction_looking")
        timestamp = entry.get("timestamp", "")
        
        # Build response
        response_parts = [f"Your {object_name} is {location}"]
        
        if room and room != "unknown":
            response_parts.append(f" in {room}")
        
        response_parts.append(".")
        
        # Add direction hint if available
        if direction:
            response_parts.append(f" It should be to my {direction}. [CMD_LOOK_{direction.upper()}]")
        
        logger.info(f"recall_location_tool: Found '{object_name}' at '{location}'")
        return "".join(response_parts)
    
    # Try partial match
    for key, entry in _spatial_memory.items():
        if object_key in key or key in object_key:
            location = entry["location_description"]
            logger.info(f"recall_location_tool: Partial match '{object_name}' -> '{key}'")
            return f"I think you might mean your {entry['object_name']}. It's {location}."
    
    logger.info(f"recall_location_tool: '{object_name}' not found in memory")
    return f"I don't remember where your {object_name} is. Did you tell me where you put it?"


@tool
def list_remembered_objects_tool() -> str:
    """
    List all objects currently in spatial memory.
    
    Use this to check what objects you remember the location of.
    
    Returns:
        List of remembered objects and their locations
    """
    if not _spatial_memory:
        return "I don't have any objects in my memory right now."
    
    items = []
    for key, entry in _spatial_memory.items():
        items.append(f"- {entry['object_name']}: {entry['location_description']}")
    
    return "Here's what I remember:\n" + "\n".join(items)


@tool
def forget_object_tool(object_name: str) -> str:
    """
    Forget the location of an object.
    
    Use this when the user says they've moved something or it's no longer
    in the remembered location.
    
    Args:
        object_name: Name of the object to forget
        
    Returns:
        Confirmation that the object was forgotten
    """
    object_key = object_name.lower().strip()
    
    if object_key in _spatial_memory:
        del _spatial_memory[object_key]
        logger.info(f"forget_object_tool: Forgot '{object_name}'")
        return f"Okay, I've forgotten where your {object_name} was."
    
    return f"I didn't have a memory of where your {object_name} was anyway."


@tool
def set_location_context_tool(location_name: str) -> str:
    """
    Set the current location context for spatial memory.
    
    Use this when you arrive at a new location (hotel room, office, home, etc.)
    This helps organize spatial memories by location.
    
    Args:
        location_name: Name of the current location (e.g., "hotel_tokyo", "home_office")
        
    Returns:
        Confirmation of the location update
    """
    set_current_location(location_name)
    return f"Got it! I'll remember that we're now in {location_name}. Any objects you tell me about will be associated with this location."


def get_all_memory_tools():
    """Get all memory tools as a list."""
    return [
        remember_location_tool,
        recall_location_tool,
        list_remembered_objects_tool,
        forget_object_tool,
        set_location_context_tool,
    ]


def get_spatial_memory_state() -> dict:
    """Get the current spatial memory state for persistence."""
    return {
        "current_location": _current_location,
        "objects": _spatial_memory.copy(),
    }


def load_spatial_memory_state(state: dict):
    """Load spatial memory state from persistence."""
    global _current_location, _spatial_memory
    _current_location = state.get("current_location", "unknown")
    _spatial_memory = state.get("objects", {})
    logger.info(f"Loaded spatial memory: {len(_spatial_memory)} objects")
