"""
Spatial Memory Manager for object location tracking.

Enables Reachy to remember where objects are placed across different
locations as the user travels. Memory is organized by:
- Location (hotel_tokyo, home_office, etc.)
- Object name
- Position description
- Head direction when seen
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class SpatialMemoryEntry:
    """A single entry in spatial memory."""
    
    object_name: str
    location_description: str
    room_context: str = ""
    head_direction: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    last_confirmed: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SpatialMemoryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def age_days(self) -> float:
        """Get the age of this memory in days."""
        try:
            created = datetime.fromisoformat(self.timestamp)
            return (datetime.now() - created).days
        except:
            return 0


class SpatialMemoryManager:
    """
    Manages spatial memory for object location tracking.
    
    Supports:
    - Adding new object locations
    - Recalling object locations
    - Location context switching
    - Memory aging and cleanup
    """
    
    def __init__(self, store=None):
        """
        Initialize spatial memory manager.
        
        Args:
            store: Optional LangGraph Store for persistence
        """
        self._store = store
        self._memory: dict[str, dict[str, SpatialMemoryEntry]] = {}
        self._current_location: str = "unknown"
        self._namespace = ("spatial_memory",)
        
    @property
    def current_location(self) -> str:
        return self._current_location
    
    def set_location(self, location: str):
        """Set the current location context."""
        self._current_location = location.lower().replace(" ", "_")
        logger.info(f"Spatial memory location set to: {self._current_location}")
        
        # Ensure location exists in memory
        if self._current_location not in self._memory:
            self._memory[self._current_location] = {}
    
    async def remember(
        self,
        object_name: str,
        location_description: str,
        head_direction: Optional[str] = None,
    ) -> SpatialMemoryEntry:
        """
        Remember where an object is located.
        
        Args:
            object_name: Name of the object
            location_description: Where it is
            head_direction: Optional direction Reachy was looking
            
        Returns:
            The created memory entry
        """
        # Normalize object name
        object_key = object_name.lower().strip()
        
        # Create memory entry
        entry = SpatialMemoryEntry(
            object_name=object_name,
            location_description=location_description,
            room_context=self._current_location,
            head_direction=head_direction,
        )
        
        # Store in local memory
        if self._current_location not in self._memory:
            self._memory[self._current_location] = {}
        self._memory[self._current_location][object_key] = entry
        
        # Persist to store if available
        if self._store:
            try:
                key = f"{self._current_location}:{object_key}"
                await self._store.aput(self._namespace, key, entry.to_dict())
                logger.debug(f"Persisted spatial memory: {key}")
            except Exception as e:
                logger.warning(f"Failed to persist spatial memory: {e}")
        
        logger.info(f"Remembered: {object_name} at {location_description} in {self._current_location}")
        return entry
    
    async def recall(
        self,
        object_name: str,
        search_all_locations: bool = False,
    ) -> Optional[SpatialMemoryEntry]:
        """
        Recall where an object was last seen.
        
        Args:
            object_name: Name of the object to find
            search_all_locations: If True, search all locations
            
        Returns:
            Memory entry if found, None otherwise
        """
        object_key = object_name.lower().strip()
        
        # First check current location
        if self._current_location in self._memory:
            if object_key in self._memory[self._current_location]:
                entry = self._memory[self._current_location][object_key]
                logger.info(f"Recalled: {object_name} at {entry.location_description}")
                return entry
            
            # Try partial match
            for key, entry in self._memory[self._current_location].items():
                if object_key in key or key in object_key:
                    logger.info(f"Partial match: {object_name} -> {key}")
                    return entry
        
        # Search all locations if requested
        if search_all_locations:
            for location, objects in self._memory.items():
                if object_key in objects:
                    entry = objects[object_key]
                    logger.info(f"Found {object_name} in {location}")
                    return entry
                
                # Partial match
                for key, entry in objects.items():
                    if object_key in key or key in object_key:
                        logger.info(f"Partial match in {location}: {key}")
                        return entry
        
        # Try store if available
        if self._store:
            try:
                key = f"{self._current_location}:{object_key}"
                result = await self._store.aget(self._namespace, key)
                if result:
                    entry = SpatialMemoryEntry.from_dict(result.value)
                    # Cache locally
                    if self._current_location not in self._memory:
                        self._memory[self._current_location] = {}
                    self._memory[self._current_location][object_key] = entry
                    return entry
            except Exception as e:
                logger.warning(f"Failed to retrieve from store: {e}")
        
        logger.info(f"Object not found in memory: {object_name}")
        return None
    
    async def forget(self, object_name: str) -> bool:
        """
        Forget an object's location.
        
        Args:
            object_name: Name of the object
            
        Returns:
            True if forgotten, False if not found
        """
        object_key = object_name.lower().strip()
        
        if self._current_location in self._memory:
            if object_key in self._memory[self._current_location]:
                del self._memory[self._current_location][object_key]
                
                # Remove from store
                if self._store:
                    try:
                        key = f"{self._current_location}:{object_key}"
                        await self._store.adelete(self._namespace, key)
                    except Exception as e:
                        logger.warning(f"Failed to delete from store: {e}")
                
                logger.info(f"Forgot: {object_name}")
                return True
        
        return False
    
    async def list_objects(self, location: Optional[str] = None) -> list[SpatialMemoryEntry]:
        """
        List all remembered objects.
        
        Args:
            location: Optional location to filter by
            
        Returns:
            List of memory entries
        """
        location = location or self._current_location
        
        if location in self._memory:
            return list(self._memory[location].values())
        
        return []
    
    async def cleanup_old(self, max_age_days: int = 30):
        """
        Remove memories older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days
        """
        removed = 0
        for location, objects in list(self._memory.items()):
            for key, entry in list(objects.items()):
                if entry.age_days() > max_age_days:
                    del objects[key]
                    removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old spatial memories")
    
    def get_state(self) -> dict:
        """Get the current memory state for persistence."""
        return {
            "current_location": self._current_location,
            "memory": {
                loc: {k: v.to_dict() for k, v in objs.items()}
                for loc, objs in self._memory.items()
            }
        }
    
    def load_state(self, state: dict):
        """Load memory state from persistence."""
        self._current_location = state.get("current_location", "unknown")
        
        memory_data = state.get("memory", {})
        self._memory = {}
        
        for location, objects in memory_data.items():
            self._memory[location] = {}
            for key, entry_data in objects.items():
                self._memory[location][key] = SpatialMemoryEntry.from_dict(entry_data)
        
        logger.info(f"Loaded spatial memory: {sum(len(o) for o in self._memory.values())} objects")
