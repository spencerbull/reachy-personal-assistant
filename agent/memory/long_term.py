"""
Long-term Memory Manager using LangGraph Store.

Enables Reachy to remember:
- User preferences
- Learned facts
- Important information across sessions
"""

import logging
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class MemoryFact:
    """A fact stored in long-term memory."""
    
    key: str
    value: Any
    category: str = "general"
    source: str = "user"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5  # 0-1 scale
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryFact":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LongTermMemoryManager:
    """
    Manages long-term memory for user preferences and facts.
    
    Uses LangGraph Store for persistence across sessions.
    Memory is organized by:
    - User ID
    - Category (preferences, facts, reminders, etc.)
    """
    
    # Memory categories
    CATEGORY_PREFERENCES = "preferences"
    CATEGORY_FACTS = "facts"
    CATEGORY_REMINDERS = "reminders"
    CATEGORY_HABITS = "habits"
    
    def __init__(self, store=None, user_id: str = "default"):
        """
        Initialize long-term memory manager.
        
        Args:
            store: LangGraph Store for persistence
            user_id: User identifier for scoping memory
        """
        self._store = store
        self._user_id = user_id
        self._cache: dict[str, MemoryFact] = {}
        
    @property
    def namespace(self) -> tuple:
        """Get the namespace for this user's memory."""
        return ("long_term_memory", self._user_id)
    
    def set_user(self, user_id: str):
        """Set the current user."""
        self._user_id = user_id
        self._cache.clear()
        logger.info(f"Long-term memory user set to: {user_id}")
    
    async def remember(
        self,
        key: str,
        value: Any,
        category: str = "general",
        importance: float = 0.5,
    ) -> MemoryFact:
        """
        Store a fact in long-term memory.
        
        Args:
            key: Identifier for the fact
            value: The value to remember
            category: Category for organization
            importance: How important this memory is (0-1)
            
        Returns:
            The created memory fact
        """
        fact = MemoryFact(
            key=key,
            value=value,
            category=category,
            source="user",
            importance=importance,
        )
        
        # Cache locally
        cache_key = f"{category}:{key}"
        self._cache[cache_key] = fact
        
        # Persist to store
        if self._store:
            try:
                await self._store.aput(self.namespace, cache_key, fact.to_dict())
                logger.debug(f"Persisted long-term memory: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to persist long-term memory: {e}")
        
        logger.info(f"Remembered [{category}]: {key} = {value}")
        return fact
    
    async def recall(
        self,
        key: str,
        category: Optional[str] = None,
    ) -> Optional[MemoryFact]:
        """
        Recall a fact from long-term memory.
        
        Args:
            key: The key to look up
            category: Optional category to narrow search
            
        Returns:
            The memory fact if found
        """
        # Check cache first
        if category:
            cache_key = f"{category}:{key}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        else:
            # Search all categories
            for cache_key, fact in self._cache.items():
                if cache_key.endswith(f":{key}"):
                    return fact
        
        # Check store
        if self._store:
            try:
                if category:
                    cache_key = f"{category}:{key}"
                    result = await self._store.aget(self.namespace, cache_key)
                    if result:
                        fact = MemoryFact.from_dict(result.value)
                        self._cache[cache_key] = fact
                        return fact
                else:
                    # Search all categories
                    for cat in [self.CATEGORY_PREFERENCES, self.CATEGORY_FACTS, 
                               self.CATEGORY_REMINDERS, self.CATEGORY_HABITS, "general"]:
                        cache_key = f"{cat}:{key}"
                        result = await self._store.aget(self.namespace, cache_key)
                        if result:
                            fact = MemoryFact.from_dict(result.value)
                            self._cache[cache_key] = fact
                            return fact
            except Exception as e:
                logger.warning(f"Failed to recall from store: {e}")
        
        return None
    
    async def recall_category(self, category: str) -> list[MemoryFact]:
        """
        Recall all facts in a category.
        
        Args:
            category: The category to retrieve
            
        Returns:
            List of memory facts
        """
        results = []
        
        # Check cache
        for cache_key, fact in self._cache.items():
            if cache_key.startswith(f"{category}:"):
                results.append(fact)
        
        # Check store for more
        if self._store:
            try:
                # Search by prefix
                items = await self._store.asearch(self.namespace, query=f"{category}:")
                for item in items:
                    fact = MemoryFact.from_dict(item.value)
                    cache_key = f"{fact.category}:{fact.key}"
                    if cache_key not in self._cache:
                        self._cache[cache_key] = fact
                        results.append(fact)
            except Exception as e:
                logger.debug(f"Store search not supported or failed: {e}")
        
        return results
    
    async def forget(self, key: str, category: Optional[str] = None) -> bool:
        """
        Forget a fact from long-term memory.
        
        Args:
            key: The key to forget
            category: Optional category
            
        Returns:
            True if forgotten
        """
        forgotten = False
        
        if category:
            cache_key = f"{category}:{key}"
            if cache_key in self._cache:
                del self._cache[cache_key]
                forgotten = True
            
            if self._store:
                try:
                    await self._store.adelete(self.namespace, cache_key)
                    forgotten = True
                except Exception as e:
                    logger.warning(f"Failed to delete from store: {e}")
        else:
            # Remove from all categories
            keys_to_remove = [k for k in self._cache if k.endswith(f":{key}")]
            for k in keys_to_remove:
                del self._cache[k]
                forgotten = True
        
        return forgotten
    
    async def set_preference(self, key: str, value: Any) -> MemoryFact:
        """Convenience method for storing preferences."""
        return await self.remember(key, value, self.CATEGORY_PREFERENCES, importance=0.7)
    
    async def get_preference(self, key: str, default: Any = None) -> Any:
        """Convenience method for getting preferences."""
        fact = await self.recall(key, self.CATEGORY_PREFERENCES)
        return fact.value if fact else default
    
    async def add_fact(self, key: str, value: Any) -> MemoryFact:
        """Convenience method for storing facts."""
        return await self.remember(key, value, self.CATEGORY_FACTS, importance=0.5)
    
    async def get_fact(self, key: str) -> Optional[Any]:
        """Convenience method for getting facts."""
        fact = await self.recall(key, self.CATEGORY_FACTS)
        return fact.value if fact else None
    
    def get_cache_state(self) -> dict:
        """Get current cache state for debugging."""
        return {k: v.to_dict() for k, v in self._cache.items()}
