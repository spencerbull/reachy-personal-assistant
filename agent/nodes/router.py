"""
Router node for intent classification.

Determines which path the conversation should take:
- conversation: Simple chitchat and casual responses
- vision: Questions requiring image understanding
- tools: Requests requiring tool execution (movement, memory, etc.)
"""

import json
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig

logger = logging.getLogger(__name__)

# Route types
RouteType = Literal["conversation", "vision", "tools"]

ROUTER_SYSTEM_PROMPT = """You are a router that classifies user intents for a robot assistant named Reachy.

Analyze the user's message and determine the best route:

1. "conversation" - Simple chitchat, greetings, casual talk, jokes, general questions that don't need tools or vision
   Examples: "Hello", "Tell me a joke", "How are you?", "What can you do?"

2. "vision" - Questions requiring the robot to see/analyze the camera view
   Examples: "What do you see?", "What am I holding?", "Describe my surroundings", "What color is my shirt?"

3. "tools" - Requests requiring actions or memory operations
   Examples: 
   - Movement: "Look left", "Turn around", "Look at me"
   - Memory: "Remember I put my keys here", "Where did I put my passport?"
   - Emotions: "Show me you're happy", "Dance for me"
   - External: "What's on my calendar?", "Check my schedule"

IMPORTANT: If the message contains ANY indication of needing to see something, route to "vision".
If the message asks the robot to DO something physical or remember something, route to "tools".

Respond with ONLY a JSON object in this exact format:
{"route": "conversation" | "vision" | "tools", "reason": "brief explanation"}"""


def create_router_llm(config: AgentConfig) -> ChatOpenAI:
    """Create the router LLM instance."""
    return ChatOpenAI(
        model=config.models.router_model,
        base_url=config.models.router_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.router_temperature,
    )


def extract_user_message(state: ReachyAgentState) -> str:
    """Extract the latest user message from state."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        # Handle both dict and Message objects
        if hasattr(msg, "type"):
            if msg.type == "human":
                return msg.content
        elif isinstance(msg, dict):
            if msg.get("role") == "user" or msg.get("type") == "human":
                return msg.get("content", "")
    return ""


def parse_router_response(response: str) -> tuple[RouteType, str]:
    """Parse the router's JSON response."""
    try:
        # Try to extract JSON from the response
        # Handle cases where the model adds extra text
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            route = data.get("route", "conversation")
            reason = data.get("reason", "")
            
            # Validate route
            if route not in ("conversation", "vision", "tools"):
                logger.warning(f"Invalid route '{route}', defaulting to conversation")
                route = "conversation"
            
            return route, reason
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse router response: {e}")
    
    # Default fallback
    return "conversation", "Failed to parse, defaulting to conversation"


async def router_node(state: ReachyAgentState, config: AgentConfig) -> StateUpdate:
    """
    Router node that classifies the user's intent.
    
    Args:
        state: Current agent state
        config: Agent configuration
        
    Returns:
        State update with the route decision
    """
    user_message = extract_user_message(state)
    
    if not user_message:
        logger.warning("No user message found in state")
        return {"route": "conversation"}
    
    # Check for obvious patterns first (fast path)
    lower_msg = user_message.lower()
    
    # Vision keywords
    vision_keywords = [
        "what do you see", "what am i", "look at this", "describe",
        "what's in front", "what color", "what is this", "show me",
        "holding", "wearing", "surroundings", "environment", "scene"
    ]
    if any(kw in lower_msg for kw in vision_keywords):
        logger.info(f"Router: Fast path -> vision (keyword match)")
        return {"route": "vision"}
    
    # Tool keywords
    tool_keywords = [
        "look left", "look right", "look up", "look down", "look at me",
        "turn", "remember", "where did i", "where is my", "where are my",
        "calendar", "schedule", "dance", "happy", "sad", "excited",
        "face track", "follow me", "track my face"
    ]
    if any(kw in lower_msg for kw in tool_keywords):
        logger.info(f"Router: Fast path -> tools (keyword match)")
        return {"route": "tools"}
    
    # Use LLM for ambiguous cases
    try:
        llm = create_router_llm(config)
        
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"User message: {user_message}")
        ]
        
        response = await llm.ainvoke(messages)
        route, reason = parse_router_response(response.content)
        
        logger.info(f"Router: LLM decided '{route}' - {reason}")
        return {"route": route}
        
    except Exception as e:
        logger.error(f"Router LLM error: {e}")
        # Fallback to conversation on error
        return {"route": "conversation"}


def should_route_to_vision(state: ReachyAgentState) -> bool:
    """Check if the route is vision."""
    return state.get("route") == "vision"


def should_route_to_tools(state: ReachyAgentState) -> bool:
    """Check if the route is tools."""
    return state.get("route") == "tools"


def should_route_to_conversation(state: ReachyAgentState) -> bool:
    """Check if the route is conversation."""
    return state.get("route") == "conversation"


def get_next_node(state: ReachyAgentState) -> str:
    """Determine the next node based on the route."""
    route = state.get("route", "conversation")
    
    if route == "vision":
        return "vision"
    elif route == "tools":
        return "tools"
    else:
        return "conversation"
