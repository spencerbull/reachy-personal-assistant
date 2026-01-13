"""
Router node for intent classification.

Determines which path the conversation should take:
- conversation: Simple chitchat and casual responses
- vision: Questions requiring image understanding
- tools: Requests requiring tool execution (movement, memory, etc.)
"""

import json
from typing import Literal

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig

# Route types
RouteType = Literal["conversation", "vision", "tools"]

ROUTER_SYSTEM_PROMPT = """You are a router that classifies user intents for a robot assistant named Reachy.

Analyze the user's message and determine the best route:

1. "conversation" - Simple chitchat, greetings, casual talk, jokes, general questions that don't need tools or vision
   Examples: "Hello", "Tell me a joke", "How are you?", "What can you do?"

2. "vision" - Questions requiring the robot to see/analyze the camera view
   Examples: "What do you see?", "What am I holding?", "Describe my surroundings", "What color is my shirt?"

3. "tools" - Requests requiring actions, memory operations, or external services
   Examples: 
   - Movement: "Look left", "Turn around", "Look at me"
   - Memory: "Remember I put my keys here", "Where did I put my passport?"
   - Emotions: "Show me you're happy", "Be excited"
   - Dance: "Dance for me", "Do a dance", "Show me your moves", "Celebrate!", "Do something silly"
   - Calendar: "What's on my calendar?", "Check my schedule", "Am I free tomorrow at 3pm?", "Create an event", "What meetings do I have today?"
   - Email: "Check my email", "Do I have new emails?", "Send an email to John", "Read my latest email"

IMPORTANT: If the message contains ANY indication of needing to see something, route to "vision".
If the message asks the robot to DO something physical, remember something, or access external services (email, calendar), route to "tools".

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
    
    # Vision keywords - anything that requires seeing/analyzing visual input
    vision_keywords = [
        # Direct vision requests
        "what do you see", "what can you see", "what am i", "look at this",
        "describe what", "describe the", "describe my", "describe this",
        "what's in front", "what is in front",
        # Appearance questions
        "what color", "what colour", "how many", "count the", "count my",
        # Object/person identification
        "what is this", "what is that", "what are these", "what are those",
        "who is", "who am i", "who's there",
        # Actions the user is performing
        "holding", "wearing", "doing", "showing you",
        # Environment description
        "surroundings", "environment", "scene", "room", "around you",
        # Hand/body related vision
        "fingers", "hand", "hands", "face", "shirt", "clothes",
        # General visual queries
        "can you see", "do you see", "see my", "see this", "see the",
        "read this", "read the", "read my",
    ]
    if any(kw in lower_msg for kw in vision_keywords):
        logger.info(f"Router: Fast path -> vision (keyword match)")
        return {"route": "vision"}
    
    # Tool keywords - physical actions, memory, and external services
    tool_keywords = [
        # Head movement
        "look left", "look right", "look up", "look down", "look at me",
        "look over", "look toward", "look towards",
        # Body movement
        "turn left", "turn right", "turn around", "spin",
        # Memory operations
        "remember", "don't forget", "memorize",
        "where did i", "where is my", "where are my", "where's my",
        "find my", "locate my",
        # Face tracking
        "face track", "follow me", "track my face", "watch me",
        "maintain eye contact", "eye contact",
        # Emotions and expressions
        "show me you're", "express", "be happy", "be sad", "be excited",
        # Dance and movement requests
        "dance", "dancing", "celebrate", "wave", "nod", "shake your head",
        "show me a move", "do a move", "do a dance", "groove", "boogie",
        "headbang", "sway", "spin around", "bust a move", "show off",
        # Calendar
        "calendar", "schedule", "appointment", "meeting",
        "what's on my calendar", "check my calendar", "my events",
        "upcoming events", "events today", "events tomorrow", "events this week",
        "create event", "add event", "schedule event", "book a meeting",
        "free time", "am i free", "am i available", "availability",
        "when am i free", "find a time", "busy", "freebusy",
        # Email/Gmail
        "email", "gmail", "inbox", "send email", "send an email",
        "read email", "read my email", "check email", "check my email",
        "unread email", "new email", "latest email", "recent email",
        "mail", "mailbox", "message from", "email from",
        "reply to", "forward email", "compose email", "draft email",
        "search email", "find email",
        # General reminders
        "reminder",
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
