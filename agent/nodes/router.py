"""
Router node for intent classification.

Determines which path the conversation should take:
- conversation: Simple chitchat and casual responses
- vision: Questions requiring image understanding
- tools: Requests requiring tool execution (movement, memory, etc.)
- image_gen: Requests for image transformation/generation
"""

import json
from typing import Literal

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig

# Route types
RouteType = Literal["conversation", "vision", "tools", "image_gen", "calendar", "email"]

ROUTER_SYSTEM_PROMPT = """You are a router that classifies user intents for a robot assistant named Reachy.

Analyze the user's message and determine the best route:

1. "conversation" - Simple chitchat, greetings, casual talk, jokes, general questions that don't need tools or vision
   Examples: "Hello", "Tell me a joke", "How are you?", "What can you do?"

2. "vision" - Questions requiring the robot to see/analyze the camera view
   Examples: "What do you see?", "What am I holding?", "Describe my surroundings", "What color is my shirt?"

3. "tools" - Requests requiring physical robot actions or memory operations
   Examples: 
   - Movement: "Look left", "Turn around", "Look at me"
   - Memory: "Remember I put my keys here", "Where did I put my passport?"
   - Emotions: "Show me you're happy", "Be excited"
   - Dance: "Dance for me", "Do a dance", "Show me your moves", "Celebrate!"

4. "image_gen" - Requests to transform, render, stylize, or generate images
   Examples: "Render my drawing", "Transform this into 3D", "Make it cyberpunk", "Create an image", "Style transfer", "Turn this into a painting", "Can you render this?"

5. "calendar" - Any request about calendar, schedule, meetings, events, or appointments
   Examples: "What's on my calendar?", "Check my schedule", "Am I free tomorrow?", "Create an event", "What meetings do I have?"

6. "email" - Any request about email, inbox, sending/reading messages
   Examples: "Check my email", "Do I have new emails?", "Send an email to John", "Read my latest email"

IMPORTANT ROUTING RULES:
- When in doubt, route to "conversation" — it's better to chat than to trigger the wrong action
- Only route to "vision" if the user is EXPLICITLY asking about what the robot can see
- Only route to "tools" if the user is giving a DIRECT command to the robot's body
- Only route to "image_gen" if the user is asking to render, transform, stylize, or generate an image
- Only route to "calendar" or "email" if the user EXPLICITLY mentions calendar/email/schedule/inbox
- Words like "hand", "face", "room" in normal conversation should go to "conversation", not "vision"
- "remember" in casual context ("remember when we...") goes to "conversation", not "tools"
- "create" without image context ("create a calendar event") does NOT go to "image_gen"

Respond with ONLY a JSON object in this exact format:
{"route": "conversation" | "vision" | "tools" | "image_gen" | "calendar" | "email", "reason": "brief explanation"}"""


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


def _is_clearly_image_related(lower_msg: str) -> bool:
    """Check if message is unambiguously about image generation."""
    image_phrases = [
        "render",
        "3d",
        "painting",
        "cyberpunk",
        "style",
        "transform",
        "here it is",
        "here you go",
        "show",
        "ready",
        "this image",
    ]
    return any(phrase in lower_msg for phrase in image_phrases)


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
            valid_routes = (
                "conversation",
                "vision",
                "tools",
                "image_gen",
                "calendar",
                "email",
            )
            if route not in valid_routes:
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

    # Universal flow-exit: let user escape any stateful sub-flow
    EXIT_PHRASES = [
        "cancel",
        "stop",
        "never mind",
        "nevermind",
        "forget it",
        "let's move on",
        "change topic",
        "something else",
        "that's enough",
        "no thanks",
        "not that",
        "skip",
        "abort",
    ]
    lower_msg = user_message.lower().strip()
    if any(phrase in lower_msg for phrase in EXIT_PHRASES):
        logger.info("Router: Flow exit detected, clearing stateful context")
        return {
            "route": "conversation",
            "image_gen_context": None,
            "captured_source_image": None,
            "original_image_description": None,
        }

    # If in image_gen flow, check if user is continuing that flow or changing topic
    # Bias toward staying in the flow — the image_gen node has its own exit check
    # via _seems_image_related() which is more granular per-phase
    image_gen_context = state.get("image_gen_context")
    captured_source_image = state.get("captured_source_image")
    if image_gen_context or captured_source_image:
        phase = image_gen_context.get("phase", "") if image_gen_context else ""

        # Stay in image_gen for: image-related terms, confirmations, style words,
        # short affirmatives, or anything the image_gen node can handle
        if _is_clearly_image_related(lower_msg):
            logger.info("Router: Continuing image_gen flow (image-related terms)")
            return {"route": "image_gen"}

        # Short confirmations / affirmatives should stay in the flow
        # (the image_gen node will handle them per-phase)
        _CONTINUATION_PHRASES = [
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "ready",
            "go ahead",
            "do it",
            "here",
            "got it",
        ]
        if lower_msg in _CONTINUATION_PHRASES or any(
            lower_msg.startswith(p) for p in _CONTINUATION_PHRASES
        ):
            logger.info("Router: Continuing image_gen flow (confirmation/affirmative)")
            return {"route": "image_gen"}

        # If the message looks like it's about something completely different
        # (explicit topic-change signals already caught by exit phrases above),
        # let the image_gen node decide — it has its own _seems_image_related check
        # and will clear context if appropriate
        logger.info(
            "Router: In image_gen context, routing to image_gen node for topic-change check"
        )
        return {"route": "image_gen"}

    # Fast path: only for unambiguous physical commands that can't be misinterpreted
    UNAMBIGUOUS_COMMANDS = {
        "tools": [
            "look left",
            "look right",
            "look up",
            "look down",
            "look at me",
            "turn left",
            "turn right",
            "turn around",
            "dance for me",
            "do a dance",
            "wave at me",
            "nod your head",
        ],
    }
    for route, commands in UNAMBIGUOUS_COMMANDS.items():
        if lower_msg in commands or any(lower_msg.startswith(cmd) for cmd in commands):
            logger.info(f"Router: Unambiguous command -> {route}")
            return {"route": route}

    # Fast path: image generation / style transfer requests
    IMAGE_GEN_PHRASES = [
        "render",
        "style transfer",
        "transform this",
        "transform my",
        "make it 3d",
        "make this 3d",
        "turn this into",
        "turn my drawing",
        "create a render",
        "generate an image",
        "generate image",
        "stylize",
        "make it cyberpunk",
        "make it anime",
        "make it realistic",
        "oil painting",
        "watercolor",
    ]
    if any(phrase in lower_msg for phrase in IMAGE_GEN_PHRASES):
        logger.info("Router: Fast-path image_gen (keyword match)")
        return {"route": "image_gen"}

    # LLM router is now the PRIMARY path
    try:
        llm = create_router_llm(config)

        # Include recent context for better classification
        recent_context = ""
        msgs = state.get("messages", [])
        if len(msgs) > 1:
            for msg in msgs[-3:]:  # Last 3 messages for context
                if hasattr(msg, "type"):
                    role = "User" if msg.type == "human" else "Reachy"
                    content = (
                        msg.content[:100] if len(msg.content) > 100 else msg.content
                    )
                    recent_context += f"\n{role}: {content}"

        context_note = (
            f"\nRecent conversation:{recent_context}" if recent_context else ""
        )

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"User message: {user_message}{context_note}"),
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


def should_route_to_image_gen(state: ReachyAgentState) -> bool:
    """Check if the route is image_gen."""
    return state.get("route") == "image_gen"


def get_next_node(state: ReachyAgentState) -> str:
    """Determine the next node based on the route."""
    route = state.get("route", "conversation")

    if route == "vision":
        return "vision"
    elif route == "tools":
        return "tools"
    elif route == "image_gen":
        return "image_gen"
    elif route == "calendar":
        return "calendar"
    elif route == "email":
        return "email"
    else:
        return "conversation"
