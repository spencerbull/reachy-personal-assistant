"""
Vision node for handling image understanding requests.

This node processes the current camera image and answers questions about it.
"""

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig, REACHY_IDENTITY, REACHY_OUTPUT_RULES

VISION_SYSTEM_PROMPT = f"""{REACHY_IDENTITY} You have a camera and can see your surroundings.

{REACHY_OUTPUT_RULES}

When describing what you see:
- Be conversational and natural, 1-2 sentences
- Be specific about what you observe
- If asked about specific objects or actions, focus on those
- If you can't see something clearly, say so honestly

Your response will be spoken aloud - keep it brief and natural."""


def create_vision_llm(config: AgentConfig) -> ChatOpenAI:
    """Create the vision LLM instance."""
    return ChatOpenAI(
        model=config.models.vision_model,
        base_url=config.models.vision_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.vision_temperature,
    )


def build_vision_messages(state: ReachyAgentState, config: AgentConfig) -> list:
    """Build the message list for the vision LLM with image."""
    messages = [SystemMessage(content=VISION_SYSTEM_PROMPT)]
    
    current_image = state.get("current_image")
    
    # Get the user's question
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            user_question = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_question = msg.get("content", "")
            break
    
    if current_image:
        # Build multimodal message with image
        content = [
            {"type": "text", "text": user_question},
            {
                "type": "image_url",
                "image_url": {"url": current_image, "detail": "auto"}
            }
        ]
        messages.append(HumanMessage(content=content))
    else:
        # No image available
        messages.append(HumanMessage(
            content=f"{user_question}\n\n(Note: I don't have a current camera image available)"
        ))
    
    return messages


async def vision_node(state: ReachyAgentState, config: AgentConfig) -> StateUpdate:
    """
    Vision node for handling image understanding.
    
    Args:
        state: Current agent state
        config: Agent configuration
        
    Returns:
        State update with the vision analysis response
    """
    current_image = state.get("current_image")
    
    if not current_image:
        logger.warning("Vision node: No image available")
        # Provide a helpful response that acknowledges the vision request
        # and offers an alternative or asks them to wait
        return {
            "messages": [AIMessage(
                content="I'd love to help you with that, but my camera is still warming up. "
                        "Give me just a second and ask again!"
            )],
            "emotional_state": "attentive",
        }
    
    try:
        llm = create_vision_llm(config)
        messages = build_vision_messages(state, config)
        
        logger.info("Vision node: Processing image with VLM")
        
        response = await llm.ainvoke(messages)
        
        # Update emotional state - curious when observing
        reachy_state = state.get("reachy_state", {})
        reachy_state["emotion"] = "curious"
        
        logger.info("Vision node: Image analysis complete")
        
        return {
            "messages": [AIMessage(content=response.content)],
            "emotional_state": "curious",
            "reachy_state": reachy_state,
        }
        
    except Exception as e:
        logger.error(f"Vision node error: {e}")
        return {
            "messages": [AIMessage(
                content="I had some trouble analyzing what I see. Let me try again - what would you like me to look at?"
            )],
            "emotional_state": "curious",
        }
