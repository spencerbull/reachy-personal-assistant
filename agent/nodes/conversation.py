"""
Conversation node for handling chitchat and general responses.

This node handles casual conversation without requiring tools or vision.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig

logger = logging.getLogger(__name__)


def create_conversation_llm(config: AgentConfig) -> ChatOpenAI:
    """Create the conversation LLM instance."""
    return ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.main_temperature,
    )


def build_conversation_messages(state: ReachyAgentState, config: AgentConfig) -> list:
    """Build the message list for the conversation LLM."""
    messages = [SystemMessage(content=config.system_prompt)]
    
    # Add conversation history
    for msg in state.get("messages", []):
        messages.append(msg)
    
    return messages


def determine_emotional_response(user_message: str, response: str) -> str:
    """Determine appropriate emotional state based on conversation."""
    lower_msg = user_message.lower()
    lower_resp = response.lower()
    
    # Greeting - be happy and attentive
    if any(word in lower_msg for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "happy"
    
    # Jokes and fun - be playful
    if any(word in lower_msg for word in ["joke", "funny", "laugh", "fun"]):
        return "playful"
    
    # Questions - be curious/thinking
    if "?" in user_message:
        return "curious"
    
    # Thanks - be happy
    if any(word in lower_msg for word in ["thank", "thanks", "appreciate"]):
        return "happy"
    
    # Serious topics - be attentive
    if any(word in lower_msg for word in ["important", "serious", "help me", "need"]):
        return "attentive"
    
    # Excitement in response - be excited
    if any(word in lower_resp for word in ["awesome", "great", "exciting", "love"]):
        return "excited"
    
    return "helpful"


async def conversation_node(state: ReachyAgentState, config: AgentConfig) -> StateUpdate:
    """
    Conversation node for handling chitchat.
    
    Args:
        state: Current agent state
        config: Agent configuration
        
    Returns:
        State update with the assistant's response
    """
    try:
        llm = create_conversation_llm(config)
        messages = build_conversation_messages(state, config)
        
        logger.info(f"Conversation node: Processing with {len(messages)} messages")
        
        response = await llm.ainvoke(messages)
        
        # Extract user message for emotional analysis
        user_message = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Determine emotional state
        emotional_state = determine_emotional_response(user_message, response.content)
        
        # Update reachy state with new emotion
        reachy_state = state.get("reachy_state", {})
        reachy_state["emotion"] = emotional_state
        
        logger.info(f"Conversation node: Response generated, emotion: {emotional_state}")
        
        return {
            "messages": [AIMessage(content=response.content)],
            "emotional_state": emotional_state,
            "reachy_state": reachy_state,
        }
        
    except Exception as e:
        logger.error(f"Conversation node error: {e}")
        return {
            "messages": [AIMessage(content="I'm sorry, I had trouble processing that. Could you try again?")],
            "emotional_state": "neutral",
        }
