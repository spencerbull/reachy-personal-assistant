"""
Conversation node for handling chitchat and general responses.

This node handles casual conversation without requiring tools or vision.
"""

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage


def create_conversation_llm(config):
    """Create the conversation LLM instance."""
    return ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.main_temperature,
    )


def build_conversation_messages(state, config):
    """Build the message list for the conversation LLM."""
    messages = [SystemMessage(content=config.system_prompt)]

    # Add conversation history
    for msg in state.get("messages", []):
        messages.append(msg)

    return messages


async def conversation_node(state, config):
    try:
        llm = create_conversation_llm(config)
        messages = build_conversation_messages(state, config)
        response = await llm.ainvoke(messages)
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"Conversation node error: {e}")
        import random

        error_responses = [
            "Hmm, something went sideways on my end. Want to try that again?",
            "I got a bit tangled up processing that. Could you rephrase?",
            "My circuits got crossed on that one. What were you saying?",
            "I stumbled on that request. Mind giving it another go?",
        ]
        return {"messages": [AIMessage(content=random.choice(error_responses))]}
