"""
Image Generation Agent node.

This is a dedicated sub-agent for handling image style transfer requests.
It manages a multi-phase flow:
1. awaiting_image - Ask user to show the image
2. collecting_details - Capture image, ask about style
3. Generate when style is specified
"""

from typing import Optional

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig, REACHY_IDENTITY
from agent.tools.comfyui_tools import generate_image_tool

# Phases of the image generation flow
PHASE_AWAITING_IMAGE = "awaiting_image"
PHASE_COLLECTING_DETAILS = "collecting_details"


def create_vision_llm(config: AgentConfig) -> ChatOpenAI:
    """Create vision LLM for describing images."""
    return ChatOpenAI(
        model=config.models.vision_model,
        base_url=config.models.vision_model_base_url,
        api_key=config.models.api_key,
        temperature=0.3,
    )


def create_main_llm(config: AgentConfig) -> ChatOpenAI:
    """Create main LLM for prompt generation."""
    return ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=0.7,
    )


async def generate_detailed_prompt(
    image_description: str,
    user_style_request: str,
    config: AgentConfig
) -> str:
    """
    Use LLM to generate a detailed, professional prompt for image generation.
    
    Args:
        image_description: Description of the source image
        user_style_request: User's requested style/transformation
        config: Agent configuration
        
    Returns:
        A detailed prompt optimized for image generation
    """
    try:
        llm = create_main_llm(config)
        
        system_prompt = """You are an expert prompt engineer for image generation AI systems.
Your task is to create detailed, professional prompts that will produce high-quality images.

Given an image description and a user's style request, generate a comprehensive prompt that:
1. Preserves the key elements and composition from the original image
2. Applies the requested style transformation with specific details
3. Includes technical quality terms (high resolution, detailed, professional lighting, etc.)
4. Adds artistic direction appropriate for the style (color palette, mood, atmosphere)

Output ONLY the prompt text, nothing else. Make it 2-4 sentences, rich with descriptive details."""

        user_prompt = f"""Original image: {image_description}

User's style request: {user_style_request}

Generate a detailed image generation prompt that transforms the original into the requested style:"""

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        detailed_prompt = response.content.strip()
        logger.info(f"Generated detailed prompt: {detailed_prompt[:200]}...")
        return detailed_prompt
        
    except Exception as e:
        logger.error(f"Failed to generate detailed prompt: {e}")
        # Fallback to simple concatenation
        return f"{image_description}. Transform into {user_style_request}. High quality, detailed, professional rendering."


async def describe_image(image_b64: str, config: AgentConfig) -> str:
    """Use vision LLM to describe an image."""
    try:
        llm = create_vision_llm(config)
        
        # Build vision message
        content = [
            {"type": "text", "text": "Describe this image briefly in 1-2 sentences. Focus on the main subject and composition."},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]
        
        response = await llm.ainvoke([HumanMessage(content=content)])
        return response.content
    except Exception as e:
        logger.error(f"Failed to describe image: {e}")
        return "an image"


def get_user_message(state: ReachyAgentState) -> str:
    """Extract the latest user message from state."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def is_image_confirmation(message: str) -> bool:
    """Check if message indicates user is showing an image."""
    if not message:
        return False
    
    lower = message.lower()
    confirmation_phrases = [
        "here it is", "here you go", "take a look", "look at this",
        "it's right in front", "in front of you", "right in front",
        "here's the", "this is it", "okay here", "ok here",
        "yep", "yes", "yeah", "ready", "got it", "there you go",
        "showing you", "can you see", "do you see",
    ]
    
    return any(phrase in lower for phrase in confirmation_phrases)


def has_style_keyword(message: str) -> bool:
    """Check if the user message contains a style keyword."""
    if not message:
        return False
    
    lower = message.lower()
    style_keywords = [
        "3d", "render", "realistic", "painting", "cyberpunk", "anime",
        "watercolor", "oil", "cartoon", "sketch", "photorealistic",
        "futuristic", "vintage", "modern", "abstract", "minimalist",
        "neon", "retro", "steampunk", "fantasy", "sci-fi", "noir",
    ]
    
    return any(kw in lower for kw in style_keywords)


async def image_gen_node(
    state: ReachyAgentState,
    config: AgentConfig,
    tools: Optional[list] = None
) -> StateUpdate:
    """
    Image Generation Agent node.
    
    Manages a multi-phase flow:
    1. awaiting_image - Ask user to show the image
    2. collecting_details - Capture image, ask about style
    3. Generate when style is specified
    """
    current_image = state.get("current_image")
    captured_source_image = state.get("captured_source_image")
    image_gen_context = state.get("image_gen_context") or {}
    phase = image_gen_context.get("phase", PHASE_AWAITING_IMAGE)
    user_message = get_user_message(state)
    image_description = state.get("original_image_description", "")
    
    logger.info(f"Image gen node: Phase={phase}, has_captured={captured_source_image is not None}")
    
    try:
        # ===== PHASE 1: AWAITING IMAGE =====
        if phase == PHASE_AWAITING_IMAGE:
            if is_image_confirmation(user_message):
                # Capture the current image NOW
                if current_image:
                    logger.info("Image gen node: Capturing source image from user confirmation")
                    
                    # Describe what we see
                    image_description = await describe_image(current_image, config)
                    logger.info(f"Image gen node: Described image: {image_description[:100]}...")
                    
                    new_context = {
                        "phase": PHASE_COLLECTING_DETAILS,
                        "previous_messages": [],
                    }
                    
                    return {
                        "messages": [AIMessage(
                            content=f"I can see {image_description}. What style would you like me to transform this into? For example, realistic 3D render, oil painting, cyberpunk, or something else?"
                        )],
                        "captured_source_image": current_image,
                        "original_image_description": image_description,
                        "image_gen_context": new_context,
                        "emotional_state": "curious",
                    }
                else:
                    return {
                        "messages": [AIMessage(
                            content="I don't see an image yet. Could you hold it up in front of me and let me know when you're ready?"
                        )],
                        "image_gen_context": {"phase": PHASE_AWAITING_IMAGE},
                        "emotional_state": "attentive",
                    }
            else:
                # Initial request - ask user to show the image
                new_context = {
                    "phase": PHASE_AWAITING_IMAGE,
                    "previous_messages": [],
                }
                
                return {
                    "messages": [AIMessage(
                        content="Sure, I can help transform an image! Please show me what you'd like me to work with and let me know when you're ready."
                    )],
                    "image_gen_context": new_context,
                    "emotional_state": "helpful",
                }
        
        # ===== PHASE 2: COLLECTING DETAILS / GENERATING =====
        elif phase == PHASE_COLLECTING_DETAILS:
            source_image = captured_source_image or current_image
            
            if not source_image:
                logger.warning("Image gen node: No source image available")
                return {
                    "messages": [AIMessage(
                        content="I seem to have lost track of the image. Could you show it to me again?"
                    )],
                    "image_gen_context": {"phase": PHASE_AWAITING_IMAGE},
                    "emotional_state": "apologetic",
                }
            
            # Check if user specified a style
            if has_style_keyword(user_message):
                # User specified a style - generate the image!
                logger.info(f"Image gen node: Style detected, generating with: {user_message}")
                
                # Use LLM to generate a detailed, professional prompt
                logger.info("Image gen node: Generating detailed prompt with LLM...")
                generation_prompt = await generate_detailed_prompt(
                    image_description=image_description,
                    user_style_request=user_message,
                    config=config
                )
                
                logger.info(f"Image gen node: Calling generate_image_tool with prompt: {generation_prompt[:100]}...")
                logger.info("Image generation started - this may take up to 2 minutes...")
                
                try:
                    result = await generate_image_tool.ainvoke({
                        "prompt": generation_prompt,
                        "input_image_b64": source_image
                    })
                except Exception as tool_error:
                    logger.error(f"Tool execution error: {tool_error}", exc_info=True)
                    result = {
                        "success": False,
                        "image_url": None,
                        "message": f"Tool error: {str(tool_error)}"
                    }
                
                if result.get("success") and result.get("image_url"):
                    image_url = result["image_url"]
                    logger.info(f"Image generation successful! URL: {image_url}")
                    
                    # Format response - URL will be appended to transcript by bot
                    spoken_response = f"I've created your {user_message} image! What do you think?"
                    
                    # Full response includes URL for extraction
                    response_content = f"{spoken_response}\n\nImage URL: {image_url}"
                    
                    return {
                        "messages": [AIMessage(content=response_content)],
                        "generated_image": image_url,
                        "emotional_state": "excited",
                        "image_gen_context": None,  # Clear context after generation
                        "captured_source_image": None,  # Clear captured image
                    }
                else:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Image generation failed: {error_msg}")
                    return {
                        "messages": [AIMessage(
                            content=f"I had trouble generating that image. {error_msg} Want to try again with different settings?"
                        )],
                        "emotional_state": "helpful",
                    }
            else:
                # No style detected - ask for style
                logger.info("Image gen node: No style detected, asking user")
                return {
                    "messages": [AIMessage(
                        content="What style would you like me to transform this into? For example: realistic 3D render, oil painting, cyberpunk, or something else?"
                    )],
                    "emotional_state": "curious",
                }
        
        # Fallback
        return {
            "messages": [AIMessage(
                content="I can help you transform an image. What would you like me to do?"
            )],
            "emotional_state": "helpful",
        }
        
    except Exception as e:
        logger.error(f"Image gen node error: {e}", exc_info=True)
        return {
            "messages": [AIMessage(
                content="I had trouble processing that. Could you try again?"
            )],
            "emotional_state": "neutral",
        }
