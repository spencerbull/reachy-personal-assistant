"""
LangGraph LLM Service for Pipecat integration.

This module provides a Pipecat-compatible LLM service that uses the LangGraph
agent for processing messages, enabling stateful conversations with memory,
tool use, and vision understanding.
"""

import asyncio
import base64
import io
import os
import sys
from typing import Optional

from PIL import Image
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMMessagesFrame,
    LLMRunFrame,
    LLMTextFrame,
    UserImageRawFrame,
    OutputImageRawFrame,
    StartInterruptionFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserSpeakingFrame,
    BotSpeakingFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

# Add parent directory to path for agent imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import create_graph
from agent.config import AgentConfig

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class LangGraphLLMService(LLMService):
    """
    LangGraph-based LLM service for Pipecat.
    
    This service integrates the LangGraph agent with Pipecat's pipeline,
    handling:
    - Message context aggregation
    - Image capture and inclusion
    - Stateful conversation via thread IDs
    - Response streaming to TTS
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        max_image_dimension: int = 512,
        image_quality: int = 60,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._config = config or AgentConfig.from_env()
        self._graph = create_graph(config=self._config)
        
        # Image processing settings
        self._max_image_dimension = max_image_dimension
        self._image_quality = image_quality
        
        # State tracking
        self._user_id: Optional[str] = None
        self._thread_id: str = "default"
        self._last_image: Optional[UserImageRawFrame] = None
        self._current_turn_has_image: bool = False
        self._pending_image_future: Optional[asyncio.Future] = None
        
        logger.info(f"LangGraphLLMService initialized with {len(self._graph.nodes)} nodes")

    def set_user_id(self, user_id: str):
        """Set the user ID for image requests and thread identification."""
        self._user_id = user_id
        self._thread_id = f"user_{user_id}"
        logger.info(f"User ID set to {user_id}, thread_id: {self._thread_id}")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to stay within max dimension while preserving aspect ratio."""
        width, height = image.size
        max_dim = self._max_image_dimension

        if width <= max_dim and height <= max_dim:
            return image

        if width > height:
            new_width = max_dim
            new_height = int((max_dim / width) * height)
        else:
            new_height = max_dim
            new_width = int((max_dim / height) * width)

        return image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    def _encode_image_to_base64(self, frame: UserImageRawFrame) -> Optional[str]:
        """Encode UserImageRawFrame to base64 JPEG data URL."""
        try:
            image_format = getattr(frame, 'format', 'RGB')
            image_size = getattr(frame, 'size', (640, 360))
            
            image = Image.frombytes(image_format, image_size, frame.image)
            image = self._resize_image(image)

            # Convert to RGB if needed
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(
                    image,
                    mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None
                )
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._image_quality)
            image_bytes = buffer.getvalue()

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _convert_context_to_langchain(self, context) -> list:
        """Convert Pipecat LLM context to LangChain messages."""
        langchain_messages = []
        
        messages = context.messages if hasattr(context, 'messages') else context
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', '')
            
            # Handle multimodal content (extract text)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content = ' '.join(text_parts)
            
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
        return langchain_messages

    def _extract_user_message(self, messages: list) -> str:
        """Extract the latest user message from the list."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
            elif isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
        return ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the Pipecat pipeline."""
        
        # Skip logging for high-frequency audio/video frames
        high_freq_frames = (
            InputAudioRawFrame, OutputAudioRawFrame,
            UserImageRawFrame, OutputImageRawFrame,
            UserSpeakingFrame, BotSpeakingFrame
        )
        if not isinstance(frame, high_freq_frames):
            frame_name = type(frame).__name__
            if 'LLM' in frame_name or 'Context' in frame_name:
                logger.debug(f">>> LLM FRAME: {frame_name}")
            else:
                logger.debug(f"Processing {frame_name}")
        
        # Reset turn state on interruption
        if isinstance(frame, StartInterruptionFrame):
            self._current_turn_has_image = False
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return
        
        # OutputImageRawFrame should pass straight through to transport output
        if isinstance(frame, OutputImageRawFrame):
            if not hasattr(self, '_video_frame_count'):
                self._video_frame_count = 0
            self._video_frame_count += 1
            if self._video_frame_count == 1:
                logger.info("LangGraphLLMService: First OutputImageRawFrame received, passing through")
            if self._video_frame_count % 100 == 0:
                logger.debug(f"LangGraphLLMService: {self._video_frame_count} OutputImageRawFrames passed through")
            await self.push_frame(frame, direction)
            return
        
        # Capture incoming images for later use (for vision LLM)
        if isinstance(frame, UserImageRawFrame):
            self._last_image = frame
            
            if self._pending_image_future and not self._pending_image_future.done():
                logger.debug("Image captured for pending request")
                self._pending_image_future.set_result(frame)
            # Don't pass UserImageRawFrame downstream - it's for LLM vision only
            return
        
        # Log LLMRunFrame specifically - normally this is consumed by the upstream aggregator
        # but if it reaches us, we should pass it through
        if isinstance(frame, LLMRunFrame):
            logger.info(">>> LLMRunFrame received - this is unexpected, aggregator should have consumed it")
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Process LLM context frames
        if isinstance(frame, LLMContextFrame):
            context = frame.context
            messages = context.messages if hasattr(context, 'messages') else []
            
            logger.info(f"LLMContextFrame received with {len(messages)} messages")
            
            # Log message details for debugging
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        content_preview = content[:80] if len(content) > 80 else content
                        logger.info(f"  Message {i} ({role}): {content_preview}")
                    else:
                        logger.info(f"  Message {i} ({role}): multimodal content")
            
            # Check for user message or system prompt to respond
            has_user_message = any(
                (isinstance(msg, dict) and msg.get('role') == 'user') or
                (hasattr(msg, 'role') and msg.role == 'user')
                for msg in messages
            )
            
            # Check for "Say hello" type system messages (the second system message)
            has_greeting_prompt = False
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'system':
                    content = msg.get('content', '').lower()
                    if 'hello' in content or 'greet' in content:
                        has_greeting_prompt = True
                        break
            
            if not has_user_message and not has_greeting_prompt:
                logger.info("No user message or greeting prompt, skipping LLM call")
                # Pass through to allow context to build up downstream
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                return
            
            logger.info(f"Processing LLM request: has_user={has_user_message}, has_greeting={has_greeting_prompt}")
            
            # Process through LangGraph (don't pass frame downstream - we generate our own response)
            await self._process_with_langgraph(frame, context)
            return
        
        # Pass all other frames through unchanged - MUST explicitly push downstream
        # The base LLMService.process_frame() only handles InterruptionFrame/LLMConfigureOutputFrame
        # It does NOT pass frames downstream, so we must do it here
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
    
    async def _process_with_langgraph(self, frame: LLMContextFrame, context):
        """Process the context through the LangGraph agent."""
        try:
            # Convert context to LangChain messages
            messages = self._convert_context_to_langchain(context)
            user_message = self._extract_user_message(messages)
            
            # If no user message, check if this is a greeting prompt
            if not user_message:
                # Check for "Say hello" type prompts
                for msg in reversed(messages):
                    if isinstance(msg, SystemMessage) and 'hello' in msg.content.lower():
                        user_message = "Please greet me and introduce yourself."
                        logger.info("Detected greeting prompt, generating initial greeting")
                        break
            
            if not user_message:
                logger.warning("No message to process, skipping")
                return
            
            logger.info(f"Processing message: {user_message[:100]}...")
            
            # Prepare image if available
            image_data = None
            logger.debug(f"Image state: last_image={self._last_image is not None}, turn_has_image={self._current_turn_has_image}")
            if self._last_image and not self._current_turn_has_image:
                image_data = self._encode_image_to_base64(self._last_image)
                self._current_turn_has_image = True
                logger.info(f"Attached image to request (data length: {len(image_data) if image_data else 0})")
            elif not self._last_image:
                logger.warning("No image available from camera")
            
            # Build input state
            input_state = {
                "messages": [HumanMessage(content=user_message)],
                "current_image": image_data,
            }
            
            # Invoke the graph
            config = {"configurable": {"thread_id": self._thread_id}}
            
            logger.info("Invoking LangGraph agent...")
            result = await self._graph.ainvoke(input_state, config)
            
            # Extract the response
            response_text = ""
            pending_commands = []
            
            if "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage):
                        response_text = msg.content
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break
            
            # Get any pending Reachy commands
            if "pending_reachy_commands" in result:
                pending_commands = result["pending_reachy_commands"]
            
            logger.info(f"Agent response: {response_text[:100]}...")
            logger.info(f"Full response length: {len(response_text)} chars")
            
            # Send response through pipeline
            # Use LLMTextFrame (which extends TextFrame) - same as OpenAI LLM service
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=response_text))
            await self.push_frame(LLMFullResponseEndFrame())
            logger.info("Response frames pushed")
            
        except Exception as e:
            logger.error(f"LangGraph processing error: {e}", exc_info=True)
            
            # Send error response
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(
                text="I'm sorry, I had trouble processing that. Could you try again?"
            ))
            await self.push_frame(LLMFullResponseEndFrame())
