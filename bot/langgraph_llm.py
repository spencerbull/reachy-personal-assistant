"""
LangGraph LLM Service for Pipecat integration.

This module provides a Pipecat-compatible LLM service that uses the LangGraph
agent for processing messages, enabling stateful conversations with memory,
tool use, and vision understanding.

Also integrates with the Soul System (agent/soul/) for continuous embodiment:
- Emotion inference from conversation context
- Smooth movement blending
- Idle behaviors (breathing, micro-movements)
"""

import asyncio
import base64
import io
import os
import re
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
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotSpeakingFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

# Add parent directory to path for agent imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import create_graph, create_graph_with_mcp
from agent.config import AgentConfig
from agent.tools.mcp_loader import (
    get_all_mcp_configs,
    MCPToolLoader,
    is_gmail_configured,
)
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Soul System integration for continuous embodiment
try:
    from agent.soul import SoulLoop, SoulConfig

    SOUL_AVAILABLE = True
except ImportError:
    SOUL_AVAILABLE = False
    SoulLoop = None
    SoulConfig = None

# Movement mode for setting PROCESSING mode during LLM inference
try:
    from services.moves import MovementMode
except ImportError:
    try:
        from bot.services.moves import MovementMode
    except ImportError:
        MovementMode = None


class LangGraphLLMService(LLMService):
    """
    LangGraph-based LLM service for Pipecat.

    This service integrates the LangGraph agent with Pipecat's pipeline,
    handling:
    - Message context aggregation
    - Image capture and inclusion
    - Stateful conversation via thread IDs
    - Response streaming to TTS
    - Soul System for continuous embodiment (emotion, movement, idle behaviors)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        max_image_dimension: int = 512,
        image_quality: int = 60,
        enable_mcp: bool = True,
        mcp_tools: Optional[list] = None,
        enable_soul: bool = True,
        soul_config: Optional["SoulConfig"] = None,
        reachy_service: Optional[object] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._config = config or AgentConfig.from_env()
        self._enable_mcp = enable_mcp
        self._mcp_tools = mcp_tools or []
        self._mcp_loader = None
        self._mcp_initialized = False

        # Create checkpointer for state persistence across turns
        self._checkpointer = MemorySaver()

        # Create initial graph with checkpointer (MCP tools loaded in initialize_mcp())
        self._graph = create_graph(
            config=self._config,
            checkpointer=self._checkpointer,
            additional_tools=self._mcp_tools,
        )

        # Image processing settings
        self._max_image_dimension = max_image_dimension
        self._image_quality = image_quality

        # State tracking
        self._user_id: Optional[str] = None
        self._thread_id: str = "default"
        self._last_image: Optional[UserImageRawFrame] = None
        self._current_turn_has_image: bool = False
        self._pending_image_future: Optional[asyncio.Future] = None
        self._transport = None  # Transport for sending chat messages directly
        self._rtvi_processor = None  # RTVI processor for sending server messages

        # Interruption / cancellation tracking (Phase 3)
        # Monotonically increasing generation ID. Each LLM invocation snapshots
        # the current value; if an interruption bumps it before the response is
        # pushed to TTS, the response is considered stale and discarded.
        self._generation_id: int = 0
        self._inflight_task: Optional[asyncio.Task] = None

        # Soul System integration for continuous embodiment
        self._enable_soul = enable_soul and SOUL_AVAILABLE
        self._soul: Optional["SoulLoop"] = None
        self._reachy_service = reachy_service

        if self._enable_soul and SOUL_AVAILABLE:
            soul_cfg = soul_config or SoulConfig.from_env()
            self._soul = SoulLoop(
                config=soul_cfg,
                reachy_service=reachy_service,
            )
            logger.info("Soul System enabled for continuous embodiment")
        elif enable_soul and not SOUL_AVAILABLE:
            logger.warning("Soul System requested but not available (import failed)")

        # Log MCP status
        if is_gmail_configured():
            logger.info("Gmail MCP is configured - tools will be loaded on first use")
        else:
            logger.info(
                "Gmail MCP not configured (run: npx @gongrzhe/server-gmail-autoauth-mcp auth)"
            )

        logger.info(
            f"LangGraphLLMService initialized with {len(self._graph.nodes)} nodes"
        )

    async def start_soul(self):
        """
        Start the Soul System embodiment loop.

        Call this when the pipeline starts to enable continuous presence.
        The soul loop runs in the background, managing:
        - Emotion inference from conversation
        - Movement blending for smooth transitions
        - Idle behaviors (breathing, micro-movements)
        """
        if self._soul and not self._soul.is_running:
            await self._soul.start()
            logger.info("Soul System started")

    async def stop_soul(self):
        """Stop the Soul System embodiment loop."""
        if self._soul:
            try:
                await self._soul.stop()
                logger.info("Soul System stopped")
            except asyncio.CancelledError:
                logger.info("Soul System stop cancelled")
                raise
            except Exception as e:
                logger.warning(f"Soul System stop error: {e}")

    async def cleanup(self):
        """
        Clean up resources. Call this on shutdown.

        This method is safe to call multiple times and handles
        cancellation gracefully.
        """
        logger.info("LangGraphLLMService cleanup starting...")

        # Stop soul system
        try:
            await self.stop_soul()
        except asyncio.CancelledError:
            # Force cancel the soul task if we're being cancelled
            if self._soul and self._soul._task and not self._soul._task.done():
                self._soul._task.cancel()
            raise
        except Exception as e:
            logger.warning(f"Soul cleanup error: {e}")

        # Close MCP connections
        if self._mcp_loader:
            try:
                await self._mcp_loader.close()
                logger.info("MCP connections closed")
            except Exception as e:
                logger.warning(f"MCP cleanup error: {e}")

        logger.info("LangGraphLLMService cleanup complete")

    def set_reachy_service(self, service):
        """Set or update the Reachy service for soul movement commands."""
        self._reachy_service = service
        if self._soul:
            self._soul._reachy_service = service
        logger.info("Reachy service connected to Soul System")

    def get_soul_status(self) -> Optional[dict]:
        """Get the current soul loop status for debugging."""
        if self._soul:
            return self._soul.get_status()
        return None

    async def initialize_mcp(self):
        """Load MCP tools asynchronously. Call this before first use."""
        if self._mcp_initialized or not self._enable_mcp:
            return

        try:
            # Get all configured MCP servers
            mcp_configs = get_all_mcp_configs()
            enabled_configs = [c for c in mcp_configs if c.enabled]

            if not enabled_configs:
                logger.info("No MCP servers enabled")
                self._mcp_initialized = True
                return

            logger.info(f"Loading MCP tools from {len(enabled_configs)} servers...")

            # Load tools from MCP servers
            loader = MCPToolLoader(mcp_configs)
            mcp_tools = await loader.load_tools()

            if mcp_tools:
                logger.info(f"Loaded {len(mcp_tools)} MCP tools:")
                for tool in mcp_tools:
                    logger.info(f"  - {tool.name}")

                # Recreate graph with MCP tools and checkpointer
                all_tools = self._mcp_tools + mcp_tools
                self._graph = create_graph(
                    config=self._config,
                    checkpointer=self._checkpointer,
                    additional_tools=all_tools,
                )
                self._mcp_loader = loader
            else:
                logger.info("No MCP tools loaded")

            self._mcp_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            self._mcp_initialized = True  # Don't retry on failure

    def set_user_id(self, user_id: str):
        """Set the user ID for image requests and thread identification."""
        self._user_id = user_id
        self._thread_id = f"user_{user_id}"
        logger.info(f"User ID set to {user_id}, thread_id: {self._thread_id}")

    def set_transport(self, transport):
        """Set the transport for sending chat messages directly."""
        self._transport = transport
        logger.info("Transport configured for direct chat messaging")

    def set_rtvi_processor(self, rtvi):
        """Set the RTVI processor for sending server messages."""
        self._rtvi_processor = rtvi
        logger.info("RTVI processor configured for server messaging")

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
            image_format = getattr(frame, "format", "RGB")
            image_size = getattr(frame, "size", (640, 360))

            image = Image.frombytes(image_format, image_size, frame.image)
            image = self._resize_image(image)

            # Convert to RGB if needed
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                background.paste(
                    image,
                    mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None,
                )
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._image_quality)
            image_bytes = buffer.getvalue()

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _convert_context_to_langchain(self, context) -> list:
        """Convert Pipecat LLM context to LangChain messages."""
        langchain_messages = []

        messages = context.messages if hasattr(context, "messages") else context

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")

            # Handle multimodal content (extract text)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        return langchain_messages

    def _extract_user_message(self, messages: list) -> str:
        """Extract the latest user message from the list."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the Pipecat pipeline."""

        # Skip logging for high-frequency audio/video frames
        high_freq_frames = (
            InputAudioRawFrame,
            OutputAudioRawFrame,
            UserImageRawFrame,
            OutputImageRawFrame,
            UserSpeakingFrame,
            BotSpeakingFrame,
        )
        if not isinstance(frame, high_freq_frames):
            frame_name = type(frame).__name__
            if "LLM" in frame_name or "Context" in frame_name:
                logger.debug(f">>> LLM FRAME: {frame_name}")
            else:
                logger.debug(f"Processing {frame_name}")

        # Reset turn state on interruption
        if isinstance(frame, StartInterruptionFrame):
            self._current_turn_has_image = False
            # Bump generation ID so any in-flight LLM result is marked stale
            self._generation_id += 1
            # Cancel in-flight LangGraph task if running
            if self._inflight_task and not self._inflight_task.done():
                self._inflight_task.cancel()
                logger.info("Interruption: cancelled in-flight LangGraph task")
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # OutputImageRawFrame should pass straight through to transport output
        if isinstance(frame, OutputImageRawFrame):
            if not hasattr(self, "_video_frame_count"):
                self._video_frame_count = 0
            self._video_frame_count += 1
            if self._video_frame_count == 1:
                logger.info(
                    "LangGraphLLMService: First OutputImageRawFrame received, passing through"
                )
            if self._video_frame_count % 100 == 0:
                logger.debug(
                    f"LangGraphLLMService: {self._video_frame_count} OutputImageRawFrames passed through"
                )
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

        # Feed conversation events to Soul System
        # Use one-shot frames (UserStarted/UserStopped) instead of high-frequency
        # UserSpeakingFrame to avoid flooding the soul event queue.
        if isinstance(frame, UserStartedSpeakingFrame) and self._soul:
            self._soul.on_user_speaking()

        if isinstance(frame, UserStoppedSpeakingFrame) and self._soul:
            self._soul.on_user_silent()

        # Log LLMRunFrame specifically - normally this is consumed by the upstream aggregator
        # but if it reaches us, we should pass it through
        if isinstance(frame, LLMRunFrame):
            logger.info(
                ">>> LLMRunFrame received - this is unexpected, aggregator should have consumed it"
            )
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Process LLM context frames
        if isinstance(frame, LLMContextFrame):
            context = frame.context
            messages = context.messages if hasattr(context, "messages") else []

            logger.info(f"LLMContextFrame received with {len(messages)} messages")

            # Log message details for debugging
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content_preview = content[:80] if len(content) > 80 else content
                        logger.info(f"  Message {i} ({role}): {content_preview}")
                    else:
                        logger.info(f"  Message {i} ({role}): multimodal content")

            has_user_message = any(
                (isinstance(msg, dict) and msg.get("role") == "user")
                or (hasattr(msg, "role") and msg.role == "user")
                for msg in messages
            )

            if not has_user_message:
                logger.info("No user message, skipping LLM call")
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                return

            logger.info(f"Processing LLM request: has_user={has_user_message}")

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
        # Snapshot generation ID so we can detect staleness after ainvoke
        my_generation = self._generation_id

        try:
            # Convert context to LangChain messages
            messages = self._convert_context_to_langchain(context)
            user_message = self._extract_user_message(messages)

            if not user_message:
                logger.warning("No message to process, skipping")
                return

            logger.info(f"Processing message: {user_message[:100]}...")

            # Feed user message to Soul System
            if self._soul:
                self._soul.on_user_message(user_message)

            # Always attach the latest camera image for each request
            # This ensures vision requests always have the current view
            image_data = None
            if self._last_image:
                image_data = self._encode_image_to_base64(self._last_image)
                logger.info(
                    f"Attached image to request (data length: {len(image_data) if image_data else 0})"
                )
            else:
                logger.warning("No image available from camera")

            # Build input state
            input_state = {
                "messages": [HumanMessage(content=user_message)],
                "current_image": image_data,
            }

            # Set PROCESSING mode — soul shows "thinking" pose while LLM infers
            if (
                MovementMode
                and self._reachy_service
                and hasattr(self._reachy_service, "motion_manager")
            ):
                mm = self._reachy_service.motion_manager
                if mm:
                    mm.set_mode(MovementMode.PROCESSING)

            # ── Filler speech (Phase 6) ──────────────────────────────────
            # Push a quick filler phrase to TTS so the user hears something
            # while the main LLM is processing. Only for non-trivial requests.
            # Disabled by default — enable via FILLER_SPEECH_ENABLED=true env var.
            # Note: This sends a separate LLM response frame pair, so TTS will
            # speak the filler, then the main response follows shortly after.
            if (
                os.getenv("FILLER_SPEECH_ENABLED", "false").lower() == "true"
                and len(user_message) > 10
            ):
                import random

                fillers = [
                    "Let me check on that.",
                    "One moment.",
                    "On it.",
                    "Let me think about that.",
                    "Hmm, give me a sec.",
                ]
                filler = random.choice(fillers)
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text=filler))
                await self.push_frame(LLMFullResponseEndFrame())
                logger.debug(f"Filler speech: '{filler}'")

            # Invoke the graph as a tracked task so interruption can cancel it
            config = {"configurable": {"thread_id": self._thread_id}}

            logger.info("Invoking LangGraph agent...")
            invoke_coro = self._graph.ainvoke(input_state, config)
            self._inflight_task = asyncio.ensure_future(invoke_coro)
            try:
                result = await self._inflight_task
            finally:
                self._inflight_task = None

            # ── Staleness check ──────────────────────────────────────────
            # If the user interrupted while the LLM was running, the
            # generation ID will have been bumped. Discard the stale result.
            if self._generation_id != my_generation:
                logger.info(
                    f"Discarding stale LLM result (generation {my_generation} "
                    f"vs current {self._generation_id})"
                )
                return

            # Extract the response
            response_text = ""
            pending_commands = []

            if "messages" in result:
                # Iterate in REVERSE to get the LAST (most recent) AIMessage
                # With MemorySaver, all previous messages are kept in state
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        response_text = msg.content
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break

            # Get any pending Reachy commands
            if "pending_reachy_commands" in result:
                pending_commands = result["pending_reachy_commands"]

            # Also check tool_results for command tokens (backup extraction)
            if "tool_results" in result:
                for tool_result in result["tool_results"]:
                    tool_output = str(tool_result.get("result", ""))
                    # Extract any command tokens from tool output
                    cmd_matches = re.findall(r"\[CMD_[A-Z_]+\]", tool_output)
                    for cmd in cmd_matches:
                        if cmd not in response_text:
                            # Append command tokens to response so ReachyCommandProcessor can find them
                            response_text = f"{response_text} {cmd}"
                            logger.info(f"Appended command token to response: {cmd}")

            logger.info(f"Agent response: {response_text[:100]}...")
            logger.info(f"Full response length: {len(response_text)} chars")
            if pending_commands:
                logger.info(f"Pending Reachy commands: {pending_commands}")

            # Feed response to Soul System for emotion inference
            if self._soul:
                self._soul.on_bot_response(response_text)

            # Check if the response contains an image URL
            url_match = re.search(r"https?://[^\s]+", response_text)

            if "\n\nImage URL:" in response_text:
                # Split into spoken part and URL part
                parts = response_text.split("\n\nImage URL:", 1)
                spoken_text = parts[0].strip()
                url_text = parts[1].strip() if len(parts) > 1 else ""

                # 1) Send spoken portion through TTS normally
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text=spoken_text))
                await self.push_frame(LLMFullResponseEndFrame())
                logger.info(f"Spoken response pushed: {spoken_text[:80]}...")

                # 2) Send URL as text-only (skip TTS) so it appears in chat UI
                if url_text:
                    url_start = LLMFullResponseStartFrame()
                    url_start.skip_tts = True
                    url_frame = LLMTextFrame(text=f"\n🖼️ {url_text}")
                    url_frame.skip_tts = True
                    url_end = LLMFullResponseEndFrame()
                    url_end.skip_tts = True
                    await self.push_frame(url_start)
                    await self.push_frame(url_frame)
                    await self.push_frame(url_end)
                    logger.info(f"URL text-only frame pushed: {url_text[:80]}...")
            elif url_match:
                # Response has a URL but not in "Image URL:" format — split generically
                url = url_match.group(0)
                spoken_text = response_text.replace(url, "").strip()
                spoken_text = re.sub(r"\s+", " ", spoken_text)

                # Send spoken part
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text=spoken_text))
                await self.push_frame(LLMFullResponseEndFrame())

                # Send URL as text-only
                url_start = LLMFullResponseStartFrame()
                url_start.skip_tts = True
                url_frame = LLMTextFrame(text=f"\n🖼️ {url}")
                url_frame.skip_tts = True
                url_end = LLMFullResponseEndFrame()
                url_end.skip_tts = True
                await self.push_frame(url_start)
                await self.push_frame(url_frame)
                await self.push_frame(url_end)
                logger.info(f"URL text-only frame pushed (generic): {url[:80]}...")
            else:
                # Normal text response — no URL
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text=response_text))
                await self.push_frame(LLMFullResponseEndFrame())

            logger.info("Response frames pushed")

        except asyncio.CancelledError:
            logger.info("LangGraph invocation cancelled by interruption")
            # Don't push any frames — the user interrupted, pipeline handles cleanup
            return

        except Exception as e:
            logger.error(f"LangGraph processing error: {e}", exc_info=True)
            import random

            error_responses = [
                "Hmm, something went sideways on my end. Want to try that again?",
                "I got a bit tangled up processing that. Could you rephrase?",
                "My circuits got crossed on that one. What were you saying?",
                "I stumbled on that request. Mind giving it another go?",
            ]
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=random.choice(error_responses)))
            await self.push_frame(LLMFullResponseEndFrame())
