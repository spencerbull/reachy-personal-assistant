#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys
import re

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, Frame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import LLMTextFrame
from pipecat.runner.types import RunnerArguments


class URLExtractorProcessor(FrameProcessor):
    """
    Safety net: strips any URLs that might still be in LLMTextFrame text
    headed for TTS. The main URL handling (skip_tts frames) is done
    upstream in LangGraphLLMService, but this catches edge cases.
    """

    def __init__(self, transport_output=None):
        super().__init__()
        self._transport = transport_output  # kept for backwards compat, unused

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and frame.text:
            # If skip_tts is set, pass through unchanged (text-only message)
            if frame.skip_tts:
                await self.push_frame(frame, direction)
                return

            # Safety: strip any URLs that slipped through to TTS path
            url_match = re.search(r"https?://[^\s]+", frame.text)
            if url_match:
                filtered = re.sub(r"https?://[^\s]+", "", frame.text)
                filtered = re.sub(r"\s+", " ", filtered).strip()
                if filtered:
                    logger.info(
                        f"URLExtractor: Stripped leftover URL for TTS: {filtered[:50]}..."
                    )
                    frame = LLMTextFrame(text=filtered)
                else:
                    # Nothing left after stripping URL — drop the frame
                    return

        await self.push_frame(frame, direction)


from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
import aiohttp
import asyncio

# Add parent directory to path for agent imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Choose LLM backend: "langgraph" or "nat"
LLM_BACKEND = os.getenv("LLM_BACKEND", "langgraph")

if LLM_BACKEND == "langgraph":
    from langgraph_llm import LangGraphLLMService
else:
    from nat_vision_llm import NATVisionLLMService

from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor
from services.camera_service import CameraFrameProcessor


load_dotenv(override=True)


class ReachyCommandProcessor(FrameProcessor):
    """
    Processes command tokens in LLM output and executes Reachy robot actions.

    Command tokens like [CMD_LOOK_LEFT], [CMD_FACE_TRACK_ON], etc. are parsed
    from the text and executed on the robot, then removed from the output
    before it goes to TTS.
    """

    # Command pattern to match [CMD_*] tokens
    CMD_PATTERN = re.compile(r"\[CMD_([A-Z_]+)\]")

    def __init__(self):
        super().__init__()
        self.reachy_service = ReachyService.get_instance()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text

            # Find and execute commands
            commands = self.CMD_PATTERN.findall(text)

            for cmd in commands:
                await self._execute_command(cmd)

            # Remove command tokens from text before TTS
            clean_text = self.CMD_PATTERN.sub("", text).strip()

            # Handle multiple spaces that might result from removal
            clean_text = re.sub(r"\s+", " ", clean_text)

            if clean_text:
                await self.push_frame(TextFrame(text=clean_text), direction)
            return

        await self.push_frame(frame, direction)

    async def _execute_command(self, command: str):
        """Execute a Reachy robot command."""
        command = command.lower()
        logger.info(f"Executing Reachy command: {command}")

        try:
            if command == "look_left":
                self.reachy_service.look_at("left")
            elif command == "look_right":
                self.reachy_service.look_at("right")
            elif command == "look_up":
                self.reachy_service.look_at("up")
            elif command == "look_down":
                self.reachy_service.look_at("down")
            elif command == "look_front":
                self.reachy_service.look_at("front")
            elif command == "turn_left":
                # Body turn - larger movement
                logger.info("Body turn left requested")
                # TODO: Implement body turn via ReachyService
            elif command == "turn_right":
                logger.info("Body turn right requested")
                # TODO: Implement body turn via ReachyService
            elif command == "face_track_on":
                logger.info("Face tracking enabled")
                cam_processor = CameraFrameProcessor.get_instance()
                if cam_processor:
                    cam_processor.enable_face_tracking(True)
            elif command == "face_track_off":
                logger.info("Face tracking disabled")
                cam_processor = CameraFrameProcessor.get_instance()
                if cam_processor:
                    cam_processor.enable_face_tracking(False)
            elif command.startswith("emotion_"):
                emotion = command.replace("emotion_", "")
                logger.info(f"Emotion expression: {emotion}")
                # Trigger emotion expression via motion manager
                self._trigger_emotion(emotion)
            elif command.startswith("dance_"):
                dance = command.replace("dance_", "")
                logger.info(f"Dance requested: {dance}")
                self._trigger_dance(dance)
            elif command == "nod_yes":
                logger.info("Nodding yes")
                # TODO: Implement nod animation
            elif command == "nod_no":
                logger.info("Shaking no")
                # TODO: Implement shake animation
            elif command.startswith("scan_"):
                scan_type = command.replace("scan_", "")
                logger.info(f"Room scan: {scan_type}")
                # TODO: Implement room scanning
            else:
                logger.warning(f"Unknown command: {command}")
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")

    def _trigger_emotion(self, emotion: str):
        """Trigger an emotion expression."""
        if not self.reachy_service.connected:
            return

        # Import dance/emotion moves
        try:
            from services.dance_emotion_moves import EmotionQueueMove
            from reachy_mini.motion.recorded_move import RecordedMoves

            if self.reachy_service.motion_manager:
                # Try to load recorded emotion move
                try:
                    recorded_moves = RecordedMoves()
                    emotion_move = EmotionQueueMove(emotion, recorded_moves)
                    self.reachy_service.motion_manager.queue_move(emotion_move)
                    logger.info(f"Queued emotion: {emotion}")
                except Exception as e:
                    logger.debug(f"No recorded move for {emotion}: {e}")
        except ImportError:
            logger.debug("Emotion moves not available")

    def _trigger_dance(self, dance_name: str):
        """Trigger a dance move."""
        if not self.reachy_service.connected:
            return

        try:
            from services.dance_emotion_moves import DanceQueueMove

            if self.reachy_service.motion_manager:
                dance_move = DanceQueueMove(dance_name)
                self.reachy_service.motion_manager.queue_move(dance_move)
                logger.info(f"Queued dance: {dance_name}")
        except Exception as e:
            logger.debug(f"Dance not available: {e}")


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=False,  # Disable browser camera input
        video_out_enabled=True,  # Enable video output from bot
        video_out_is_live=True,  # CRITICAL: Enable live video streaming
        video_out_width=1280,
        video_out_height=720,
        video_out_framerate=15,  # Match CameraInputService fps
        video_out_color_format="RGB",
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=float(os.getenv("VAD_STOP_SECS", "0.7")),
            )
        ),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,  # CRITICAL: Enable live video streaming
        video_out_width=1280,
        video_out_height=720,
        video_out_framerate=15,
        video_out_color_format="RGB",
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=float(os.getenv("VAD_STOP_SECS", "0.7")),
            )
        ),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Track LLM service for cleanup
    llm = None

    async with aiohttp.ClientSession() as session:
        stt = ElevenLabsSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            aiohttp_session=session,
        )

        tts = ElevenLabsHttpTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id="TtRFBnwQdH1k01vR0hMz",
            aiohttp_session=session,
        )

        # Create LLM service based on backend choice
        if LLM_BACKEND == "langgraph":
            logger.info("Using LangGraph LLM backend")
            llm = LangGraphLLMService()
        else:
            logger.info("Using NAT LLM backend")
            llm = NATVisionLLMService(
                api_key=os.getenv("NVIDIA_API_KEY"),
                base_url="http://localhost:8001/v1",
            )

        # Initialize camera frame processor (injects video frames into pipeline)
        # NOTE: This starts camera capture immediately in __init__ to avoid race conditions
        camera_processor = CameraFrameProcessor(
            fps=15, target_width=1280, target_height=720
        )

        # Initialize Reachy command processor for handling robot commands in LLM output
        command_processor = ReachyCommandProcessor()

        # System message for context (used by NAT backend; LangGraph uses its own config)
        messages = [
            {
                "role": "system",
                "content": "You are Reachy, a friendly robot assistant. Keep responses short and conversational. Your text goes directly to text-to-speech - speak naturally, no markdown or emojis.",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        rtvi = RTVIProcessor()

        # Store transport processors for direct access
        transport_input = transport.input()
        transport_output = transport.output()

        # URL extractor sends full text (with URL) to transport, strips URL for TTS
        url_extractor = URLExtractorProcessor(transport_output)

        # Wobbler processor — feeds TTS audio to head wobble and fires soul events
        wobbler_processor = ReachyWobblerProcessor()

        pipeline = Pipeline(
            [
                transport_input,  # Transport user input
                camera_processor,  # Inject camera frames into pipeline
                rtvi,  # RTVI protocol processor
                stt,  # STT
                transcript.user(),  # Capture user transcripts
                context_aggregator.user(),  # User responses
                llm,  # LLM (LangGraph or NAT)
                command_processor,  # Process [CMD_*] tokens and execute robot commands
                url_extractor,  # Extract URL -> send to chat, strip for TTS
                tts,  # TTS (speaks text without URLs)
                wobbler_processor,
                transport_output,  # Transport bot output
                transcript.assistant(),  # Capture assistant transcripts
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transcript.event_handler("on_transcript_update")
        async def handle_transcript_update(processor, frame):
            """Handle transcript updates and log them"""
            for message in frame.messages:
                logger.info(f"Transcript [{message.role}]: {message.content}")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")

            await maybe_capture_participant_camera(transport, client)

            client_id = get_transport_client_id(transport, client)

            # Set references for camera processor - use the stored transport_output
            camera_processor.set_task(task)
            camera_processor.set_output_transport(transport_output)
            await camera_processor.start()

            # Set the user_id for automatic image fetching
            llm.set_user_id(client_id)

            # Set transport and RTVI processor for direct chat messaging (for links, images, etc.)
            if LLM_BACKEND == "langgraph":
                llm.set_transport(transport)
                llm.set_rtvi_processor(rtvi)

            # Initialize MCP tools (Gmail, etc.) if configured
            if LLM_BACKEND == "langgraph":
                logger.info("Initializing MCP tools...")
                await llm.initialize_mcp()

                # Start Soul System for continuous embodiment (emotion, movement, idle behaviors)
                logger.info("Starting Soul System...")
                reachy_service = ReachyService.get_instance()
                llm.set_reachy_service(reachy_service)
                await llm.start_soul()

                # Wire soul reference into processors so they can fire events
                soul_ref = llm._soul
                if soul_ref:
                    wobbler_processor.set_soul(soul_ref)
                    if camera_processor._worker:
                        camera_processor._worker.set_soul(soul_ref)
                    logger.info("Soul wired into wobbler and camera processors")

                    # Apply SoulConfig mode weights to MovementManager
                    if reachy_service.motion_manager and hasattr(soul_ref, "config"):
                        from services.moves import MovementMode

                        cfg = soul_ref.config
                        reachy_service.motion_manager.configure_mode_weights(
                            weights_table={
                                MovementMode.IDLE: cfg.mode_weights_idle,
                                MovementMode.LISTENING: cfg.mode_weights_listening,
                                MovementMode.PROCESSING: cfg.mode_weights_processing,
                                MovementMode.SPEAKING: cfg.mode_weights_speaking,
                                MovementMode.SCANNING: cfg.mode_weights_scanning,
                            },
                            blend_duration=cfg.mode_transition_blend_s,
                        )
                        logger.info(
                            f"Mode weights configured from SoulConfig "
                            f"(blend={cfg.mode_transition_blend_s}s)"
                        )

            # Let the LLM generate a natural greeting
            messages.append(
                {
                    "role": "user",
                    "content": "[New session started]",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await camera_processor.stop()

            # Stop Soul System
            if LLM_BACKEND == "langgraph":
                logger.info("Stopping Soul System...")
                await llm.stop_soul()

            # Close MCP connections if using LangGraph
            if (
                LLM_BACKEND == "langgraph"
                and hasattr(llm, "_mcp_loader")
                and llm._mcp_loader
            ):
                logger.info("Closing MCP connections...")
                await llm._mcp_loader.close()

            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        try:
            await runner.run(task)
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled, cleaning up...")
            raise
        finally:
            # Ensure cleanup happens even on Ctrl+C
            logger.info("Running cleanup handlers...")

            # Stop camera processor
            try:
                await camera_processor.stop()
            except Exception as e:
                logger.debug(f"Camera cleanup error: {e}")

            # Clean up LLM service (stops soul, closes MCP)
            if LLM_BACKEND == "langgraph" and hasattr(llm, "cleanup"):
                try:
                    await llm.cleanup()
                except asyncio.CancelledError:
                    logger.info("LLM cleanup cancelled")
                except Exception as e:
                    logger.warning(f"LLM cleanup error: {e}")


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
