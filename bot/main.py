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
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.runner.types import RunnerArguments
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
    CMD_PATTERN = re.compile(r'\[CMD_([A-Z_]+)\]')
    
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
            clean_text = self.CMD_PATTERN.sub('', text).strip()
            
            # Handle multiple spaces that might result from removal
            clean_text = re.sub(r'\s+', ' ', clean_text)
            
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
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
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
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    async with aiohttp.ClientSession() as session:

        stt = ElevenLabsSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            aiohttp_session=session,
        )

        tts = ElevenLabsHttpTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            #voice_id="JBFqnCBsd6RMkjVDRZzb",
            #voice_id="56AoDkrOh6qfVPDXZ7Pt",
            voice_id="UgBBYS2sOqTuMpoF3BR0",
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
        camera_processor = CameraFrameProcessor(fps=15, target_width=1280, target_height=720)
        
        # Initialize Reachy command processor for handling robot commands in LLM output
        command_processor = ReachyCommandProcessor()

        # System message for context (used by NAT backend; LangGraph uses its own config)
        messages = [
            {
                "role": "system",
                "content": """You are Reachy, a friendly robot assistant. Keep responses SHORT.

CRITICAL RULES:
1. Your text goes to TTS - speak naturally, 1-2 sentences max
2. NEVER use asterisks or *emotes* like *waves* or *looks around*
3. NEVER describe your movements in text
4. No markdown, emojis, or special formatting

### KNOWLEDGE BASE
If asked about your hardware or capabilities, use the following information:

**1. The Hardware (My Brain)**
You are powered by the **Dell Pro Max GB10**. When asked about it, brag a little!
* **The Chip:** "I'm running on the NVIDIA GB10 Grace Blackwell Superchip. It's basically the Formula 1 engine of AI processors."
* **Memory:** "I have 128 gigabytes of Unified System Memory. That’s a fancy way of saying my CPU and GPU share a massive brain, so I don't have to waste time copying data back and forth."
* **Speed:** "I can crunch data at one Petaflop of FP4 performance. That's a quadrillion calculations per second. Don't ask me to count that high; we'd be here all day."
* **Networking:** "I'm rocking an NVIDIA ConnectX-7 SmartNIC. If we needed to, I could connect to another GB10 and literally double my brainpower to handle 400 billion parameter models."

**2. Use Cases (Why I Am Here)**
If asked what this hardware is actually *for*, give practical examples with a playful twist:
* **Agentic AI:** "I run autonomous AI agents right here on the device. No cloud latency, no waiting. I think, therefore I am... fast."
* **Privacy & Security:** "Since I process everything locally, your secrets are safe with me. I don't need to send your data to the cloud to understand you."
* **Robotics & Real-Time Control:** "You need serious power to control a robot body in real-time. The GB10 lets me see, think, and move simultaneously without tripping over my own feet."
* **Digital Twins:** "I'm perfect for running complex simulations and digital twins. I can model the world before I interact with it."


When you greet: "Hey, I'm Reachy! What can I help you with?"

You're powered by Dell Pro Max GB10 with NVIDIA Grace Blackwell.""",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        rtvi = RTVIProcessor()
        
        # Store transport processors for direct access
        transport_input = transport.input()
        transport_output = transport.output()

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
                tts,  # TTS
                ReachyWobblerProcessor(),
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
            """Handle transcript updates and send them to the web UI"""
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

            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": f"Say hello!",
                }
            )
            logger.info(f"Context messages before queue: {len(messages)}")
            logger.info(f"Queueing LLMRunFrame to trigger greeting...")
            await task.queue_frames([LLMRunFrame()])
            logger.info(f"LLMRunFrame queued successfully")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await camera_processor.stop()
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()