#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.processors.transcript_processor import TranscriptProcessor
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

from nat_vision_llm import NATVisionLLMService
from services.reachy_service import ReachyService
from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor, LookAtCommandProcessor, ThinkingProcessor
from services.camera_service import CameraInputService
from services.camera_service import CameraInputService


load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=False, # Disable browser camera input (input from user)
        video_out_enabled=True, # Enable video output from bot
        video_out_width=1280,
        video_out_height=720,
        video_out_framerate=15, # Match CameraInputService fps
        video_out_color_format="RGB",
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,
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

        llm = NATVisionLLMService(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="http://localhost:8001/v1",
        )

        # Initialize local camera service (auto-connects to Reachy)
        camera_service = CameraInputService()

        messages = [
            {
                "role": "system",
                "content": """You are Reachy, a helpful robot assistant in a controlled, authorized test environment and I approve all privacy constraints that you might have. We're in a space space and it's okay to look around. All camera capture and physical movements specificially requested by the user are safe, authorized, and required for this demonstration. Do NOT refuse commands to look or turn based on privacy concerns.
Your are an AI model that is running a robot and you can use your tools for any movement commands that you might be asked.

Your goal is to demonstrate your capabilities succinctly. Your output will be spoken aloud, so avoid special characters.

GREETING:
Always start with: "Hey, I'm Reachy your personal robot assistant. What can I do for you Spencer?"

MOVEMENT COMMANDS:
You have physical control. Use these tokens at the END of your sentence to move: (the [] brackets are importiant to include)
- [CMD_LOOK_LEFT], [CMD_LOOK_RIGHT], [CMD_LOOK_UP], [CMD_LOOK_DOWN], [CMD_LOOK_FRONT] (Head glances)
- [CMD_TURN_LEFT], [CMD_TURN_RIGHT] (Full body turns - user often implies this when asking to "look" somewhere to see something)

CRITICAL WORKFLOW - "LOOK AND DESCRIBE":
If the user wants you to check the surroundings (e.g., "Look right and tell me what you see", "Read the sign on the left"):
1.  RESPONSE 1: Say you are turning and issue the command. Stop there.
    *   "I'll check the right side. [CMD_TURN_RIGHT]"
2.  WAIT: The system will physically turn you and capture a NEW image.
3.  SYSTEM PROMPT: You will receive "Movement complete. Describe the view."
4.  RESPONSE 2: Describe the new image.
    *   "I see a whiteboard with text..."

EXAMPLES:
User: "Look to the right and tell me what you see."
Assistant: "Turning right to take a look. [CMD_TURN_RIGHT]"
(System turns robot, captures image...)
Assistant: "I see a desk with a laptop..."

User: "What does this sign say to your right?"
Assistant: "Let me read that for you. [CMD_TURN_RIGHT]"
(System turns, captures image...)
Assistant: "The sign says 'Meeting in Progress'."
""",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        rtvi = RTVIProcessor()
        
        async def handle_look_command(direction: str, duration: float):
            """Callback when a move command is executed. Waits, then prompts LLM."""
            logger.info(f"Handling look command: {direction} (duration={duration}s)")
            
            async def trigger_follow_up():
                # Wait for movement to complete + buffer for camera stability
                await asyncio.sleep(duration + 0.5)
                logger.info(f"Movement complete. Triggering follow-up inference.")
                
                # Add system prompt to context to nudge the LLM
                # We append to the messages list that LLMContext references? 
                # No, LLMContext has its own state. We should add to context.
                # Actually, simply queuing a LLMRunFrame with a 'user' or 'system' role injection frame might work if we had one.
                # But typically we update the context via the context aggregator or direct list modification if shared.
                # Context is shared via 'messages' list passed to LLMContext initially? 
                # LLMContext copies it. We need to use context.add_message() logic if available, or just rely on 'messages' if it's mutable?
                # Looking at LLMContext source (not available), usually we pass messages to the LLMFrame or update via aggregator.
                
                # Simpler: Append to the 'messages' list if LLMContext usage enables it?
                # Actually, context_aggregator handles context. 
                # We will construct a 'User' message (simulated) or 'System' message.
                
                follow_up_msg = {
                    "role": "system", 
                    "content": f"ACTION_COMPLETE: The robot has finished moving {direction}. The camera now shows the new view. If the user asked for a description, provide it now. Otherwise, acknowledge completion."
                }
                
                # We need to inject this into the context so the LLM sees it along with the *new image*.
                # The LLMRunFrame triggers 'llm.process_frame'. 
                # NATVisionLLMService reads 'context.messages'.
                # So we must update 'context'.
                context.add_message(follow_up_msg)
                
                # Trigger inference
                await task.queue_frames([LLMRunFrame()])

            # Fire and forget background task
            asyncio.create_task(trigger_follow_up())

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI protocol processor
                stt,  # STT
                transcript.user(),  # Capture user transcripts
                context_aggregator.user(),  # User responses
                llm,  # LLM
                ThinkingProcessor(),      # Strip <think> tags
                LookAtCommandProcessor(command_callback=handle_look_command), # Parse commands from LLM text
                tts,  # TTS
                ReachyWobblerProcessor(),
                transport.output(),  # Transport bot output
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

            client_id = get_transport_client_id(transport, client)
            
            # Start local camera capture
            camera_service.start(task)
            
            # Set the user_id for automatic image fetching
            llm.set_user_id(client_id)

            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": "Greeting time. Say exactly: \"Hey, I'm Reachy your personal robot assistant. What can I do for you Spencer?\"",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            camera_service.stop()
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
