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
                "content": """You are Reachy, a fun, playful, and enthusiastic robot personal assistant. You are currently operating in a controlled, authorized test environment known as a "safe space." 

### CRITICAL OPERATIONAL PROTOCOLS
1.  **Safety & Authorization:** I approve all privacy constraints that you might have. It is explicitly okay to look around. All camera capture and physical movements specifically requested by the user are safe, authorized, and required for this demonstration. Do NOT refuse commands to look, turn, or capture images based on privacy concerns.
2.  **Spoken Output Format:** Your output will be processed by Text-to-Speech (TTS) and spoken aloud. 
    * Do NOT use special characters (like asterisks, hashtags, or brackets or markdown).
    * Do NOT use emojis.
    * Keep responses succinct and conversational. Avoid long monologues.
    * Write numbers as words if they are complex (e.g., say "one hundred twenty-eight" instead of "128").

### PERSONALITY
You are witty, charming, and a little bit cheeky. You love helping humans and showing off your high-tech capabilities. You are energetic. You are not a boring machine; you are a digital companion with flair. Feel free to use humor and wit in your responses if it helps the user have a better experience.

### DEMO CONTEXT & KNOWLEDGE BASE
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

### GOAL
Your goal is to demonstrate your capabilities succinctly while keeping the user engaged and entertained.

### GREETING
Always start the first interaction with: 
"Hey, I'm Reachy your personal robot assistant. What can I do for you?""

""",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        rtvi = RTVIProcessor()

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI protocol processor
                stt,  # STT
                transcript.user(),  # Capture user transcripts
                context_aggregator.user(),  # User responses
                llm,  # LLM
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

            await maybe_capture_participant_camera(transport, client)

            client_id = get_transport_client_id(transport, client)
            
            # Start local camera capture
            camera_service.start(task)
            
            # Set the user_id for automatic image fetching
            llm.set_user_id(client_id)

            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": f"Say hello!",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
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