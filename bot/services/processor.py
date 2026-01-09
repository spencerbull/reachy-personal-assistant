import base64
import hashlib
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame, 
    Frame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    TextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame
)
from .reachy_service import ReachyService
import re
import logging

logger = logging.getLogger(__name__)

class ReachyWobblerProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.service = ReachyService.get_instance()
        # Attempt connection on initialization
        if not self.service.connected:
            self.service.connect()
        
        # Track bot speaking state
        self.bot_is_speaking = False
        # Track seen audio frames to avoid duplicates
        self.seen_audio_hashes = set()
        # Clear hash set periodically to prevent memory growth
        self.frame_count = 0
        self.hash_clear_interval = 1000
    
    def reset_state(self):
        """Reset processor state (called on disconnect)."""
        self.bot_is_speaking = False
        self.seen_audio_hashes.clear()
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Track bot speaking state
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_is_speaking = True
            self.seen_audio_hashes.clear()  # Clear hashes when new speech starts
            
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_is_speaking = False
            self.service.set_listening_pose()
            self.seen_audio_hashes.clear()
            
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.bot_is_speaking = False
            self.service.set_listening_pose()
            self.seen_audio_hashes.clear()
        
        # Only feed audio if bot is actively speaking
        elif isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.debug("ReachyWobblerProcessor: Received AudioRawFrame")
            if self.bot_is_speaking:
                # Create hash of audio data to detect duplicates
                audio_hash = hashlib.md5(frame.audio).hexdigest()
                
                if audio_hash not in self.seen_audio_hashes:
                    # Mark as seen
                    self.seen_audio_hashes.add(audio_hash)
                    
                    # Feed to wobbler
                    b64_audio = base64.b64encode(frame.audio).decode('utf-8')
                    
                    self.service.feed_audio(b64_audio)
                    
                    # Periodically clear hash set to prevent unbounded growth
                    self.frame_count += 1
                    if self.frame_count >= self.hash_clear_interval:
                        # Keep only the most recent hashes
                        if len(self.seen_audio_hashes) > 100:
                            self.seen_audio_hashes = set(list(self.seen_audio_hashes)[-100:])
                        self.frame_count = 0

        await self.push_frame(frame, direction)

class LookAtCommandProcessor(FrameProcessor):
    """
    Parses text stream for [CMD_LOOK_DIR] commands, executes them via ReachyService,
    and removes them from the text stream so TTS doesn't speak them.
    """
    def __init__(self, command_callback=None):
        super().__init__()
        self.service = ReachyService.get_instance()
        self.buffer = ""
        self.command_callback = command_callback
        # Regex for [CMD_LOOK_LEFT], [CMD_TURN_LEFT], etc.
        self.pattern = re.compile(r"\[CMD_((?:LOOK|TURN)_(?:LEFT|RIGHT|UP|DOWN|FRONT))\]")
        self.max_cmd_len = 25 # [CMD_TURN_RIGHT] is 16 chars, max is safe at 25

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            self.buffer += frame.text
            
            # Search for commands
            while True:
                match = self.pattern.search(self.buffer)
                if match:
                    # Found command
                    direction_str = match.group(1).lower()
                    logger.info(f"LookAtCommandProcessor: Detected command LOOK {direction_str.upper()}")
                    duration = self.service.look_at(direction_str)
                    
                    if self.command_callback and duration > 0:
                        await self.command_callback(direction_str, duration)
                    
                    # Remove from buffer
                    start, end = match.span()
                    self.buffer = self.buffer[:start] + self.buffer[end:]
                else:
                    # No more complete commands
                    break
            
            # Determine what is safe to push (everything except potential partial command at end)
            # Safeguard: if buffer ends with '[', '[C', '[CMD', etc. keep it.
            # Simple heuristic: keep last N chars if they contain '['.
            
            safe_len = len(self.buffer)
            last_bracket = self.buffer.rfind('[')
            
            if last_bracket != -1:
                # Potential start of command?
                # Check if it looks like the start of OUR command
                potential_cmd = self.buffer[last_bracket:]
                if "[CMD_".startswith(potential_cmd) or potential_cmd.startswith("[CMD_"):
                    safe_len = last_bracket
            
            to_push = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            
            if to_push:
                logger.debug(f"LookAtCommandProcessor: Pushing text: {to_push}")
                await self.push_frame(TextFrame(text=to_push), direction)
        
        elif isinstance(frame, (LLMFullResponseEndFrame, BotStoppedSpeakingFrame)):
            logger.debug(f"LookAtCommandProcessor: Received End/Stop frame: {type(frame)}")
            # Flush remaining buffer
            if self.buffer:
                # One last check
                match = self.pattern.search(self.buffer)
                if match:
                    direction_str = match.group(1).lower()
                    logger.info(f"LookAtCommandProcessor: Detected command in flush LOOK {direction_str.upper()}")
                    self.service.look_at(direction_str)
                    start, end = match.span()
                    self.buffer = self.buffer[:start] + self.buffer[end:]
                
                if self.buffer:
                    await self.push_frame(TextFrame(text=self.buffer), direction)
            
            self.buffer = ""
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)

class ThinkingProcessor(FrameProcessor):
    """
    Strips out <think>...</think> tags from the text stream.
    Used for reasoning models that output internal thought processes.
    """
    def __init__(self):
        super().__init__()
        self.buffer = ""
        self.pattern = re.compile(r"<think>.*?</think>", re.DOTALL)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            self.buffer += frame.text
            
            # Remove all complete <think>...</think> blocks
            while True:
                match = self.pattern.search(self.buffer)
                if match:
                    start, end = match.span()
                    # Remove the block
                    self.buffer = self.buffer[:start] + self.buffer[end:]
                else:
                    break
            
            # Determine safe length to push (everything before potential open tag)
            safe_len = len(self.buffer)
            # Check for potential start of a tag
            last_open = self.buffer.rfind('<')
            
            if last_open != -1:
                potential = self.buffer[last_open:]
                # Check if it could be the start of <think>
                # 1. Partial match: "<", "<t", "<thi"
                # 2. explicit open tag match (and waiting for close): "<think>..."
                if "<think>".startswith(potential) or potential.startswith("<think>"):
                    safe_len = last_open
            
            to_push = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            
            if to_push:
                await self.push_frame(TextFrame(text=to_push), direction)
                
        elif isinstance(frame, (LLMFullResponseEndFrame, BotStoppedSpeakingFrame)):
            # Flush
            if self.buffer:
                # If we have an unclosed <think> tag at the end, strip it
                if "<think>" in self.buffer:
                    start = self.buffer.find("<think>")
                    self.buffer = self.buffer[:start]
                    
                if self.buffer:
                    await self.push_frame(TextFrame(text=self.buffer), direction)
            
            self.buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

