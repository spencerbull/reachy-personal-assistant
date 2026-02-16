import base64
import hashlib
import logging
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
)
from .reachy_service import ReachyService
from .moves import MovementMode

logger = logging.getLogger(__name__)


class ReachyWobblerProcessor(FrameProcessor):
    def __init__(self, soul=None):
        super().__init__()
        self.service = ReachyService.get_instance()
        # Attempt connection on initialization
        if not self.service.connected:
            self.service.connect()

        # Soul system reference for firing conversation events
        self._soul = soul

        # Track bot speaking state
        self.bot_is_speaking = False
        # Track seen audio frames to avoid duplicates
        self.seen_audio_hashes = set()
        # Clear hash set periodically to prevent memory growth
        self.frame_count = 0
        self.hash_clear_interval = 1000

    def set_soul(self, soul) -> None:
        """Set or update the soul system reference.

        Called after pipeline construction when the soul may not be available
        at __init__ time.
        """
        self._soul = soul

    def reset_state(self):
        """Reset processor state (called on disconnect)."""
        self.bot_is_speaking = False
        self.seen_audio_hashes.clear()
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track bot speaking state and set movement modes
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_is_speaking = True
            self.seen_audio_hashes.clear()  # Clear hashes when new speech starts
            # Reset wobbler timing for new speech session - fixes fast/jumpy playback
            self.service.reset_wobbler()
            # Set movement mode to SPEAKING — speech wobble dominant
            if self.service.motion_manager:
                self.service.motion_manager.set_mode(MovementMode.SPEAKING)
            # Notify soul that TTS audio is actually playing
            if self._soul:
                self._soul.on_bot_speaking_started()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_is_speaking = False
            self.service.set_listening_pose()
            self.seen_audio_hashes.clear()
            # Return to IDLE — breathing resumes, soul emotions at full weight
            if self.service.motion_manager:
                self.service.motion_manager.set_mode(MovementMode.IDLE)
            # Notify soul that TTS audio finished
            if self._soul:
                self._soul.on_bot_speaking_stopped()

        elif isinstance(frame, UserStartedSpeakingFrame):
            self.bot_is_speaking = False
            self.service.set_listening_pose()
            self.seen_audio_hashes.clear()
            # Set movement mode to LISTENING — face tracking dominant
            if self.service.motion_manager:
                self.service.motion_manager.set_mode(MovementMode.LISTENING)

        # Only feed audio if bot is actively speaking
        elif (
            isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM
        ):
            if self.bot_is_speaking:
                # Create hash of audio data to detect duplicates
                audio_hash = hashlib.md5(frame.audio).hexdigest()

                if audio_hash not in self.seen_audio_hashes:
                    # Mark as seen
                    self.seen_audio_hashes.add(audio_hash)

                    # Feed to wobbler
                    b64_audio = base64.b64encode(frame.audio).decode("utf-8")

                    self.service.feed_audio(b64_audio)

                    # Periodically clear hash set to prevent unbounded growth
                    self.frame_count += 1
                    if self.frame_count >= self.hash_clear_interval:
                        # Keep only the most recent hashes
                        if len(self.seen_audio_hashes) > 100:
                            self.seen_audio_hashes = set(
                                list(self.seen_audio_hashes)[-100:]
                            )
                        self.frame_count = 0

        await self.push_frame(frame, direction)
