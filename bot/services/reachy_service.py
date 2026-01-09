import threading
import logging
from reachy_mini import ReachyMini
from .moves import MovementManager
from .wobbler import HeadWobbler
from .dance_emotion_moves import GotoQueueMove
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

class ReachyService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, host='localhost'):
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.host = host
        self.connected = False

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = ReachyService()
        return cls._instance

    def connect(self):
        # If already connected, return
        if self.connected:
            logger.debug("Reachy already connected")
            return
            
        # If previously disconnected, clean up any leftover state
        if self.robot or self.motion_manager or self.wobbler:
            logger.info("Cleaning up previous Reachy connection...")
            self.disconnect()
            
        try:
            import os
            import time
            
            # Verify DISPLAY is set
            display = os.getenv('DISPLAY')
            logger.info(f"DISPLAY environment variable: {display}")
            
            # Give Xvfb extra time to stabilize
            logger.info("Waiting for display to be ready...")
            time.sleep(3)
            
            logger.info(f"Starting Reachy Mini daemon (expecting sim mode)...")
            
            self.robot = ReachyMini(
                use_sim=True,
                spawn_daemon=False,
                localhost_only=False,     
                timeout=15.0,          # Increased timeout
                log_level='DEBUG'      
            )
            logger.info("Successfully connected to Reachy Mini daemon")
            
            # 1. Initialize Motor Cortex (Background Thread)
            self.motion_manager = MovementManager(self.robot)
            self.motion_manager.start() 
            
            # 2. Initialize Auditory Cortex (Links Audio -> Motion)
            self.wobbler = HeadWobbler(self.motion_manager.set_speech_offsets)
            self.wobbler.start()
            
            self.connected = True
            logger.info("Reachy Service Started: Breathing & Sway active.")
        except Exception as e:
            import traceback
            logger.warning(f"Reachy Mini daemon not available: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Pipeline will continue without Reachy robot control.")
            logger.warning("To enable Reachy: start daemon with 'mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only'")
            
            # Clean up partial robot object to avoid destructor errors
            self.robot = None
            # Don't raise - allow pipeline to run without Reachy

    def feed_audio(self, audio_chunk_base64):
        """Feeds audio from TTS to the wobble engine."""
        if self.wobbler:
            logger.info("Feeding audio to Reachy")
            self.wobbler.feed(audio_chunk_base64)
    
    def set_listening_pose(self):
        """Sets robot back to listening/idle pose."""
        if self.motion_manager:
            self.motion_manager.set_listening(True)
            logger.info("Reachy set to listening pose")

    def look_at(self, direction: str):
        """Maps semantic direction to robot pose."""
        if not self.connected or not self.motion_manager or not self.robot:
            logger.debug(f"Reachy not connected - ignoring look_at({direction})")
            return

        # Mapping adapted from Reachy tools
        DELTAS = {
            "left": (0, 0, 0, 0, 0, 40),
            "right": (0, 0, 0, 0, 0, -40),
            "up": (0, 0, 0, 0, -30, 0),
            "down": (0, 0, 0, 0, 30, 0),
            "front": (0, 0, 0, 0, 0, 0),
        }
        deltas = DELTAS.get(direction, DELTAS["front"])
        
        try:
            target_pose = create_head_pose(*deltas, degrees=True)
            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()

            goto_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=0, 
                start_body_yaw=0,
                duration=1.0
            )
            self.motion_manager.queue_move(goto_move)
            self.motion_manager.set_moving_state(1.0)
            logger.info(f"Reachy looking {direction}")
        except Exception as e:
            logger.error(f"Look at failed: {e}")

    def disconnect(self):
        """Disconnect and cleanup Reachy resources."""
        if not self.connected:
            return
            
        logger.info("Disconnecting Reachy service...")
        
        # Stop background threads
        if self.motion_manager:
            self.motion_manager.stop()
        if self.wobbler:
            self.wobbler.stop()
        
        # Disconnect robot
        if self.robot:
            try:
                # The robot client should disconnect gracefully
                if hasattr(self.robot, 'client') and self.robot.client:
                    self.robot.client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")
        
        # Reset state
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.connected = False
        
        logger.info("Reachy service disconnected")
    
    def stop(self):
        """Alias for disconnect for backwards compatibility."""
        self.disconnect()