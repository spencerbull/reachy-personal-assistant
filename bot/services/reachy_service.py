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

        # Mapping: direction -> (rx, ry, rz, tx, ty, tz, body_yaw)
        # Head coords: x, y, z (meters), roll, pitch, yaw (degrees)
        # Note: create_head_pose args are (x, y, z, roll, pitch, yaw)
        # ReachyMini create_head_pose might be (roll, pitch, yaw)? 
        # Wait, line 7: from reachy_mini.utils import create_head_pose
        # Usage at 113: create_head_pose(*deltas, degrees=True)
        # Current DELTAS has 6 values.
        
        # Let's map direction to head_yaw AND body_yaw
        
        # Head Yaw (last element of 6-tuple currently)
        # Body Yaw (separate variable)
        
        target_head_yaw = 0
        target_head_pitch = 0
        target_body_yaw = 0
        
        # "Look" = Head dominant (Use neck)
        if direction == "left":
            target_head_yaw = 40
            target_body_yaw = 0
        elif direction == "right":
            target_head_yaw = -40
            target_body_yaw = 0
            
        # "Turn" = Body dominant (Use torso, keep head aligned or slightly helping)
        elif direction == "turn_left":
            target_body_yaw = 45 # Rotate body left
            target_head_yaw = target_body_yaw # Align head with body
        elif direction == "turn_right":
            target_body_yaw = -45
            target_head_yaw = target_body_yaw # Align head with body
            
        elif direction == "up":
            target_head_pitch = -30
        elif direction == "down":
            target_head_pitch = 30
            
        # Assuming deltas was (0,0,0, 0, pitch, yaw)
        deltas = (0, 0, 0, 0, target_head_pitch, target_head_yaw)
        
        try:
            # Get current pose first to preserve frame origin (XYZ)
            current_head_pose = self.robot.get_current_head_pose()
            current_x = current_head_pose[0, 3] # float
            current_y = current_head_pose[1, 3] # float
            current_z = current_head_pose[2, 3] # float
            
            # Use current XYZ but apply target rotations
            # Note: create_head_pose args: x, y, z, roll, pitch, yaw
            target_pose = create_head_pose(
                current_x, 
                current_y, 
                current_z, 
                0, # target roll is 0 
                target_head_pitch, 
                target_head_yaw, 
                degrees=True,
                mm=False
            )
            
            _, current_antennas = self.robot.get_current_joint_positions()

            # Determine duration based on move type
            # Body turns should be slower
            duration = 1.0
            if abs(target_body_yaw) > 10:
                duration = 3.0

            goto_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=target_body_yaw, 
                start_body_yaw=0, 
                duration=duration
            )
            self.motion_manager.queue_move(goto_move)
            self.motion_manager.set_moving_state(duration)
            logger.info(f"Reachy action {direction}: head_yaw={target_head_yaw}, body_yaw={target_body_yaw}, duration={duration}, xyz=({current_x:.2f},{current_y:.2f},{current_z:.2f})")
            return duration
        except Exception as e:
            logger.error(f"Look at failed: {e}")
            return 0.0

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
