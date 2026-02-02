import os
import threading
import logging
import numpy as np
from reachy_mini import ReachyMini
from .moves import MovementManager
from .wobbler import HeadWobbler
from .dance_emotion_moves import GotoQueueMove, AntennaWaveQueueMove
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)


def get_localhost_only() -> bool:
    """
    Determine ReachyMini localhost_only setting based on environment.

    - REACHY_DAEMON_HOST set (e.g., 'host.docker.internal') -> False (allow network)
    - Default -> True (localhost only)
    """
    daemon_host = os.getenv('REACHY_DAEMON_HOST')
    if daemon_host and daemon_host != 'localhost':
        logger.info(f"Docker mode detected (REACHY_DAEMON_HOST={daemon_host}), allowing network connections")
        return False
    return True


class ReachyService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, host: str = None):
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.host = host or os.getenv('REACHY_DAEMON_HOST', 'localhost')
        self.connected = False
        self._localhost_only = get_localhost_only()

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
            
            logger.info(f"Connecting to Reachy Mini daemon (localhost_only={self._localhost_only}, host={self.host})...")

            self.robot = ReachyMini(
                use_sim=True,
                spawn_daemon=False,
                localhost_only=self._localhost_only,
                timeout=15.0,
                log_level='DEBUG'
            )
            logger.info(f"Successfully connected to Reachy Mini daemon")
            
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

    def apply_soul_pose(self, pose_dict: dict):
        """
        Apply a pose from the Soul System as secondary offsets.

        The pose_dict comes from MovementBlender.to_reachy_command() with keys:
        - head_pitch, head_yaw, head_roll (in DEGREES)
        - head_z_offset (in meters)
        - left_antenna, right_antenna (in radians)
        - body_yaw (in degrees)

        Soul poses are applied as secondary offsets that blend smoothly
        with the primary movement system (breathing, dances, etc.) and
        other secondary offsets (speech sway, face tracking).

        Args:
            pose_dict: Dictionary with movement parameters
        """
        if not self.connected or not self.motion_manager:
            return

        try:
            # Extract pose values and convert to offset format
            # Note: head angles come in degrees, need to convert to radians
            head_pitch_rad = np.deg2rad(pose_dict.get('head_pitch', 0.0))
            head_yaw_rad = np.deg2rad(pose_dict.get('head_yaw', 0.0))
            head_roll_rad = np.deg2rad(pose_dict.get('head_roll', 0.0))
            head_z_offset = pose_dict.get('head_z_offset', 0.0)  # already in meters

            # Antennas are already in radians
            left_antenna = pose_dict.get('left_antenna', 0.0)
            right_antenna = pose_dict.get('right_antenna', 0.0)

            # Apply as secondary offsets
            # Format: (x, y, z, roll, pitch, yaw) in meters and radians
            head_offsets = (
                0.0,  # x
                0.0,  # y
                head_z_offset,  # z
                head_roll_rad,  # roll
                head_pitch_rad,  # pitch
                head_yaw_rad,  # yaw
            )
            antenna_offsets = (left_antenna, right_antenna)

            self.motion_manager.set_soul_offsets(head_offsets, antenna_offsets)

        except Exception as e:
            logger.debug(f"Soul pose application failed: {e}")

    def antenna_wave(self):
        """Perform an antenna wave greeting gesture."""
        if not self.connected or not self.motion_manager:
            return

        try:
            _, current_antennas = self.robot.get_current_joint_positions()
            wave_move = AntennaWaveQueueMove(
                start_antennas=(current_antennas[0], current_antennas[1])
            )
            self.motion_manager.queue_move(wave_move)
            logger.info("Antenna wave queued")
        except Exception as e:
            logger.debug(f"Antenna wave failed: {e}")

    def set_face_tracking(self, enabled: bool):
        """Enable or disable face tracking mode."""
        if not self.connected:
            return

        # Face tracking is handled by the camera processor
        # This is a hook for the soul system to control it
        try:
            from services.camera_service import CameraFrameProcessor
            cam_processor = CameraFrameProcessor.get_instance()
            if cam_processor:
                cam_processor.enable_face_tracking(enabled)
                logger.info(f"Face tracking {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logger.debug(f"Face tracking control failed: {e}")

    def reset_wobbler(self):
        """Reset the wobbler timing for new speech session."""
        if self.wobbler:
            self.wobbler.reset()
            logger.debug("Wobbler timing reset")

    def look_around(self, yaw_rad: float, pitch_rad: float, duration: float = 1.0):
        """
        Make Reachy look in a specific direction (for room scanning, etc.).

        Args:
            yaw_rad: Target yaw angle in radians (+ left, - right)
            pitch_rad: Target pitch angle in radians (- up, + down)
            duration: Duration of the movement
        """
        if not self.connected or not self.motion_manager or not self.robot:
            return

        try:
            target_pose = create_head_pose(
                x=0.0,
                y=0.0,
                z=0.0,
                roll=0.0,
                pitch=np.rad2deg(pitch_rad),
                yaw=np.rad2deg(yaw_rad),
                degrees=True,
                mm=False
            )

            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()

            goto_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0.0, 0.0),  # Neutral antennas during scan
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=0,
                start_body_yaw=0,
                duration=duration
            )
            self.motion_manager.queue_move(goto_move)
            self.motion_manager.set_moving_state(duration)
            logger.debug(f"Look around: yaw={np.rad2deg(yaw_rad):.1f}°, pitch={np.rad2deg(pitch_rad):.1f}°")

        except Exception as e:
            logger.debug(f"Look around failed: {e}")

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