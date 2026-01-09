import time
import logging
import threading
import asyncio
import numpy as np
from typing import Any, List, Tuple
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from pipecat.frames.frames import UserImageRawFrame, OutputImageRawFrame, Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from .reachy_service import ReachyService

logger = logging.getLogger(__name__)

class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking.
    
    Ported from user provided example.
    """

    def __init__(self, reachy_mini: ReachyMini, head_tracker: Any = None) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        # Thread-safe frame storage
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Face tracking state
        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        # Face tracking timing variables
        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 2.0
        self.interpolation_duration = 1.0

        # Track state changes
        self.previous_head_tracking_state = self.is_head_tracking_enabled

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        logger.debug("Starting camera working loop")
        neutral_pose = np.eye(4)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                if not self.reachy_mini:
                     logger.warning("CameraWorker: reachy_mini is None")
                     time.sleep(1)
                     continue

                # current_time = time.time()
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                    # Trace every 300 frames (~10s) to avoid spam, or just once
                    if not hasattr(self, "_logged_first_frame"):
                        logger.info("CameraWorker: Successfully captured first frame from ReachyMini")
                        self._logged_first_frame = True
                else:
                    if not hasattr(self, "_logged_none_frame"):
                        logger.warning("CameraWorker: get_frame() returned None")
                        self._logged_none_frame = True

                    # (Simplified) Face tracking logic would go here
                    # For now we just buffer the frame as requested.
                
                time.sleep(0.04) # ~25 FPS

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")


class CameraInputService:
    def __init__(self):
        self._task = None
        self._stop_event = threading.Event()
        self._thread = None
        self._worker = None
        self._stream_to_browser = True
        self._fps = 15

    def start(self, task):
        self._task = task
        self._stop_event.clear()
        
        # Capture the event loop where the pipeline is running
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("No running event loop found in start()")
            self._loop = None
            return

        # Get Reachy instance
        service = ReachyService.get_instance()
        if not service.connected or not service.robot:
             logger.warning("Reachy not connected, trying to connect...")
             service.connect()
        
        if service.robot:
            self._worker = CameraWorker(service.robot)
            self._worker.start()
            
            self._thread = threading.Thread(target=self._push_loop, daemon=True)
            self._thread.start()
            logger.info("CameraInputService started using Reachy SDK")
        else:
            logger.error("Failed to obtain Reachy robot instance. Camera service will not run.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._worker:
            self._worker.stop()
        logger.info("CameraInputService stopped")

    def _push_loop(self):
        while not self._stop_event.is_set():
            if self._worker:
                frame = self._worker.get_latest_frame()
                if frame is not None and self._task and self._loop:
                    try:
                        # Frame is BGR from OpenCV (used by Reachy SDK)
                        # Convert to RGB
                        # Import cv2 locally if needed or assume installed
                        import cv2
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame_rgb.shape
                        
                        # Log resolution once
                        if not hasattr(self, "_logged_resolution"):
                             logger.info(f"CameraInputService: Captured frame resolution: {w}x{h}, channels: {c}")
                             self._logged_resolution = True

                        user_frame = UserImageRawFrame(
                            user_id="robot_eye",
                            image=frame_rgb.tobytes(),
                            size=(w, h),
                            format="RGB"
                        )
                        
                        frames_to_send = [user_frame]
                        
                        if self._stream_to_browser:
                            out_frame = OutputImageRawFrame(
                                image=frame_rgb.tobytes(),
                                size=(w, h),
                                format="RGB"
                            )
                            frames_to_send.append(out_frame)

                        asyncio.run_coroutine_threadsafe(
                            self._task.queue_frames(frames_to_send),
                            self._loop
                        )
                    except Exception as e:
                        logger.error(f"Error pushing frame: {e}")
            
            time.sleep(1.0 / self._fps)
