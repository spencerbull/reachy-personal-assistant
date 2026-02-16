import time
import threading
import asyncio
import queue
import numpy as np
from typing import Any, List, Tuple, Optional
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from loguru import logger

from pipecat.frames.frames import UserImageRawFrame, OutputImageRawFrame, Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from .reachy_service import ReachyService

# Try to import face detection libraries
_FACE_DETECTION_AVAILABLE = False
_face_detector = None

try:
    import cv2

    # Try MediaPipe first (more accurate)
    try:
        import mediapipe as mp

        _FACE_DETECTION_AVAILABLE = True
        _FACE_DETECTION_METHOD = "mediapipe"
        logger.info("Face detection: Using MediaPipe")
    except ImportError:
        # Fall back to OpenCV Haar cascades
        _FACE_DETECTION_AVAILABLE = True
        _FACE_DETECTION_METHOD = "opencv"
        logger.info("Face detection: Using OpenCV Haar cascades")
except ImportError:
    logger.warning("Face detection not available: cv2 not installed")


class FaceDetector:
    """Face detection wrapper supporting multiple backends."""

    def __init__(self):
        self._method = None
        self._detector = None
        self._initialized = False

        if not _FACE_DETECTION_AVAILABLE:
            logger.warning("Face detection not available")
            return

        try:
            if _FACE_DETECTION_METHOD == "mediapipe":
                import mediapipe as mp

                self._mp_face_detection = mp.solutions.face_detection
                self._detector = self._mp_face_detection.FaceDetection(
                    model_selection=0,  # 0 for short-range (2m), 1 for full-range (5m)
                    min_detection_confidence=0.5,
                )
                self._method = "mediapipe"
            else:
                import cv2

                # Load Haar cascade
                cascade_path = (
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self._detector = cv2.CascadeClassifier(cascade_path)
                self._method = "opencv"

            self._initialized = True
            logger.info(f"FaceDetector initialized with {self._method}")

        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect face in frame.

        Returns:
            Tuple of (center_x, center_y, width, height) normalized to 0-1,
            or None if no face detected
        """
        if not self._initialized or frame is None:
            return None

        try:
            import cv2

            if self._method == "mediapipe":
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._detector.process(rgb_frame)

                if results.detections:
                    # Get the largest face by bounding box area
                    largest_detection = max(
                        results.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height,
                    )
                    bbox = largest_detection.location_data.relative_bounding_box

                    # Calculate center
                    center_x = bbox.xmin + bbox.width / 2
                    center_y = bbox.ymin + bbox.height / 2

                    return (center_x, center_y, bbox.width, bbox.height)

            else:  # OpenCV Haar cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(faces) > 0:
                    # Get the largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                    # Normalize to 0-1
                    height, width = frame.shape[:2]
                    center_x = (x + w / 2) / width
                    center_y = (y + h / 2) / height
                    norm_w = w / width
                    norm_h = h / height

                    return (center_x, center_y, norm_w, norm_h)

        except Exception as e:
            logger.debug(f"Face detection error: {e}")

        return None

    def close(self):
        """Clean up detector resources."""
        if self._method == "mediapipe" and self._detector:
            self._detector.close()


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking.

    Enhanced with real face detection and head pose offset calculation.
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

        # Face detection
        self._face_detector = FaceDetector()

        # Face tracking state
        self.is_face_tracking_enabled = True  # Enabled by default
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,  # x, y, z
            0.0,
            0.0,
            0.0,  # roll, pitch, yaw
        ]
        self.face_tracking_lock = threading.Lock()

        # Last detected face position (normalized 0-1)
        self.last_face_position: Optional[Tuple[float, float]] = None
        self.last_face_size: Optional[Tuple[float, float]] = None

        # Face tracking timing variables
        self.last_face_detected_time: float | None = None
        self.face_lost_delay = 2.0  # Seconds before resetting to neutral

        # Offset-based tracking configuration (smooth 100Hz blending via MovementManager)
        # Target position: top 1/3 of frame so Reachy looks UP at user naturally
        self._target_face_x: float = 0.5  # Horizontal center
        self._target_face_y: float = 0.33  # Top third of frame

        # Smoothing for offset calculation
        # Higher alpha = faster response, lower = smoother but slower
        self._offset_smoothing_alpha = 0.15  # Moderate smoothing - responds in ~0.5 sec

        # Current smoothed offsets (pitch and yaw in radians)
        self._smoothed_pitch_offset: float = 0.0
        self._smoothed_yaw_offset: float = 0.0

        # Gain: how much the head moves per unit of face deviation
        # These need to be high enough to actually move the head to center the face
        self._pitch_gain = (
            0.8  # radians per normalized deviation (~46 deg for full deviation)
        )
        self._yaw_gain = (
            1.0  # radians per normalized deviation (~57 deg for full deviation)
        )

        # Dead zone - ignore small deviations (prevents micro-movements)
        self._dead_zone_threshold = 0.08  # 8% of frame

        # Maximum offset limits (prevent extreme head positions)
        self._max_pitch_offset = 0.5  # ~28 degrees
        self._max_yaw_offset = 0.7  # ~40 degrees

        # Movement logging
        self._movement_log_file = "/tmp/reachy_movements.log"
        self._init_movement_log()

        # Antenna wave state - track if we've already waved for this face
        self._was_face_detected = False
        self._antenna_wave_triggered = False

        # Soul system callback for face events
        # Set via set_soul() after construction
        self._soul = None

        # Track consecutive errors for timeout protection
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3

    def set_soul(self, soul) -> None:
        """Set the soul system reference for face detection events.

        Called after pipeline construction to wire face detected/lost events
        into the soul system.
        """
        self._soul = soul

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            return tuple(self.face_tracking_offsets)

    def enable_face_tracking(self, enable: bool = True):
        """Enable or disable face tracking."""
        self.is_face_tracking_enabled = enable
        self._log_movement(
            "CONFIG", f"Face tracking {'enabled' if enable else 'disabled'}"
        )
        logger.info(f"Face tracking {'enabled' if enable else 'disabled'}")

        if not enable:
            # Reset all state when disabling
            with self.face_tracking_lock:
                self.face_tracking_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self._was_face_detected = False
            self._antenna_wave_triggered = False
            self.last_face_detected_time = None
            self.last_face_position = None
            self.last_face_size = None
            # Reset offset smoothing state
            self._smoothed_pitch_offset = 0.0
            self._smoothed_yaw_offset = 0.0

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
        if self._face_detector:
            self._face_detector.close()
        logger.debug("Camera worker stopped")

    def _init_movement_log(self):
        """Initialize the movement log file."""
        try:
            with open(self._movement_log_file, "w") as f:
                f.write(
                    f"# Reachy Movement Log - Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write("# Format: timestamp | action | details\n")
                f.write("-" * 80 + "\n")
            logger.info(f"Movement log initialized: {self._movement_log_file}")
        except Exception as e:
            logger.warning(f"Failed to init movement log: {e}")

    def _log_movement(self, action: str, details: str):
        """Log a movement to the movement log file."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            with open(self._movement_log_file, "a") as f:
                f.write(f"{timestamp} | {action} | {details}\n")
        except Exception:
            pass  # Don't let logging failures affect tracking

    def _update_face_tracking(self, frame: np.ndarray):
        """Update face tracking offsets for smooth 100Hz blending via MovementManager.

        Calculates pitch/yaw offsets based on face deviation from target position.
        These offsets are read by the MovementManager at 100Hz and smoothly blended
        with the primary pose, creating fluid natural movement.
        """
        if not self.is_face_tracking_enabled:
            return

        face_result = self._face_detector.detect(frame)
        current_time = time.time()

        if face_result:
            center_x_norm, center_y_norm, width_norm, height_norm = face_result
            self.last_face_detected_time = current_time
            self.last_face_position = (center_x_norm, center_y_norm)
            self.last_face_size = (width_norm, height_norm)

            # Check if this is the first face detection (for antenna wave and soul event)
            if not self._was_face_detected:
                self._was_face_detected = True
                self._trigger_antenna_wave()
                self._log_movement(
                    "WAVE",
                    f"First face detected at ({center_x_norm:.2f}, {center_y_norm:.2f})",
                )
                logger.info("Face detected - triggered antenna wave greeting")
                # Notify soul system of new face
                if self._soul:
                    try:
                        self._soul.on_face_detected(
                            position=(center_x_norm, center_y_norm),
                            is_new=True,
                        )
                    except Exception as e:
                        logger.debug(f"Soul face detected event failed: {e}")

            # Calculate deviation from target position
            # Positive x deviation = face is to the RIGHT of target = turn RIGHT (negative yaw)
            # Positive y deviation = face is BELOW target = look DOWN (positive pitch)
            deviation_x = center_x_norm - self._target_face_x
            deviation_y = center_y_norm - self._target_face_y

            # Apply dead zone - ignore small deviations
            if abs(deviation_x) < self._dead_zone_threshold:
                deviation_x = 0.0
            if abs(deviation_y) < self._dead_zone_threshold:
                deviation_y = 0.0

            # Calculate target offsets (with gain and sign correction)
            # Yaw: negative because looking right requires negative yaw
            # Pitch: positive because looking down requires positive pitch
            target_yaw = -deviation_x * self._yaw_gain
            target_pitch = deviation_y * self._pitch_gain

            # Clamp to maximum values
            target_yaw = max(
                -self._max_yaw_offset, min(self._max_yaw_offset, target_yaw)
            )
            target_pitch = max(
                -self._max_pitch_offset, min(self._max_pitch_offset, target_pitch)
            )

            # Apply heavy smoothing (low-pass filter) for fluid movement
            # This is where the magic happens - very slow smooth transitions
            self._smoothed_yaw_offset = (
                self._offset_smoothing_alpha * target_yaw
                + (1 - self._offset_smoothing_alpha) * self._smoothed_yaw_offset
            )
            self._smoothed_pitch_offset = (
                self._offset_smoothing_alpha * target_pitch
                + (1 - self._offset_smoothing_alpha) * self._smoothed_pitch_offset
            )

            # Update the face tracking offsets (read by MovementManager at 100Hz)
            with self.face_tracking_lock:
                self.face_tracking_offsets = [
                    0.0,  # x translation
                    0.0,  # y translation
                    0.0,  # z translation
                    0.0,  # roll
                    self._smoothed_pitch_offset,  # pitch
                    self._smoothed_yaw_offset,  # yaw
                ]

            # Log occasionally (not every frame)
            if (
                not hasattr(self, "_last_log_time")
                or current_time - self._last_log_time > 2.0
            ):
                self._log_movement(
                    "OFFSET",
                    f"yaw={self._smoothed_yaw_offset:.3f}, pitch={self._smoothed_pitch_offset:.3f}, dev=({deviation_x:.2f},{deviation_y:.2f})",
                )
                self._last_log_time = current_time

        elif self.last_face_detected_time:
            # Face lost - gradually return offsets to zero
            time_since_face = current_time - self.last_face_detected_time

            if time_since_face > self.face_lost_delay:
                # Reset antenna wave state
                if self._was_face_detected:
                    # Notify soul system of face lost (before resetting state)
                    if self._soul:
                        try:
                            self._soul.on_face_lost()
                        except Exception as e:
                            logger.debug(f"Soul face lost event failed: {e}")
                self._was_face_detected = False

                # Decay offsets toward zero (smooth return to neutral)
                decay = 0.95  # Slow decay for smooth return
                self._smoothed_yaw_offset *= decay
                self._smoothed_pitch_offset *= decay

                # Update offsets
                with self.face_tracking_lock:
                    self.face_tracking_offsets = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        self._smoothed_pitch_offset,
                        self._smoothed_yaw_offset,
                    ]

                # Reset when very close to zero
                if (
                    abs(self._smoothed_yaw_offset) < 0.001
                    and abs(self._smoothed_pitch_offset) < 0.001
                ):
                    self.last_face_detected_time = None
                    self._log_movement(
                        "NEUTRAL", f"Returned to neutral after face lost"
                    )

    def _trigger_antenna_wave(self):
        """Trigger an antenna wave greeting when a face is first detected."""
        try:
            from .reachy_service import ReachyService
            from .dance_emotion_moves import AntennaWaveMove

            service = ReachyService.get_instance()
            if service and service.motion_manager:
                wave_move = AntennaWaveMove(duration=0.8)
                service.motion_manager.queue_move(wave_move)
                logger.info("Queued antenna wave greeting")
        except Exception as e:
            logger.warning(f"Failed to trigger antenna wave: {e}")

    def working_loop(self) -> None:
        logger.debug("Starting camera working loop")

        frame_count = 0
        face_detect_interval = 3  # Run face detection every N frames

        while not self._stop_event.is_set():
            try:
                if not self.reachy_mini:
                    logger.warning("CameraWorker: reachy_mini is None")
                    time.sleep(1)
                    continue

                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame

                    # Log first frame
                    if not hasattr(self, "_logged_first_frame"):
                        logger.info(
                            "CameraWorker: Successfully captured first frame from ReachyMini"
                        )
                        self._logged_first_frame = True

                    # Run face detection periodically (not every frame for performance)
                    frame_count += 1
                    if frame_count % face_detect_interval == 0:
                        self._update_face_tracking(frame)

                else:
                    if not hasattr(self, "_logged_none_frame"):
                        logger.warning("CameraWorker: get_frame() returned None")
                        self._logged_none_frame = True

                time.sleep(0.04)  # ~25 FPS

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")


class CameraFrameProcessor(FrameProcessor):
    """Frame processor that injects camera frames into the pipeline.

    This processor runs a background thread that captures camera frames and
    an async task that pushes them into the pipeline. OutputImageRawFrame goes
    to the browser, UserImageRawFrame is captured by the LLM service for vision.

    CRITICAL: Camera capture starts IMMEDIATELY on construction to avoid race
    conditions with WebRTC video track initialization.
    """

    _instance = None

    def __init__(
        self, fps: int = 15, target_width: int = 1280, target_height: int = 720
    ):
        super().__init__()
        self._fps = fps
        self._target_width = target_width
        self._target_height = target_height
        self._worker: Optional[CameraWorker] = None
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._push_task: Optional[asyncio.Task] = None  # Async task for pushing frames
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task = None  # Will be set by set_task()
        self._output_transport = None  # Will be set by set_output_transport()
        self._started = False
        self._capturing = False  # Whether capture thread is running

        # Thread-safe queue for buffering frames before async loop is ready
        self._frame_buffer: queue.Queue = queue.Queue(maxsize=2)

        # Async queue for frames (created when start() is called)
        self._frame_queue: asyncio.Queue = None

        # Track latest frame for vision queries
        self._last_user_frame: Optional[UserImageRawFrame] = None
        self._frame_count = 0

        # Latest frame data for immediate write to video track
        self._latest_frame: Optional[OutputImageRawFrame] = None

        # Create a placeholder black frame immediately
        # This ensures there's always SOMETHING in the video track buffer
        self._create_placeholder_frame()

        # Register as singleton instance
        CameraFrameProcessor._instance = self

        # Start camera capture immediately (before WebRTC connection)
        self._start_capture_early()

    def _create_placeholder_frame(self):
        """Create a black placeholder frame for immediate use.

        This ensures the video track has something to send while waiting
        for the camera to produce its first real frame.
        """
        import numpy as np

        # Create a black frame with the target dimensions
        black_frame = np.zeros(
            (self._target_height, self._target_width, 3), dtype=np.uint8
        )
        image_bytes = black_frame.tobytes()

        self._latest_frame = OutputImageRawFrame(
            image=image_bytes,
            size=(self._target_width, self._target_height),
            format="RGB",
        )

        self._last_user_frame = UserImageRawFrame(
            user_id="robot_eye",
            image=image_bytes,
            size=(self._target_width, self._target_height),
            format="RGB",
        )

        logger.info(
            f"Created placeholder black frame {self._target_width}x{self._target_height}"
        )

    @classmethod
    def get_instance(cls) -> Optional["CameraFrameProcessor"]:
        """Get the singleton instance."""
        return cls._instance

    def set_task(self, task):
        """Set the pipeline task for frame queueing."""
        self._task = task

    def set_output_transport(self, output_transport):
        """Set the output transport (kept for compatibility).

        NOTE: We no longer write directly to the output transport. Instead,
        OutputImageRawFrame is pushed through the pipeline and flows to the
        output transport's MediaOutput._video_task_handler automatically.
        """
        self._output_transport = output_transport
        logger.info(f"Output transport set: {type(output_transport).__name__}")

    def _start_capture_early(self):
        """Start camera capture BEFORE WebRTC connection is established.

        This is called during __init__ to ensure frames are ready when
        the WebRTC video track starts calling recv(). This avoids the
        race condition where the browser disables receivers because
        no frames are available.
        """
        logger.info("Starting early camera capture...")

        # Get or create Reachy connection
        service = ReachyService.get_instance()
        if not service.connected or not service.robot:
            logger.info("Connecting to Reachy for early camera capture...")
            service.connect()

        if service.robot:
            logger.info("Starting CameraWorker early (before WebRTC connection)")
            self._worker = CameraWorker(service.robot)
            self._worker.start()

            if service.motion_manager:
                service.motion_manager.camera_worker = self._worker

            # Start capture thread that buffers frames
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_frames_loop, daemon=True
            )
            self._capture_thread.start()
            self._capturing = True
            logger.info("Early camera capture started - frames are being buffered")
        else:
            logger.warning("No Reachy robot available for early capture")

    async def start(self):
        """Start pushing captured frames to the transport.

        This is called when the client connects. The capture thread should
        already be running from _start_capture_early().
        """
        if self._started:
            return

        logger.info("CameraFrameProcessor activating frame pushing...")

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("No running event loop found")
            return

        # Create async queue for frames
        self._frame_queue = asyncio.Queue(
            maxsize=2
        )  # Small buffer, we only need latest

        # If early capture didn't start, try now (fallback)
        if not self._capturing:
            logger.warning("Early capture not running, starting now...")
            service = ReachyService.get_instance()
            if not service.connected or not service.robot:
                service.connect()

            if service.robot:
                logger.info("Initializing CameraWorker (late start)")
                self._worker = CameraWorker(service.robot)
                self._worker.start()

                if service.motion_manager:
                    service.motion_manager.camera_worker = self._worker

                self._stop_event.clear()
                self._capture_thread = threading.Thread(
                    target=self._capture_frames_loop, daemon=True
                )
                self._capture_thread.start()
                self._capturing = True
            else:
                logger.warning("No Reachy robot available")
                return

        # Start async task to push frames from the queue
        self._push_task = asyncio.create_task(self._push_frames_async())

        self._started = True
        logger.info("CameraFrameProcessor started")

    async def stop(self):
        """Stop the camera capture."""
        self._stop_event.set()
        if self._push_task:
            self._push_task.cancel()
            try:
                await self._push_task
            except asyncio.CancelledError:
                pass
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._worker:
            self._worker.stop()
        self._started = False
        logger.info("CameraFrameProcessor stopped")

    def _capture_frames_loop(self):
        """Background thread that captures frames.

        This runs BEFORE the async loop is ready. It always captures and stores
        the latest frame. When the async queue is available, it also puts frames
        there for the push task to consume.
        """
        import cv2

        while not self._stop_event.is_set():
            if self._worker:
                frame = self._worker.get_latest_frame()
                if frame is not None:
                    try:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame_rgb.shape

                        # Resize to target
                        if w != self._target_width or h != self._target_height:
                            frame_rgb = cv2.resize(
                                frame_rgb, (self._target_width, self._target_height)
                            )
                            h, w = self._target_height, self._target_width

                        if self._frame_count == 0:
                            logger.info(f"CameraFrameProcessor: First frame - {w}x{h}")

                        # Create frames
                        image_bytes = frame_rgb.tobytes()

                        out_frame = OutputImageRawFrame(
                            image=image_bytes, size=(w, h), format="RGB"
                        )

                        user_frame = UserImageRawFrame(
                            user_id="robot_eye",
                            image=image_bytes,
                            size=(w, h),
                            format="RGB",
                        )

                        # Always store latest frames for immediate access
                        self._latest_frame = out_frame
                        self._last_user_frame = user_frame

                        # If async queue is ready, put frames there
                        if self._loop and self._frame_queue:
                            try:
                                self._frame_queue.put_nowait((out_frame, user_frame))
                            except asyncio.QueueFull:
                                # Drop frame if queue is full - we only want latest
                                pass

                        self._frame_count += 1
                        if self._frame_count % 100 == 0:
                            logger.debug(
                                f"CameraFrameProcessor: Captured {self._frame_count} frames"
                            )

                    except Exception as e:
                        logger.error(f"Error in camera capture loop: {e}")

            time.sleep(1.0 / self._fps)

    async def _push_frames_async(self):
        """Async task that reads from queue and writes video frames directly to transport.

        We call write_video_frame directly on the output transport to ensure
        frames reach the WebRTC video track. Also push through pipeline for LLM vision.
        """
        push_count = 0
        video_write_ok = 0
        video_write_fail = 0

        while True:
            try:
                # Wait for frames from the capture thread
                out_frame, user_frame = await self._frame_queue.get()

                # Write OutputImageRawFrame directly to output transport
                # This bypasses the pipeline and writes directly to RawVideoTrack
                if self._output_transport:
                    try:
                        # Get the underlying client from the transport
                        client = getattr(self._output_transport, "_client", None)
                        if client:
                            success = await client.write_video_frame(out_frame)
                            if success:
                                video_write_ok += 1
                            else:
                                video_write_fail += 1
                                if video_write_fail <= 5:
                                    logger.warning(
                                        f"write_video_frame returned False (frame {push_count})"
                                    )
                        else:
                            # Fallback: call write_video_frame on the transport directly
                            success = await self._output_transport.write_video_frame(
                                out_frame
                            )
                            if success:
                                video_write_ok += 1
                            else:
                                video_write_fail += 1
                                if video_write_fail <= 5:
                                    logger.warning(
                                        f"transport.write_video_frame returned False (frame {push_count})"
                                    )
                    except Exception as e:
                        video_write_fail += 1
                        if video_write_fail <= 5:
                            logger.error(f"Error writing video frame: {e}")
                else:
                    video_write_fail += 1
                    if video_write_fail <= 5:
                        logger.warning("No output transport for video frames")

                # Also push UserImageRawFrame through pipeline for LLM vision
                await self.push_frame(user_frame, FrameDirection.DOWNSTREAM)

                push_count += 1
                if push_count == 1:
                    logger.info(
                        f"CameraFrameProcessor: First video frame write attempted"
                    )
                if push_count % 100 == 0:
                    logger.info(
                        f"CameraFrameProcessor: {push_count} frames - video writes: {video_write_ok} ok, {video_write_fail} failed"
                    )

            except asyncio.CancelledError:
                logger.info("CameraFrameProcessor push task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in push frames loop: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and pass through."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    def get_last_frame(self) -> Optional[UserImageRawFrame]:
        """Get the last captured user image frame."""
        return self._last_user_frame

    def enable_face_tracking(self, enable: bool = True):
        """Enable or disable face tracking."""
        if self._worker:
            self._worker.enable_face_tracking(enable)


class CameraInputService:
    """Camera input service with face tracking control (legacy wrapper)."""

    _instance = None

    def __init__(self):
        self._task = None
        self._stop_event = threading.Event()
        self._thread = None
        self._worker: Optional[CameraWorker] = None
        self._stream_to_browser = True
        self._fps = 15
        self._use_local_camera = False
        self._local_cap = None
        CameraInputService._instance = self

    @classmethod
    def get_instance(cls) -> Optional["CameraInputService"]:
        """Get the singleton instance."""
        return cls._instance

    @property
    def worker(self) -> Optional[CameraWorker]:
        """Get the camera worker."""
        return self._worker

    def enable_face_tracking(self, enable: bool = True):
        """Enable or disable face tracking."""
        if self._worker:
            self._worker.enable_face_tracking(enable)

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets."""
        if self._worker:
            return self._worker.get_face_tracking_offsets()
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_last_face_position(self) -> Optional[Tuple[float, float]]:
        """Get last detected face position (normalized 0-1)."""
        if self._worker:
            return self._worker.last_face_position
        return None

    def start(self, task):
        logger.info("CameraInputService.start() called")
        self._task = task
        self._stop_event.clear()
        self._use_local_camera = False
        self._local_cap = None

        # Capture the event loop where the pipeline is running
        try:
            self._loop = asyncio.get_running_loop()
            logger.debug(f"Captured event loop: {self._loop}")
        except RuntimeError as e:
            logger.error(f"No running event loop found in start(): {e}")
            self._loop = None
            return

        # Try Reachy first
        service = ReachyService.get_instance()
        logger.debug(
            f"ReachyService state: connected={service.connected}, has_robot={service.robot is not None}"
        )

        if not service.connected or not service.robot:
            logger.warning("Reachy not connected, trying to connect...")
            service.connect()

        logger.debug(
            f"After connect attempt: connected={service.connected}, has_robot={service.robot is not None}"
        )

        if service.robot:
            logger.info("Initializing CameraWorker with Reachy robot")
            self._worker = CameraWorker(service.robot)
            self._worker.start()

            # Connect face tracking to movement manager
            if service.motion_manager:
                service.motion_manager.camera_worker = self._worker

            self._thread = threading.Thread(target=self._push_loop, daemon=True)
            self._thread.start()
            logger.info("CameraInputService started using Reachy SDK")
        else:
            # Fallback to local webcam
            logger.warning("Reachy not available, trying local webcam fallback...")
            try:
                import cv2

                self._local_cap = cv2.VideoCapture(0)
                if self._local_cap.isOpened():
                    self._use_local_camera = True
                    self._thread = threading.Thread(
                        target=self._push_loop_local, daemon=True
                    )
                    self._thread.start()
                    logger.info(
                        "CameraInputService started using local webcam fallback"
                    )
                else:
                    logger.error(
                        "Failed to open local webcam. Camera service will not run."
                    )
                    self._local_cap = None
            except Exception as e:
                logger.error(f"Failed to initialize local webcam: {e}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._worker:
            self._worker.stop()
        if self._local_cap:
            self._local_cap.release()
            self._local_cap = None
        logger.info("CameraInputService stopped")

    def _push_loop(self):
        logger.debug("_push_loop started (Reachy camera)")
        frame_count = 0

        # Target resolution matching transport config
        target_width = 1280
        target_height = 720

        while not self._stop_event.is_set():
            if self._worker:
                frame = self._worker.get_latest_frame()
                if frame is not None and self._task and self._loop:
                    try:
                        import cv2

                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame_rgb.shape

                        # Resize to match transport expected size
                        if w != target_width or h != target_height:
                            frame_rgb = cv2.resize(
                                frame_rgb, (target_width, target_height)
                            )
                            h, w = target_height, target_width

                        # Log resolution once
                        if not hasattr(self, "_logged_resolution"):
                            logger.info(
                                f"CameraInputService: Streaming at {w}x{h} (original camera resolution may differ)"
                            )
                            self._logged_resolution = True

                        # Create UserImageRawFrame for LLM vision processing
                        user_frame = UserImageRawFrame(
                            user_id="robot_eye",
                            image=frame_rgb.tobytes(),
                            size=(w, h),
                            format="RGB",
                        )

                        # Create OutputImageRawFrame for browser display
                        if self._stream_to_browser:
                            out_frame = OutputImageRawFrame(
                                image=frame_rgb.tobytes(), size=(w, h), format="RGB"
                            )
                            # Queue both frames
                            asyncio.run_coroutine_threadsafe(
                                self._task.queue_frames([user_frame, out_frame]),
                                self._loop,
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                self._task.queue_frames([user_frame]), self._loop
                            )

                        frame_count += 1
                        if frame_count == 1:
                            logger.info(
                                f"First frame queued to pipeline (Reachy) - size {w}x{h}"
                            )
                        elif frame_count % 100 == 0:
                            logger.debug(f"Pushed {frame_count} frames to pipeline")

                    except Exception as e:
                        logger.error(f"Error pushing frame: {e}", exc_info=True)
                elif frame is None and not hasattr(self, "_logged_no_frame"):
                    logger.warning("_push_loop: get_latest_frame returned None")
                    self._logged_no_frame = True

            time.sleep(1.0 / self._fps)

    def _push_loop_local(self):
        """Push loop for local webcam fallback."""
        import cv2

        logger.debug("_push_loop_local started (local webcam)")
        frame_count = 0

        # Target resolution matching transport config
        target_width = 1280
        target_height = 720

        while not self._stop_event.is_set():
            if self._local_cap and self._local_cap.isOpened():
                ret, frame = self._local_cap.read()
                if ret and frame is not None and self._task and self._loop:
                    try:
                        # Frame is BGR from OpenCV, convert to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame_rgb.shape

                        # Resize to match transport expected size
                        if w != target_width or h != target_height:
                            frame_rgb = cv2.resize(
                                frame_rgb, (target_width, target_height)
                            )
                            h, w = target_height, target_width

                        # Log resolution once
                        if not hasattr(self, "_logged_resolution_local"):
                            logger.info(
                                f"CameraInputService (local): Streaming at {w}x{h}"
                            )
                            self._logged_resolution_local = True

                        user_frame = UserImageRawFrame(
                            user_id="local_camera",
                            image=frame_rgb.tobytes(),
                            size=(w, h),
                            format="RGB",
                        )

                        if self._stream_to_browser:
                            out_frame = OutputImageRawFrame(
                                image=frame_rgb.tobytes(), size=(w, h), format="RGB"
                            )
                            asyncio.run_coroutine_threadsafe(
                                self._task.queue_frames([user_frame, out_frame]),
                                self._loop,
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                self._task.queue_frames([user_frame]), self._loop
                            )

                        frame_count += 1
                        if frame_count == 1:
                            logger.info(
                                f"First frame queued to pipeline (local webcam) - size {w}x{h}"
                            )
                        elif frame_count % 100 == 0:
                            logger.debug(f"Pushed {frame_count} frames (local webcam)")

                    except Exception as e:
                        logger.error(
                            f"Error pushing local camera frame: {e}", exc_info=True
                        )

            time.sleep(1.0 / self._fps)
