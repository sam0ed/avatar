"""Video display with OpenCV window for face animation frames.

Renders JPEG frames received from the server via chat_video messages.
Uses a dedicated thread for cv2.imshow (OpenCV requires the main thread
or a dedicated thread for GUI updates on Windows).

Supports:
  - Live frames: show_frame(jpeg_bytes) for real-time animation
  - Idle mode: cycle through reference frames at ~5 FPS when not speaking
  - Start/stop lifecycle matching AudioPlayer pattern
"""

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger("avatar.video")

# Idle animation frame rate (cycles through reference frames)
_IDLE_FPS = 5
_IDLE_INTERVAL = 1.0 / _IDLE_FPS

# Window name
_WINDOW_NAME = "Avatar"


class VideoDisplay:
    """Thread-based OpenCV window for displaying avatar video frames.

    Frames are pushed via show_frame() from the async message handler.
    A dedicated thread runs the cv2 event loop and renders frames.
    """

    def __init__(self) -> None:
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=60)  # ~2.4s at 25fps
        self._idle_frames: list[np.ndarray] = []
        self._idle_idx: int = 0
        self._current_frame: np.ndarray | None = None
        self._running = False
        self._idle_mode = True
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the display thread and open the OpenCV window."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()
        logger.info("VideoDisplay started")

    def stop(self) -> None:
        """Stop the display thread and close the window."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("VideoDisplay stopped")

    def show_frame(self, jpeg_bytes: bytes) -> None:
        """Decode and enqueue a JPEG frame for display."""
        frame = self._decode_jpeg(jpeg_bytes)
        if frame is not None:
            with self._lock:
                self._frame_buffer.append(frame)
                self._idle_mode = False

    def set_idle_frames(self, frames: list[bytes]) -> None:
        """Set reference frames for idle animation (JPEG bytes)."""
        decoded = []
        for jpeg in frames:
            frame = self._decode_jpeg(jpeg)
            if frame is not None:
                decoded.append(frame)
        with self._lock:
            self._idle_frames = decoded
            self._idle_idx = 0
        logger.info("Set %d idle frames", len(decoded))

    def set_idle_mode(self, enabled: bool) -> None:
        """Switch between live and idle display modes."""
        with self._lock:
            self._idle_mode = enabled
            if enabled:
                self._frame_buffer.clear()

    def advance_frame(self) -> None:
        """Advance to the next buffered frame (called by audio timing)."""
        with self._lock:
            if self._frame_buffer:
                self._current_frame = self._frame_buffer.popleft()

    @property
    def buffer_size(self) -> int:
        """Number of frames waiting in the buffer."""
        return len(self._frame_buffer)

    def _decode_jpeg(self, jpeg_bytes: bytes) -> np.ndarray | None:
        """Decode JPEG bytes to a BGR numpy array."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Failed to decode JPEG frame (%d bytes)", len(jpeg_bytes))
        return frame

    def _display_loop(self) -> None:
        """Main display thread loop — renders frames via cv2.imshow."""
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        last_idle_time = 0.0

        while self._running:
            with self._lock:
                idle = self._idle_mode
                frame = self._current_frame

            if idle:
                # Idle mode: cycle reference frames at _IDLE_FPS
                now = time.monotonic()
                if self._idle_frames and now - last_idle_time >= _IDLE_INTERVAL:
                    with self._lock:
                        frame = self._idle_frames[self._idle_idx]
                        self._idle_idx = (self._idle_idx + 1) % len(self._idle_frames)
                        self._current_frame = frame
                    last_idle_time = now

            if frame is not None:
                cv2.imshow(_WINDOW_NAME, frame)

            # cv2.waitKey is required for the window to process events.
            # 16ms ≈ 60Hz refresh rate for the GUI loop.
            key = cv2.waitKey(16) & 0xFF
            if key == 27:  # ESC to close
                self._running = False
                break

        cv2.destroyWindow(_WINDOW_NAME)
