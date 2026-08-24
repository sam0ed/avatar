"""OpenCV window for face frames, paced by the audio playback position.

Idle and speech share one continuous ping-pong cursor over the avatar's frame
cycle: idle plays local reference frames along the cursor, speech shows server
frames rendered for the same cycle positions (mouth replaced), and each shown
speech frame re-anchors the cursor — so mode switches never jump pose.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import cv2
import numpy as np

logger = logging.getLogger("avatar.video")

_DEFAULT_LIVE_FPS = 25
# Higgs generates audio 2.5-4x faster than real time, so frames arrive ~42fps
# against 25fps consumption; the buffer must absorb the whole surplus of a long
# turn or the deque evicts exactly the frames due next (video freezes mid-turn).
_FRAME_BUFFER_MAX = 3000

_WINDOW_NAME = "Avatar"


def pingpong_index(position: int, frame_count: int) -> int:
    """Map a monotonically growing position onto a forward-backward frame sweep.

    Must match the server's mapping in musetalk_render.pingpong_index exactly.
    """
    if frame_count <= 1:
        return 0
    period = 2 * frame_count - 2
    pos = position % period
    return pos if pos < frame_count else period - pos


class VideoDisplay:
    """Thread-based OpenCV window; live pacing follows audio_position()."""

    def __init__(
        self,
        audio_position: Callable[[], float | None] | None = None,
        on_key: Callable[[int], None] | None = None,
    ) -> None:
        self._audio_position = audio_position
        self._on_key = on_key
        self._frame_buffer: deque[tuple[int, bytes]] = deque(maxlen=_FRAME_BUFFER_MAX)
        self._live_fps: int = _DEFAULT_LIVE_FPS
        self._idle_frames: list[bytes] = []
        self._idle_fps: int = _DEFAULT_LIVE_FPS
        self._cursor: int = 0
        self._turn_offset: int = 0
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

    @property
    def cursor(self) -> int:
        """Current position in the shared frame cycle (send with each chat)."""
        with self._lock:
            return self._cursor

    def begin_live(self, turn_offset: int) -> None:
        """Enter live mode; server frame i maps to cycle position turn_offset + i."""
        with self._lock:
            self._turn_offset = turn_offset
            self._idle_mode = False

    def show_frame(self, jpeg_bytes: bytes, frame_idx: int, fps: int = _DEFAULT_LIVE_FPS) -> None:
        """Enqueue a JPEG frame under its per-turn index."""
        with self._lock:
            self._frame_buffer.append((frame_idx, jpeg_bytes))
            self._live_fps = fps or _DEFAULT_LIVE_FPS
            self._idle_mode = False

    def set_idle_frames(self, frames: list[bytes], fps: int = _DEFAULT_LIVE_FPS) -> None:
        """Set the full reference frame cycle for idle animation.

        Frames stay JPEG-encoded (~25MB total) and are decoded one at a time
        at display; pre-decoding 958 frames costs >1GB RAM and made the
        display loop stall on paging.
        """
        with self._lock:
            self._idle_frames = list(frames)
            self._idle_fps = fps or _DEFAULT_LIVE_FPS
        logger.info("Set %d idle frames at %d fps", len(frames), fps)

    def set_idle_mode(self, enabled: bool) -> None:
        """Switch between live and idle display modes; the cursor carries over."""
        with self._lock:
            self._idle_mode = enabled
            if enabled:
                self._frame_buffer.clear()

    def _due_frame(self) -> np.ndarray | None:
        """Newest frame due at the current audio position; None = hold."""
        position = self._audio_position() if self._audio_position else None
        if position is None:
            return None

        due_jpeg: bytes | None = None
        with self._lock:
            target_idx = int(position * self._live_fps)
            due_idx: int | None = None
            while self._frame_buffer and self._frame_buffer[0][0] <= target_idx:
                due_idx, due_jpeg = self._frame_buffer.popleft()
            if due_idx is not None:
                self._cursor = self._turn_offset + due_idx
        if due_jpeg is None:
            return None
        return self._decode_jpeg(due_jpeg)

    @property
    def buffer_size(self) -> int:
        """Number of frames waiting in the buffer."""
        return len(self._frame_buffer)

    def current_frame(self) -> np.ndarray | None:
        """Latest frame shown in the window (idle or live), for external sinks."""
        with self._lock:
            return self._current_frame

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
        last_idle_tick = time.monotonic()

        while self._running:
            now = time.monotonic()
            with self._lock:
                idle = self._idle_mode

            if idle:
                if self._idle_frames:
                    interval = 1.0 / self._idle_fps
                    advanced = int((now - last_idle_tick) / interval)
                    if advanced > 3:
                        # After a stall, resync the clock instead of skipping
                        # ahead — a visible jump is worse than lost time.
                        advanced = 1
                        last_idle_tick = now
                    elif advanced > 0:
                        last_idle_tick += advanced * interval
                    if advanced > 0:
                        with self._lock:
                            self._cursor += advanced
                            idx = pingpong_index(self._cursor, len(self._idle_frames))
                            jpeg = self._idle_frames[idx]
                        frame = self._decode_jpeg(jpeg)
                        if frame is not None:
                            with self._lock:
                                self._current_frame = frame
            else:
                last_idle_tick = now
                due = self._due_frame()
                if due is not None:
                    with self._lock:
                        self._current_frame = due

            with self._lock:
                frame = self._current_frame
            if frame is not None:
                cv2.imshow(_WINDOW_NAME, frame)

            key = cv2.waitKey(16) & 0xFF
            if key == 27:
                self._running = False
                break
            if key != 255 and self._on_key is not None:
                self._on_key(key)

        cv2.destroyWindow(_WINDOW_NAME)
