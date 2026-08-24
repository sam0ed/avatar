"""Streams frames into the OBS Virtual Camera at a fixed rate."""

import logging
import threading
from collections.abc import Callable

import cv2
import numpy as np
import pyvirtualcam

logger = logging.getLogger("avatar.output.camera")

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
FPS = 25


class VirtualCameraSink:
    """Pulls BGR frames from a source callable and sends them to the virtual camera.

    Frames of any size are letterboxed onto a fixed canvas so the camera never
    has to be reopened when the source switches between avatar and webcam.
    """

    def __init__(self, get_frame: Callable[[], np.ndarray | None], fps: int = FPS) -> None:
        self._get_frame = get_frame
        self._fps = fps
        self._running = False
        self._thread: threading.Thread | None = None
        self._canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _compose(self, frame: np.ndarray) -> np.ndarray:
        scale = min(CANVAS_WIDTH / frame.shape[1], CANVAS_HEIGHT / frame.shape[0])
        width = round(frame.shape[1] * scale)
        height = round(frame.shape[0] * scale)
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        self._canvas[:] = 0
        y = (CANVAS_HEIGHT - height) // 2
        x = (CANVAS_WIDTH - width) // 2
        self._canvas[y:y + height, x:x + width] = resized
        return self._canvas

    def _loop(self) -> None:
        try:
            with pyvirtualcam.Camera(width=CANVAS_WIDTH, height=CANVAS_HEIGHT, fps=self._fps) as cam:
                logger.info("Virtual camera streaming: %s at %d fps", cam.device, self._fps)
                while self._running:
                    frame = self._get_frame()
                    if frame is not None:
                        cam.send(cv2.cvtColor(self._compose(frame), cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()
        except Exception:
            logger.exception("Virtual camera stopped unexpectedly")
        finally:
            self._running = False
