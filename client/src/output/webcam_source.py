"""Captures the real webcam for passthrough when the avatar is toggled off."""

import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger("avatar.output.webcam")


class WebcamSource:
    """Background capture of the physical webcam; latest frame on demand."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._frame = None

    def frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def _loop(self) -> None:
        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            logger.error("Webcam %d could not be opened", self._device_index)
            self._running = False
            return
        logger.info("Webcam %d capturing for passthrough", self._device_index)
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    continue
                with self._lock:
                    self._frame = frame
        finally:
            capture.release()
