"""Copies the real microphone into VB-Cable while the avatar is toggled off."""

import logging
import threading

import sounddevice as sd

logger = logging.getLogger("avatar.output.mic")

SAMPLE_RATE = 48000
BLOCK_SIZE = 960


class MicPassthrough:
    """Duplex mic-to-cable loop so the meeting hears the real user in 'me' mode."""

    def __init__(self, cable_device: int) -> None:
        self._cable_device = cable_device
        self._stream: sd.Stream | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=1,
                dtype="int16",
                device=(None, self._cable_device),
                callback=self._copy_block,
            )
            self._stream.start()
            logger.info("Mic passthrough started (mic -> VB-Cable)")

    def stop(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        logger.info("Mic passthrough stopped")

    @staticmethod
    def _copy_block(indata, outdata, frames, time_info, status) -> None:
        outdata[:] = indata
