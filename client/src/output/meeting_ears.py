"""Feeds the meeting's own audio (system loopback) into the ASR.

The avatar's ears in meeting mode: whatever the meeting app plays on the
default speakers — i.e. the other participants — is captured via WASAPI
loopback, downmixed and resampled to the ASR's 16kHz mono format. The
avatar's own voice never appears here because meeting apps do not play a
participant's microphone back to them.
"""

import logging
import threading
from collections.abc import Callable

import numpy as np
import pyaudiowpatch
from scipy.signal import resample_poly

logger = logging.getLogger("avatar.output.ears")

ASR_SAMPLE_RATE = 16_000
ASR_BLOCK_SAMPLES = 1600


class MeetingEars:
    """WASAPI-loopback capture of the default output, pushed to the transcriber."""

    def __init__(self, push_audio: Callable[[np.ndarray], None]) -> None:
        self._push_audio = push_audio
        self._running = False
        self._thread: threading.Thread | None = None
        self._pending = np.empty(0, dtype=np.float32)

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

    def _loop(self) -> None:
        try:
            with pyaudiowpatch.PyAudio() as audio:
                device = audio.get_default_wasapi_loopback()
                rate = int(device["defaultSampleRate"])
                channels = int(device["maxInputChannels"])
                logger.info(
                    "Meeting ears on '%s' (%d Hz, %d ch)", device["name"], rate, channels
                )
                stream = audio.open(
                    format=pyaudiowpatch.paInt16,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=rate // 10,
                    input=True,
                    input_device_index=device["index"],
                )
                try:
                    while self._running:
                        raw = stream.read(rate // 10, exception_on_overflow=False)
                        self._ingest(raw, rate, channels)
                finally:
                    stream.stop_stream()
                    stream.close()
        except Exception:
            logger.exception("Meeting ears stopped unexpectedly")
        finally:
            self._running = False

    def _ingest(self, raw: bytes, rate: int, channels: int) -> None:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        mono = samples.reshape(-1, channels).mean(axis=1)
        resampled = resample_poly(mono, ASR_SAMPLE_RATE, rate).astype(np.float32)
        self._pending = np.concatenate([self._pending, resampled])
        while len(self._pending) >= ASR_BLOCK_SAMPLES:
            block, self._pending = (
                self._pending[:ASR_BLOCK_SAMPLES],
                self._pending[ASR_BLOCK_SAMPLES:],
            )
            self._push_audio(block)
