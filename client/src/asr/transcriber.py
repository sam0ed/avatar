"""ASR transcriber — Moonshine Voice wrapper for live speech-to-text.

Uses Moonshine's Transcriber with manual audio feeding from sounddevice.
Provides an async interface for consuming completed speech lines.

The Transcriber (not MicTranscriber) is used deliberately so we control
the audio capture ourselves — enabling mute/unmute for turn-taking and
future sharing of the audio stream with other components.
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np
import sounddevice as sd
from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
)

logger = logging.getLogger("avatar.asr")

# Audio capture settings
SAMPLE_RATE = 16_000  # Moonshine works at 16kHz internally
CHANNELS = 1
BLOCK_SIZE = 1600  # 100ms chunks at 16kHz
DTYPE = "float32"


class SpeechTranscriber:
    """Live speech transcriber using Moonshine Voice.

    Captures audio from the microphone via sounddevice, feeds it to
    Moonshine's Transcriber, and exposes completed speech lines as
    an async queue.

    Args:
        language: Language code (e.g., "en").
        update_interval: How often Moonshine updates transcription (seconds).
        device: sounddevice input device index or name (None = default).
    """

    def __init__(
        self,
        language: str = "en",
        update_interval: float = 0.5,
        device: int | str | None = None,
    ) -> None:
        """Initialize the speech transcriber.

        Args:
            language: Language code (e.g., "en").
            update_interval: How often Moonshine updates transcription (seconds).
            device: sounddevice input device index or name (None = default).
        """
        self._language = language
        self._update_interval = update_interval
        self._device = device
        self._loop: asyncio.AbstractEventLoop | None = None
        self._transcriber: Transcriber | None = None
        self._sd_stream: sd.InputStream | None = None
        self._listening = False

        # Async queue for completed lines (thread-safe via call_soon_threadsafe)
        self._line_queue: asyncio.Queue[str] = asyncio.Queue()
        # Async event for line-started (useful for barge-in detection)
        self._speech_started_event: asyncio.Event = asyncio.Event()
        # Speech duration tracking (for barge-in pre-filters)
        self._speech_start_time: float = 0.0
        self._last_speech_duration: float = 0.0
        # Raw audio ring buffer (for Smart Turn analysis)
        self._audio_buffer: list[np.ndarray] = []
        self._max_buffer_chunks = int(10 * SAMPLE_RATE / BLOCK_SIZE)  # 10 seconds

    async def initialize(self) -> None:
        """Download model and create transcriber.

        Must be called before start(). Downloads the model on first run
        (~130MB for English Small Streaming).
        """
        self._loop = asyncio.get_running_loop()

        logger.info("Downloading Moonshine %s model...", self._language)
        model_path, model_arch = await self._loop.run_in_executor(
            None, get_model_for_language, self._language,
        )
        logger.info("Model ready at %s (arch=%s)", model_path, model_arch)

        options = {
            "identify_speakers": "false",
            "return_audio_data": "false",
        }

        self._transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=self._update_interval,
            options=options,
        )

        # Register event listener
        listener = _TranscriberListener(self)
        self._transcriber.add_listener(listener)

        logger.info("Moonshine transcriber initialized (language=%s)", self._language)

    def start(self) -> None:
        """Start capturing audio and transcribing.

        Creates the sounddevice InputStream and starts the Moonshine session.
        """
        if not self._transcriber:
            raise RuntimeError("Call initialize() first")

        self._transcriber.start()
        self._listening = True

        self._sd_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            device=self._device,
            callback=self._audio_callback,
        )
        self._sd_stream.start()

        logger.info(
            "Microphone capture started (rate=%d, block=%d, device=%s)",
            SAMPLE_RATE, BLOCK_SIZE, self._device or "default",
        )

    def stop(self) -> None:
        """Stop capturing audio and end the transcription session."""
        self._listening = False

        if self._sd_stream is not None:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None

        if self._transcriber is not None:
            self._transcriber.stop()

        logger.info("Microphone capture stopped")

    def mute(self) -> None:
        """Temporarily stop feeding audio to transcriber (e.g., during playback).

        The sounddevice stream continues running to avoid restart overhead.
        Audio is simply not forwarded to the transcriber.
        """
        self._listening = False

    def unmute(self) -> None:
        """Resume feeding audio to transcriber."""
        self._listening = True
        self._speech_started_event.clear()

    @property
    def is_listening(self) -> bool:
        """Whether audio is being fed to the transcriber."""
        return self._listening

    async def get_next_line(self) -> str:
        """Wait for and return the next completed speech line.

        Returns:
            Transcribed text of the completed speech line.
        """
        return await self._line_queue.get()

    async def wait_for_speech_start(self) -> None:
        """Wait until speech is detected (line started).

        Useful for barge-in detection while audio is playing.
        """
        self._speech_started_event.clear()
        await self._speech_started_event.wait()

    @property
    def last_speech_duration(self) -> float:
        """Duration of the most recently completed speech segment (seconds)."""
        return self._last_speech_duration

    def get_audio_buffer(self) -> np.ndarray:
        """Get buffered raw audio for Smart Turn analysis.

        Returns:
            Concatenated float32 mono PCM from the ring buffer.
            May be up to 10 seconds of audio.
        """
        if not self._audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._audio_buffer)

    def clear_audio_buffer(self) -> None:
        """Clear the raw audio ring buffer."""
        self._audio_buffer.clear()

    def clear_queue(self) -> None:
        """Flush all pending speech lines from the queue.

        Useful after barge-in handling to discard stale transcriptions
        (e.g., echo from speaker picked up by mic).
        """
        discarded = 0
        while not self._line_queue.empty():
            try:
                self._line_queue.get_nowait()
                discarded += 1
            except asyncio.QueueEmpty:
                break
        if discarded:
            logger.info("Cleared %d stale line(s) from queue", discarded)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — feeds audio to Moonshine transcriber.

        Called from the audio thread. add_audio() is designed to buffer
        data quickly and return, so this is safe to call from the callback.
        Also buffers raw audio for Smart Turn analysis.
        """
        if status:
            logger.warning("Audio input status: %s", status)

        if not self._listening or self._transcriber is None:
            return

        # indata shape: (frames, channels), dtype float32
        # Moonshine expects a list of floats, mono PCM [-1.0, 1.0]
        mono = indata[:, 0]
        audio_chunk = mono.tolist()
        self._transcriber.add_audio(audio_chunk, SAMPLE_RATE)

        # Buffer raw audio for Smart Turn analysis (ring buffer)
        self._audio_buffer.append(mono.copy())
        if len(self._audio_buffer) > self._max_buffer_chunks:
            self._audio_buffer = self._audio_buffer[-self._max_buffer_chunks:]

    def _on_line_started(self, text: str) -> None:
        """Called from Moonshine thread when a new speech segment starts."""
        self._speech_start_time = time.monotonic()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._speech_started_event.set)

    def _on_line_text_changed(self, text: str) -> None:
        """Called from Moonshine thread when line text is updated."""
        # Could be used for real-time interim display in the future
        pass

    def _on_line_completed(self, text: str) -> None:
        """Called from Moonshine thread when a speech segment is completed."""
        if self._speech_start_time > 0:
            self._last_speech_duration = time.monotonic() - self._speech_start_time
        text = text.strip()
        if not text:
            return
        logger.info("Speech line completed: '%s' (%.2fs)", text, self._last_speech_duration)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._line_queue.put_nowait, text)

    def close(self) -> None:
        """Release all resources."""
        self.stop()
        if self._transcriber is not None:
            self._transcriber.remove_all_listeners()
            self._transcriber = None


class _TranscriberListener(TranscriptEventListener):
    """Bridge between Moonshine events and SpeechTranscriber.

    Translates Moonshine's callback-based events into the
    SpeechTranscriber's async-friendly methods.
    """

    def __init__(self, parent: SpeechTranscriber) -> None:
        self._parent = parent

    def on_line_started(self, event: Any) -> None:
        """New speech segment detected."""
        self._parent._on_line_started(event.line.text)

    def on_line_text_changed(self, event: Any) -> None:
        """Interim transcription text updated."""
        self._parent._on_line_text_changed(event.line.text)

    def on_line_completed(self, event: Any) -> None:
        """Speech segment completed — final text available."""
        self._parent._on_line_completed(event.line.text)
