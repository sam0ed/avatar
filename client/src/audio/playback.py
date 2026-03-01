"""Audio playback with cancel support for barge-in.

Plays WAV audio chunks from an async queue using sounddevice.
Supports cancellation for interrupting playback when the user
starts speaking again.
"""

import asyncio
import io
import logging
import wave

import sounddevice as sd

logger = logging.getLogger("avatar.audio")


class AudioPlayer:
    """Async audio player with cancellation support.

    Plays WAV audio chunks from a queue through speakers.
    Supports cancel() for barge-in (stops playback immediately).
    """

    def __init__(self) -> None:
        self._stream: sd.RawOutputStream | None = None
        self._cancelled = False
        self._playing = False

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently being played."""
        return self._playing

    async def play_queue(self, queue: asyncio.Queue[bytes | None]) -> int:
        """Play audio chunks from queue until sentinel (None) or cancel.

        Each item in the queue is a complete WAV file as bytes.
        None signals end of stream.

        Args:
            queue: Queue of WAV audio bytes (None = stop).

        Returns:
            Number of audio chunks played.
        """
        self._cancelled = False
        self._playing = True
        chunk_count = 0
        loop = asyncio.get_event_loop()

        try:
            while not self._cancelled:
                try:
                    audio_data = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if audio_data is None:
                    break

                if self._cancelled:
                    break

                try:
                    with wave.open(io.BytesIO(audio_data), "rb") as wf:
                        if self._stream is None:
                            self._stream = sd.RawOutputStream(
                                samplerate=wf.getframerate(),
                                channels=wf.getnchannels(),
                                dtype="int16",
                            )
                            self._stream.start()
                        pcm = wf.readframes(wf.getnframes())

                    if not self._cancelled:
                        # write() blocks until the device buffer has room.
                        await loop.run_in_executor(None, self._stream.write, pcm)
                        chunk_count += 1
                except Exception as e:
                    logger.warning("Audio playback failed: %s", e)
        finally:
            self._drain_or_abort()
            self._playing = False

        return chunk_count

    def cancel(self) -> None:
        """Cancel current playback immediately (for barge-in).

        Aborts the output stream (discards buffered audio) and sets
        the cancelled flag so the play_queue loop exits.
        """
        self._cancelled = True
        self._abort_stream()
        logger.debug("Audio playback cancelled")

    def _drain_or_abort(self) -> None:
        """Stop the stream — drain if normal finish, abort if cancelled."""
        if self._stream is None:
            return
        try:
            if self._cancelled:
                self._stream.abort()
            else:
                self._stream.stop()  # waits for remaining buffers to play
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _abort_stream(self) -> None:
        """Immediately abort the output stream (for cancel)."""
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
