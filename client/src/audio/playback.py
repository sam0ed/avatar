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

    cancel() only sets a flag + wakes the queue reader.  The
    sounddevice stream is only touched inside play_queue's own
    control flow, so there is never a concurrent abort/write race.
    """

    def __init__(self) -> None:
        self._stream: sd.RawOutputStream | None = None
        self._cancelled = False
        self._playing = False
        # Used to wake a blocked queue.get() when cancel() is called.
        self._cancel_event: asyncio.Event = asyncio.Event()

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
        self._cancel_event.clear()
        self._playing = True
        chunk_count = 0
        loop = asyncio.get_event_loop()

        try:
            while not self._cancelled:
                # Wait for audio data OR cancel — whichever comes first.
                get_task = asyncio.ensure_future(queue.get())
                cancel_wait = asyncio.ensure_future(self._cancel_event.wait())
                done, _ = await asyncio.wait(
                    [get_task, cancel_wait],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if cancel_wait in done:
                    get_task.cancel()
                    break

                cancel_wait.cancel()
                audio_data = get_task.result()

                if audio_data is None:
                    break  # Normal end-of-stream sentinel

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

                    # write() blocks until the device buffer has room.
                    await loop.run_in_executor(None, self._stream.write, pcm)
                    if self._cancelled:
                        break  # Cancel arrived while write() was blocking
                    chunk_count += 1
                except sd.PortAudioError:
                    break  # Stream was invalidated — expected during cancel
                except Exception as e:
                    logger.warning("Audio playback error: %s", e)
        finally:
            self._close_stream()
            self._playing = False

        return chunk_count

    def cancel(self) -> None:
        """Cancel current playback immediately (for barge-in).

        Safe to call from any coroutine.  Only sets a flag and wakes the
        queue reader — the stream is closed inside play_queue's finally
        block so there is never a concurrent abort/write race.
        """
        self._cancelled = True
        self._cancel_event.set()
        logger.debug("Audio playback cancelled")

    def _close_stream(self) -> None:
        """Close the output stream — abort if cancelled, drain if normal."""
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
