"""Audio playback with cancel and pause/resume support for barge-in.

Plays WAV audio chunks from an async queue using sounddevice.
Supports cancellation for interrupting playback when the user
starts speaking again, and pause/resume for tentative interruptions.
"""

import asyncio
import io
import logging
import wave

import numpy as np
import sounddevice as sd

logger = logging.getLogger("avatar.audio")

# Number of samples per mini-frame for responsive pause/resume.
# At 24 kHz (typical TTS output) this is ~43 ms per frame.
_FRAME_SAMPLES = 1024

# Fade-out length in samples when pausing (~10 ms at 24 kHz).
# Ramps the last frame to zero so the mid-word cutoff is a
# smooth fade rather than an abrupt clip.
_FADE_SAMPLES = 256


class AudioPlayer:
    """Async audio player with cancel and pause/resume support.

    Plays WAV audio chunks from a queue through speakers.
    Supports:
      - cancel(): hard stop — abort playback immediately (irreversible).
      - pause(): suspend playback; audio data keeps queuing.
      - resume(): continue playback from where we paused.

    cancel() and pause() only set flags + wake the queue reader.  The
    sounddevice stream is only touched inside play_queue's own control
    flow, so there is never a concurrent abort/write race.
    """

    def __init__(self) -> None:
        self._stream: sd.RawOutputStream | None = None
        self._cancelled = False
        self._playing = False
        self._paused = False
        # Used to wake a blocked queue.get() when cancel() is called.
        self._cancel_event: asyncio.Event = asyncio.Event()
        # Used to wake a blocked pause wait when resume()/cancel() is called.
        self._resume_event: asyncio.Event = asyncio.Event()

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently being played (or paused)."""
        return self._playing

    @property
    def is_paused(self) -> bool:
        """Whether playback is currently paused."""
        return self._paused

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
        self._paused = False
        self._cancel_event.clear()
        self._resume_event.clear()
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
                        channels = wf.getnchannels()

                    # Write PCM in small frames for responsive pause/resume.
                    # Checking _paused every ~40 ms lets us stop feeding
                    # the device almost instantly.  The stream keeps running
                    # so the remaining buffer (< 50 ms) plays out naturally
                    # — no click.  On resume we continue from the exact
                    # same offset, so no audio is lost or repeated.
                    bytes_per_frame = _FRAME_SAMPLES * channels * 2  # int16
                    offset = 0

                    while offset < len(pcm):
                        if self._cancelled:
                            break

                        if self._paused:
                            # Write a short fade-out so the cutoff is
                            # smooth instead of an abrupt clip.
                            fade_bytes = min(
                                _FADE_SAMPLES * channels * 2,
                                len(pcm) - offset,
                            )
                            if fade_bytes > 0:
                                fade_pcm = np.frombuffer(
                                    pcm[offset : offset + fade_bytes],
                                    dtype=np.int16,
                                ).copy()
                                ramp = np.linspace(
                                    1.0, 0.0, len(fade_pcm), dtype=np.float32,
                                )
                                fade_pcm = (fade_pcm * ramp).astype(np.int16)
                                await loop.run_in_executor(
                                    None,
                                    self._stream.write,
                                    fade_pcm.tobytes(),
                                )

                            # Wait for resume() or cancel().
                            await self._resume_event.wait()
                            if self._cancelled:
                                break
                            # Drop the rest of this sentence chunk so
                            # playback resumes at the next sentence
                            # boundary — no mid-word restart.
                            break

                        end = min(offset + bytes_per_frame, len(pcm))
                        await loop.run_in_executor(
                            None, self._stream.write, pcm[offset:end]
                        )
                        offset = end

                    if self._cancelled:
                        break
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
        # Also wake any paused wait so it exits immediately.
        self._resume_event.set()
        logger.debug("Audio playback cancelled")

    def pause(self) -> None:
        """Pause playback for tentative barge-in verification.

        Stops feeding audio frames within ~40 ms.  The device stream
        keeps running so the tiny remaining buffer fades out naturally.
        On resume(), the remainder of the current sentence is skipped
        and playback continues from the next sentence in the queue.
        Call resume() to continue or cancel() to abort.
        """
        if not self._playing or self._paused or self._cancelled:
            return
        self._paused = True
        self._resume_event.clear()
        logger.debug("Audio playback paused")

    def resume(self) -> None:
        """Resume playback after a pause (false interruption).

        Continues from where we left off — no audio is lost.
        Safe to call from any coroutine.
        """
        if not self._paused:
            return
        self._paused = False
        self._resume_event.set()
        logger.debug("Audio playback resumed")

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
