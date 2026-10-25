import asyncio
import io
import logging
import math
import time
import wave

import AVFoundation as AV
import numpy as np

from audio.play_timeline import PlayTimeline
from audio.voice_processing import VoiceProcessingIO

logger = logging.getLogger("avatar.audio")

PCM_SAMPLE_WIDTH = 2
PCM_FULL_SCALE = 32768.0
SCHEDULE_LEAD_SECONDS = 0.03
DRAIN_POLL_SECONDS = 0.02
FADE_SECONDS = 0.02
FADE_STEPS = 4
FULL_VOLUME = 1.0
OUTPUT_LATENCY_SECONDS = 0.06
NO_OPTIONS = 0
ENGINE_STOPPED = "Apple voice-processing engine stopped during playback (audio device changed?); restart the client"

_WOKEN = object()


class VoiceProcessingPlayer:
    def __init__(self, voice_io: VoiceProcessingIO) -> None:
        self._io = voice_io
        self._timeline = PlayTimeline()
        self._rate = 0.0
        self._cancelled = False
        self._playing = False
        self._paused = False
        self._frozen_seconds: float | None = None
        self.first_write_at: float | None = None
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._resume_event = asyncio.Event()

    @property
    def output_latency(self) -> float:
        return OUTPUT_LATENCY_SECONDS

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def played_seconds(self) -> float | None:
        if not self._playing:
            return None
        if self._frozen_seconds is not None:
            return self._frozen_seconds
        rate = self._rate
        if not rate:
            return 0.0
        render = self._render_position()
        heard = -math.inf if render is None else self._audible(render, rate)
        return self._timeline.content_at(heard) / rate

    async def play_queue(self, queue: asyncio.Queue[bytes | None]) -> int:
        self._begin_turn()
        try:
            return await self._play_turn(queue)
        finally:
            self._end_turn()

    def cancel(self) -> None:
        self._cancelled = True
        self._cancel_event.set()
        self._resume_event.set()
        logger.debug("Audio playback cancelled")

    def pause(self) -> None:
        if not self._playing or self._paused or self._cancelled:
            return
        self._paused = True
        self._resume_event.clear()
        self._pause_event.set()
        logger.debug("Audio playback paused")

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        self._resume_event.set()
        logger.debug("Audio playback resumed")

    def _begin_turn(self) -> None:
        self._cancelled = False
        self._paused = False
        self._frozen_seconds = None
        self.first_write_at = None
        self._rate = 0.0
        self._timeline.reset()
        self._cancel_event.clear()
        self._pause_event.clear()
        self._resume_event.clear()
        self._playing = True

    def _end_turn(self) -> None:
        self._playing = False
        self._io.player.stop()
        self._io.player.setVolume_(FULL_VOLUME)
        self._timeline.reset()

    async def _play_turn(self, queue: asyncio.Queue[bytes | None]) -> int:
        chunk_count = 0
        end_of_stream = False
        while not self._cancelled:
            self._pause_event.clear()
            if self._paused:
                await self._hold_paused()
            elif end_of_stream and self._drained():
                break
            else:
                item = await self._next_item(None if end_of_stream else queue)
                end_of_stream = end_of_stream or item is None
                if item is not None and item is not _WOKEN and self._schedule_wav(item):
                    chunk_count += 1
        if self._cancelled:
            await self._fade_out()
        return chunk_count

    async def _next_item(self, queue: asyncio.Queue[bytes | None] | None) -> object:
        waiting = [asyncio.ensure_future(self._cancel_event.wait()), asyncio.ensure_future(self._pause_event.wait())]
        getter = None if queue is None else asyncio.ensure_future(queue.get())
        if getter is not None:
            waiting.append(getter)
        timeout = DRAIN_POLL_SECONDS if getter is None else None
        try:
            done, _ = await asyncio.wait(waiting, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waiting:
                task.cancel()
        return getter.result() if getter is not None and getter in done else _WOKEN

    async def _hold_paused(self) -> None:
        await self._fade_out()
        render = self._render_position()
        paused_at = 0 if render is None else render[0]
        self._frozen_seconds = self._timeline.content_at(paused_at) / self._rate if self._rate else 0.0
        self._io.player.pause()
        await self._resume_event.wait()
        if not self._cancelled:
            self._skip_rest_of_chunk(paused_at)

    def _skip_rest_of_chunk(self, paused_at: int) -> None:
        self._io.player.setVolume_(FULL_VOLUME)
        if self._rate:
            upcoming = self._timeline.not_started(paused_at)
            self._io.player.stop()
            self._timeline.reset(round(self._timeline.content_at(paused_at)))
            self._io.player.play()
            for segment in upcoming:
                self._schedule(segment.buffer, segment.frames)
        self._frozen_seconds = None

    async def _fade_out(self) -> None:
        for step in range(1, FADE_STEPS + 1):
            self._io.player.setVolume_(FULL_VOLUME * (1 - step / FADE_STEPS))
            await asyncio.sleep(FADE_SECONDS / FADE_STEPS)

    def _drained(self) -> bool:
        end = self._timeline.end()
        if end is None:
            return True
        if not self._io.is_running:
            raise RuntimeError(ENGINE_STOPPED)
        render = self._render_position()
        return render is not None and self._audible(render, self._rate) >= end

    def _schedule_wav(self, data: bytes) -> bool:
        decoded = self._decode_wav(data)
        if decoded is None:
            return False
        rate, samples = decoded
        self._prepare_output(rate, samples.shape[1])
        if len(samples):
            self._schedule(self._pcm_buffer(samples), len(samples))
        return True

    def _decode_wav(self, data: bytes) -> tuple[int, np.ndarray] | None:
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                pcm = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError) as error:
            logger.warning("Audio playback error: %s", error)
            return None
        if width != PCM_SAMPLE_WIDTH:
            raise ValueError(f"Expected 16-bit PCM audio, got a sample width of {width} bytes")
        return rate, np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)

    def _prepare_output(self, rate: int, channels: int) -> None:
        if self._rate:
            if rate != self._rate or channels != self._io.player_format.channelCount():
                raise ValueError(f"Audio format changed within a turn: {rate} Hz, {channels} ch")
            return
        self._io.connect_player(rate, channels)
        self._rate = float(rate)
        self._io.player.setVolume_(FULL_VOLUME)
        self._io.player.play()

    def _pcm_buffer(self, samples: np.ndarray) -> AV.AVAudioPCMBuffer:
        frames, channels = samples.shape
        buffer = AV.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(self._io.player_format, frames)
        buffer.setFrameLength_(frames)
        data = buffer.floatChannelData()
        for channel in range(channels):
            target = np.frombuffer(data[channel].as_buffer(frames), dtype=np.float32)
            target[:] = samples[:, channel] / PCM_FULL_SCALE
        return buffer

    def _schedule(self, buffer: AV.AVAudioPCMBuffer, frames: int) -> None:
        render = self._render_position()
        rendered, ahead = (0, 0.0) if render is None else render
        earliest = rendered + math.ceil(SCHEDULE_LEAD_SECONDS * self._rate)
        start = self._timeline.place(buffer, frames, earliest)
        when = AV.AVAudioTime.timeWithSampleTime_atRate_(start, self._rate)
        self._io.player.scheduleBuffer_atTime_options_completionHandler_(buffer, when, NO_OPTIONS, None)
        if self.first_write_at is None:
            self.first_write_at = time.monotonic() + ahead + (start - rendered) / self._rate

    def _render_position(self) -> tuple[int, float] | None:
        player = self._io.player
        last = player.lastRenderTime()
        if last is None or not last.isHostTimeValid():
            return None
        player_time = player.playerTimeForNodeTime_(last)
        if player_time is None or not player_time.isSampleTimeValid():
            return None
        ahead = AV.AVAudioTime.secondsForHostTime_(last.hostTime()) - time.monotonic()
        return int(player_time.sampleTime()), ahead

    def _audible(self, render: tuple[int, float], rate: float) -> float:
        rendered, ahead = render
        return rendered - (ahead + OUTPUT_LATENCY_SECONDS) * rate
