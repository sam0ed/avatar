import asyncio
import logging
import time
from collections.abc import Callable

import AVFoundation as AV
import numpy as np

from asr.transcriber import BLOCK_SIZE, SAMPLE_RATE
from audio.capture_health import CaptureHealth
from audio.stream_resampler import StreamResampler

logger = logging.getLogger("avatar.audio")

INPUT_BUS = 0
MONO = 1
TAP_SECONDS = 0.1
SELF_TEST_TIMEOUT_SECONDS = 5.0
SELF_TEST_POLL_SECONDS = 0.1
INITIAL_PLAYER_RATE = 48_000
FAILURE_ADVICE = (
    "Check the Microphone permission of the app that launched the client, or run with --no-aec "
    "to use plain capture and playback without echo cancellation."
)


class VoiceProcessingError(RuntimeError):
    pass


def float_format(rate: float, channels: int) -> AV.AVAudioFormat:
    return AV.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
        AV.AVAudioPCMFormatFloat32, float(rate), channels, False
    )


def _minimal_ducking() -> AV.AVAudioVoiceProcessingOtherAudioDuckingConfiguration:
    return AV.AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
        enableAdvancedDucking=False, duckingLevel=AV.AVAudioVoiceProcessingOtherAudioDuckingLevelMin
    )


class VoiceProcessingIO:
    def __init__(self, push_audio: Callable[[np.ndarray], None]) -> None:
        self._push_audio = push_audio
        self._engine = AV.AVAudioEngine.alloc().init()
        self._player = AV.AVAudioPlayerNode.alloc().init()
        self._input = None
        self._player_format: AV.AVAudioFormat | None = None
        self._capture_rate = 0.0
        self._resampler: StreamResampler | None = None
        self._pending = np.empty(0, dtype=np.float32)
        self._health = CaptureHealth()

    @property
    def player(self) -> AV.AVAudioPlayerNode:
        return self._player

    @property
    def player_format(self) -> AV.AVAudioFormat:
        return self._player_format

    @property
    def is_running(self) -> bool:
        return bool(self._engine.isRunning())

    async def start(self) -> None:
        self._build_graph()
        self._install_tap()
        started, error = self._engine.startAndReturnError_(None)
        if not started:
            raise VoiceProcessingError(f"Apple voice-processing engine failed to start: {error}")
        self._require_agc_off()
        await self._verify_capture()

    def stop(self) -> None:
        self._player.stop()
        self._engine.stop()
        if self._input is not None:
            self._input.removeTapOnBus_(INPUT_BUS)
        logger.info("Apple voice-processing audio stopped")

    def connect_player(self, rate: float, channels: int) -> None:
        target = float_format(rate, channels)
        if self._player_format is not None and self._player_format.isEqual_(target):
            return
        self._player.stop()
        self._engine.connect_to_format_(self._player, self._engine.mainMixerNode(), target)
        self._player_format = target
        logger.info("Voice-processing playback format: %d Hz, %d ch", rate, channels)

    def _build_graph(self) -> None:
        self._engine.outputNode()
        self._engine.mainMixerNode()
        self._input = self._engine.inputNode()
        enabled, error = self._input.setVoiceProcessingEnabled_error_(True, None)
        if not enabled:
            raise VoiceProcessingError(f"Enabling Apple voice processing failed: {error}")
        self._input.setVoiceProcessingAGCEnabled_(False)
        self._require_agc_off()
        self._input.setVoiceProcessingOtherAudioDuckingConfiguration_(_minimal_ducking())
        self._engine.attachNode_(self._player)
        self.connect_player(INITIAL_PLAYER_RATE, MONO)

    def _require_agc_off(self) -> None:
        if self._input.isVoiceProcessingAGCEnabled():
            raise VoiceProcessingError("Voice-processing AGC is still on after turning it off")

    def _install_tap(self) -> None:
        hardware = self._input.outputFormatForBus_(INPUT_BUS)
        self._capture_rate = float(hardware.sampleRate())
        if hardware.channelCount() == 0 or self._capture_rate <= 0:
            raise VoiceProcessingError(f"Voice-processing input has no usable format: {hardware}")
        self._resampler = StreamResampler(round(self._capture_rate), SAMPLE_RATE)
        frames = round(self._capture_rate * TAP_SECONDS)
        tap_format = float_format(self._capture_rate, MONO)
        self._input.installTapOnBus_bufferSize_format_block_(INPUT_BUS, frames, tap_format, self._on_tap)
        logger.info(
            "Voice-processing capture: %d ch at %d Hz, tapped as mono, resampled to %d Hz",
            hardware.channelCount(),
            self._capture_rate,
            SAMPLE_RATE,
        )

    def _on_tap(self, buffer: AV.AVAudioPCMBuffer, when: AV.AVAudioTime) -> None:
        try:
            frames = int(buffer.frameLength())
            block = np.frombuffer(buffer.floatChannelData()[0].as_buffer(frames), dtype=np.float32).copy()
            self._health.observe(block)
            self._forward(self._resampler.process(block))
        except Exception:
            logger.exception("Voice-processing capture block failed")

    def _forward(self, samples: np.ndarray) -> None:
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= BLOCK_SIZE:
            self._push_audio(self._pending[:BLOCK_SIZE])
            self._pending = self._pending[BLOCK_SIZE:]

    async def _verify_capture(self) -> None:
        deadline = time.monotonic() + SELF_TEST_TIMEOUT_SECONDS
        while self._health.problem() is not None and time.monotonic() < deadline:
            await asyncio.sleep(SELF_TEST_POLL_SECONDS)
        report = self._health.describe(self._capture_rate)
        problem = self._health.problem()
        if problem is None:
            logger.info("Voice-processing capture self-test passed: %s, AGC off", report)
            return
        self.stop()
        raise VoiceProcessingError(
            f"Echo-cancelled capture failed its self-test: {problem} {SELF_TEST_TIMEOUT_SECONDS} s after the "
            f"engine started ({report}). {FAILURE_ADVICE}"
        )
