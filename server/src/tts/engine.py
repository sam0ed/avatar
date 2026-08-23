"""TTS engine contract, registry, and the WAV framing the orchestrator applies per chunk."""

import os
import struct
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

TTS_ENGINE_ENV = "TTS_ENGINE"
DEFAULT_ENGINE = "fish"

PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2


@runtime_checkable
class TTSEngine(Protocol):
    """What the orchestrator requires of a TTS engine; streaming yields raw PCM at sample_rate."""

    @property
    def sample_rate(self) -> int: ...

    @property
    def voice_enabled(self) -> bool: ...

    @property
    def reference_id(self) -> str | None: ...

    def set_reference_id(self, ref_id: str) -> None: ...

    def clear_reference_id(self) -> None: ...

    async def health_check(self) -> bool: ...

    async def warmup(self) -> bool: ...

    def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]: ...


@dataclass(frozen=True)
class EngineSpec:
    """Registry entry: how to construct an engine and what it emits."""

    name: str
    sample_rate: int
    base_url: str
    factory: Callable[[], TTSEngine]


def _fish_factory() -> TTSEngine:
    from src.tts.engine_fish_speech import FishSpeechEngine

    return FishSpeechEngine()


ENGINES: dict[str, EngineSpec] = {
    "fish": EngineSpec(
        name="fish",
        sample_rate=44100,
        base_url="http://localhost:8080",
        factory=_fish_factory,
    ),
}


def create_engine(name: str | None = None) -> TTSEngine:
    """Build the engine named by the argument or TTS_ENGINE; unknown names raise."""
    engine_name = name or os.environ.get(TTS_ENGINE_ENV, DEFAULT_ENGINE)
    spec = ENGINES.get(engine_name)
    if spec is None:
        raise ValueError(
            f"Unknown TTS engine '{engine_name}'; known engines: {sorted(ENGINES)}"
        )
    return spec.factory()


def make_wav_header(
    data_size: int,
    sample_rate: int,
    channels: int = PCM_CHANNELS,
    sample_width: int = PCM_SAMPLE_WIDTH,
) -> bytes:
    """44-byte canonical WAV header for a PCM payload of data_size bytes."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    bits_per_sample = sample_width * 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def wrap_wav(pcm: bytes, sample_rate: int) -> bytes:
    """One self-contained WAV chunk from raw PCM, as the client's player expects."""
    return make_wav_header(len(pcm), sample_rate) + pcm
