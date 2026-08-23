"""Contract-layer tests: registry, protocol conformance, WAV framing."""

import io
import wave

import pytest

from src.tts.engine import ENGINES, TTSEngine, create_engine, wrap_wav
from src.tts.engine_fish_speech import FishSpeechEngine


def test_unknown_engine_raises() -> None:
    with pytest.raises(ValueError, match="Unknown TTS engine 'bogus'"):
        create_engine("bogus")


def test_env_selects_unknown_engine_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_ENGINE", "bogus")
    with pytest.raises(ValueError, match="bogus"):
        create_engine()


def test_default_engine_is_fish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    engine = create_engine()
    assert isinstance(engine, FishSpeechEngine)
    assert engine.sample_rate == 44100


def test_registry_entries_are_consistent() -> None:
    for name, spec in ENGINES.items():
        assert spec.name == name
        assert spec.sample_rate > 0
        assert spec.base_url.startswith("http")


def test_every_registered_engine_satisfies_the_protocol() -> None:
    for name in ENGINES:
        assert isinstance(create_engine(name), TTSEngine)


@pytest.mark.parametrize("sample_rate", [44100, 24000])
def test_wrap_wav_parses_back_as_the_client_reads_it(sample_rate: int) -> None:
    pcm = b"\x01\x02" * 441
    with wave.open(io.BytesIO(wrap_wav(pcm, sample_rate)), "rb") as wf:
        assert wf.getframerate() == sample_rate
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.readframes(wf.getnframes()) == pcm


def test_fish_reference_id_round_trip() -> None:
    engine = FishSpeechEngine()
    assert not engine.voice_enabled
    engine.set_reference_id("my-voice")
    assert engine.voice_enabled
    assert engine.reference_id == "my-voice"
    engine.clear_reference_id()
    assert not engine.voice_enabled
    assert engine.reference_id is None
