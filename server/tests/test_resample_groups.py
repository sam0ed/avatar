"""Resample-group exactness for every registered engine rate."""

import math

import pytest

from src.tts.engine import ENGINES

WHISPER_SAMPLE_RATE = 16000


def expected_group(rate: int) -> int:
    return rate // math.gcd(rate, WHISPER_SAMPLE_RATE)


def test_every_registered_rate_resamples_exactly() -> None:
    for spec in ENGINES.values():
        group = expected_group(spec.sample_rate)
        out_samples = group * WHISPER_SAMPLE_RATE / spec.sample_rate
        assert out_samples == int(out_samples), spec.name
        assert group <= spec.sample_rate


def test_known_group_sizes() -> None:
    assert expected_group(44100) == 441
    assert expected_group(24000) == 3


def test_musetalk_audio_derivation_matches() -> None:
    pytest.importorskip("torch", reason="face deps not installed in the server venv")
    import sys
    from pathlib import Path

    face_dir = Path(__file__).resolve().parent.parent / "src" / "face"
    sys.path.insert(0, str(face_dir))
    import musetalk_audio

    for spec in ENGINES.values():
        assert musetalk_audio.resample_group(spec.sample_rate) == expected_group(
            spec.sample_rate
        )
