"""OpenAI-audio adapter: reference loading rules and exact request shape."""

import json
from pathlib import Path

import httpx
import pytest

from src.tts.engine import ENGINES
from src.tts.engine_openai_audio import OpenAIAudioEngine, load_reference_pairs


def make_engine(tmp_path: Path) -> OpenAIAudioEngine:
    return OpenAIAudioEngine(
        base_url="http://testserver",
        sample_rate=24000,
        model="bosonai/higgs-tts-3-4b",
        references_dir=tmp_path,
    )


def write_pair(ref_dir: Path, stem: str, text: str = "привіт") -> None:
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / f"{stem}.wav").write_bytes(b"RIFFfake")
    (ref_dir / f"{stem}.lab").write_text(text, encoding="utf-8")


def test_missing_lab_raises(tmp_path: Path) -> None:
    ref_dir = tmp_path / "voice"
    ref_dir.mkdir()
    (ref_dir / "a.wav").write_bytes(b"RIFFfake")
    with pytest.raises(ValueError, match="no transcript"):
        load_reference_pairs(ref_dir)


def test_empty_folder_raises(tmp_path: Path) -> None:
    ref_dir = tmp_path / "voice"
    ref_dir.mkdir()
    with pytest.raises(ValueError, match="no .wav files"):
        load_reference_pairs(ref_dir)


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_reference_pairs(tmp_path / "nope")


def test_reference_pairs_load_in_order(tmp_path: Path) -> None:
    ref_dir = tmp_path / "voice"
    write_pair(ref_dir, "b", "друге")
    write_pair(ref_dir, "a", "перше")
    pairs = load_reference_pairs(ref_dir)
    assert [p["text"] for p in pairs] == ["перше", "друге"]
    assert pairs[0]["audio_path"].endswith("a.wav")


@pytest.mark.asyncio
async def test_streaming_request_shape_and_chunking(tmp_path: Path) -> None:
    write_pair(tmp_path / "my-voice", "sample_01")
    engine = make_engine(tmp_path)
    engine.set_reference_id("my-voice")

    captured: dict = {}
    pcm = b"\x01\x02" * 6000

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, content=pcm)

    engine._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    chunks = [chunk async for chunk in engine.synthesize_streaming("Привіт, світе!")]

    assert captured["url"] == "http://testserver/v1/audio/speech"
    body = captured["json"]
    assert body["model"] == "bosonai/higgs-tts-3-4b"
    assert body["input"] == "Привіт, світе!"
    assert body["stream"] is True
    assert body["response_format"] == "pcm"
    assert body["voice"] == "default"
    assert len(body["references"]) == 1
    assert body["references"][0]["audio_path"].endswith("sample_01.wav")
    assert body["references"][0]["text"] == "привіт"

    assert b"".join(chunks) == pcm
    assert all(len(c) == engine._min_chunk_bytes for c in chunks[:-1])


@pytest.mark.asyncio
async def test_no_references_field_without_cloning(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, content=b"\x00\x00" * 2400)

    engine._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    [_ async for _ in engine.synthesize_streaming("test")]
    assert "references" not in captured["json"]


def test_higgs_registry_entry() -> None:
    spec = ENGINES["higgs"]
    assert spec.sample_rate == 24000
    assert spec.base_url == "http://localhost:8080"
