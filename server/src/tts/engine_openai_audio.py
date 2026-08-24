"""OpenAI /v1/audio/speech engine adapter (Higgs via SGLang-Omni); yields raw PCM."""

import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

logger = logging.getLogger("avatar.tts.openai_audio")

REFERENCES_DIR = Path(os.environ.get("REFERENCES_DIR", "/app/references"))

CHANNELS = 1
SAMPLE_WIDTH = 2
MIN_CHUNK_SECONDS = 0.1
MAX_REFERENCES = int(os.environ.get("HIGGS_MAX_REFERENCES", "1"))

SYNTHESIS_PARAMS = {
    "voice": "default",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024,
}


def load_reference_pairs(ref_dir: Path) -> list[dict[str, str]]:
    """(audio_path, text) pairs from a reference folder; a missing .lab raises."""
    if not ref_dir.is_dir():
        raise ValueError(f"Reference folder not found: {ref_dir}")
    pairs = []
    for wav_path in sorted(ref_dir.glob("*.wav")):
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            raise ValueError(
                f"Reference {wav_path.name} has no transcript ({lab_path.name}); "
                "cloning without transcripts silently degrades — refusing"
            )
        pairs.append(
            {"audio_path": str(wav_path), "text": lab_path.read_text(encoding="utf-8").strip()}
        )
    if not pairs:
        raise ValueError(f"Reference folder {ref_dir} contains no .wav files")
    return pairs


class OpenAIAudioEngine:
    """Adapter for OpenAI-compatible TTS servers; cloning via references[] file paths."""

    def __init__(
        self,
        base_url: str,
        sample_rate: int,
        model: str,
        references_dir: Path = REFERENCES_DIR,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._sample_rate = sample_rate
        self._model = model
        self._references_dir = references_dir
        self._reference_id: str | None = None
        self._references: list[dict[str, str]] = []
        self._min_chunk_bytes = max(
            int(sample_rate * CHANNELS * SAMPLE_WIDTH * MIN_CHUNK_SECONDS), 1600
        )
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def voice_enabled(self) -> bool:
        return self._reference_id is not None

    @property
    def reference_id(self) -> str | None:
        return self._reference_id

    def set_reference_id(self, ref_id: str) -> None:
        pairs = load_reference_pairs(self._references_dir / ref_id)
        # Higgs is trained for a single short reference (official guidance: one
        # 3-30s clean sample); measured on 2x4090: 5 references trigger runaway
        # generation on ~25% of closing-phrase sentences, 1 reference on 0/55.
        self._references = pairs[:MAX_REFERENCES]
        self._reference_id = ref_id
        logger.info(
            "Voice cloning enabled: ref_id='%s', using %d of %d reference pairs",
            ref_id, len(self._references), len(pairs),
        )

    def clear_reference_id(self) -> None:
        self._reference_id = None
        self._references = []
        logger.info("Voice cloning disabled (reference_id cleared)")

    def _payload(self, text: str, stream: bool) -> dict:
        payload = {
            "model": self._model,
            "input": text,
            "stream": stream,
            "response_format": "pcm",
            **SYNTHESIS_PARAMS,
        }
        if self._references:
            payload["references"] = self._references
        return payload

    async def health_check(self) -> bool:
        """True if the server answers /health with 200."""
        try:
            resp = await self._http.get(f"{self.base_url}/health", timeout=10.0)
            return resp.status_code == 200
        except Exception as e:
            logger.error("TTS health check failed: %s", e)
            return False

    async def warmup(self) -> bool:
        """One short synthesis on the active (cloning) path."""
        try:
            started = time.monotonic()
            total = 0
            async for chunk in self.synthesize_streaming("Hello world warmup test."):
                total += len(chunk)
            ok = total > 0
            logger.log(
                logging.INFO if ok else logging.WARNING,
                "TTS warmup %s: %d PCM bytes in %.1fs (ref=%s)",
                "OK" if ok else "produced no audio",
                total, time.monotonic() - started, self._reference_id or "default",
            )
            return ok
        except Exception as e:
            logger.error("TTS warmup error: %s", e)
            return False

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM (mono int16 LE at sample_rate) in chunks of at least min_chunk_bytes."""
        try:
            started = time.monotonic()
            async with self._http.stream(
                "POST",
                f"{self.base_url}/v1/audio/speech",
                json=self._payload(text, stream=True),
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(
                        "TTS streaming failed (%d): %s", resp.status_code, error_body[:500]
                    )
                    return

                buffer = b""
                total = 0
                first_byte: float | None = None
                async for raw_chunk in resp.aiter_bytes():
                    if first_byte is None:
                        first_byte = time.monotonic()
                    buffer += raw_chunk
                    while len(buffer) >= self._min_chunk_bytes:
                        chunk, buffer = (
                            buffer[: self._min_chunk_bytes],
                            buffer[self._min_chunk_bytes :],
                        )
                        total += len(chunk)
                        yield chunk

                if buffer:
                    total += len(buffer)
                    yield buffer

                logger.info(
                    "TTS streamed %d PCM bytes for '%s' (%d chars) — first byte %.2fs, total %.2fs",
                    total,
                    text[:50],
                    len(text),
                    (first_byte - started) if first_byte else -1,
                    time.monotonic() - started,
                )
        except Exception as e:
            logger.error("TTS streaming request failed: %s", e)
