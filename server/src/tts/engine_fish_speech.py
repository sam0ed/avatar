"""Fish Speech (OpenAudio S1-mini) engine: msgpack /v1/tts, yields raw 44.1 kHz PCM."""

import logging
import os
import time
from collections.abc import AsyncIterator

import httpx
import ormsgpack

logger = logging.getLogger("avatar.tts.fish")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8080")

SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2
MIN_CHUNK_SECONDS = 0.1
MIN_CHUNK_BYTES = max(int(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * MIN_CHUNK_SECONDS), 3200)

SYNTHESIS_PARAMS = {
    "format": "wav",
    "normalize": True,
    "max_new_tokens": 1024,
    "top_p": 0.8,
    "temperature": 0.8,
    "repetition_penalty": 1.1,
    "chunk_length": 200,
}


class FishSpeechEngine:
    """Async client for the fish-speech server; voice cloning via server-side reference folders."""

    def __init__(self, base_url: str = TTS_BASE_URL, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._reference_id: str | None = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def voice_enabled(self) -> bool:
        return self._reference_id is not None

    @property
    def reference_id(self) -> str | None:
        return self._reference_id

    def set_reference_id(self, ref_id: str) -> None:
        self._reference_id = ref_id
        logger.info("Voice cloning enabled with reference_id='%s'", ref_id)

    def clear_reference_id(self) -> None:
        self._reference_id = None
        logger.info("Voice cloning disabled (reference_id cleared)")

    def _payload(self, text: str, streaming: bool) -> dict:
        payload = {"text": text, "streaming": streaming, **SYNTHESIS_PARAMS}
        if self._reference_id:
            payload["reference_id"] = self._reference_id
            payload["use_memory_cache"] = "on"
        return payload

    async def health_check(self) -> bool:
        """True if the fish server answers its health endpoint."""
        try:
            resp = await self._http.get(f"{self.base_url}/v1/health", timeout=10.0)
            return resp.json().get("status") == "ok"
        except Exception as e:
            logger.error("TTS health check failed: %s", e)
            return False

    async def warmup(self) -> bool:
        """One synthesis to trigger torch.compile tracing on the active (cloning) path."""
        body = ormsgpack.packb(self._payload("Hello world warmup test.", streaming=False))
        try:
            started = time.monotonic()
            resp = await self._http.post(
                f"{self.base_url}/v1/tts",
                content=body,
                headers={"Content-Type": "application/msgpack"},
                timeout=120.0,
            )
            if resp.status_code == 200:
                logger.info(
                    "TTS warmup OK: %d bytes in %.1fs (ref=%s)",
                    len(resp.content),
                    time.monotonic() - started,
                    self._reference_id or "default",
                )
                return True
            logger.warning("TTS warmup failed (%d): %s", resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            logger.error("TTS warmup error: %s", e)
            return False

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM (44.1 kHz mono int16 LE) in chunks of at least MIN_CHUNK_BYTES."""
        body = ormsgpack.packb(self._payload(text, streaming=True))
        try:
            started = time.monotonic()
            async with self._http.stream(
                "POST",
                f"{self.base_url}/v1/tts",
                content=body,
                headers={"Content-Type": "application/msgpack"},
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
                    while len(buffer) >= MIN_CHUNK_BYTES:
                        chunk, buffer = buffer[:MIN_CHUNK_BYTES], buffer[MIN_CHUNK_BYTES:]
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
