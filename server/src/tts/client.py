"""Async client for the Fish Speech TTS API.

Handles synthesis requests and voice reference management.
Mirrors the protocol from client/src/tts_test.py but async.
"""

import logging
import os
import struct
import time
from collections.abc import AsyncIterator

import httpx
import ormsgpack

logger = logging.getLogger("avatar.tts.client")

# TTS service URL — set by Docker Compose env or fallback
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8080")


def _make_wav_header(
    data_size: int,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    """Create a minimal WAV header for PCM audio.

    Args:
        data_size: Size of the PCM data in bytes.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        sample_width: Bytes per sample (e.g. 2 for 16-bit).

    Returns:
        44-byte WAV header.
    """
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
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


class TTSClient:
    """Async client for Fish Speech TTS API.

    Sends text to the TTS service and receives synthesized WAV audio.
    Supports voice cloning via pre-uploaded reference IDs.
    """

    def __init__(
        self,
        base_url: str = TTS_BASE_URL,
        reference_id: str | None = None,
        output_format: str = "wav",
        timeout: float = 60.0,
    ) -> None:
        """Initialize TTS client.

        Args:
            base_url: Fish Speech API base URL.
            reference_id: Pre-uploaded voice reference ID for cloning.
            output_format: Audio output format (wav, mp3, pcm).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.reference_id = reference_id
        self.output_format = output_format
        self.timeout = timeout

    async def health_check(self) -> bool:
        """Check if the TTS server is healthy.

        Returns:
            True if server responds OK.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/v1/health",
                    timeout=10.0,
                )
                data = resp.json()
                return data.get("status") == "ok"
        except Exception as e:
            logger.error("TTS health check failed: %s", e)
            return False

    async def synthesize(self, text: str) -> bytes | None:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.

        Returns:
            WAV audio bytes, or None on failure.
        """
        payload: dict = {
            "text": text,
            "format": self.output_format,
            "streaming": False,
            "normalize": True,
            "max_new_tokens": 1024,
            "top_p": 0.8,
            "temperature": 0.8,
            "repetition_penalty": 1.1,
            "chunk_length": 200,
        }

        if self.reference_id:
            payload["reference_id"] = self.reference_id

        body = ormsgpack.packb(payload)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/v1/tts",
                    content=body,
                    headers={"Content-Type": "application/msgpack"},
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    logger.info(
                        "TTS synthesized %d bytes for '%s' (%d chars)",
                        len(resp.content),
                        text[:50],
                        len(text),
                    )
                    return resp.content
                else:
                    logger.error("TTS failed (%d): %s", resp.status_code, resp.text[:500])
                    return None
        except Exception as e:
            logger.error("TTS request failed: %s", e)
            return None

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text to audio with streaming.

        Yields complete WAV audio chunks as they become available from
        the TTS server. Each chunk is a self-contained WAV file that
        can be played independently.

        Args:
            text: Text to synthesize.

        Yields:
            WAV audio bytes (each chunk is a complete, playable WAV file).
        """
        payload: dict = {
            "text": text,
            "format": "wav",  # streaming only supports wav
            "streaming": True,
            "normalize": True,
            "max_new_tokens": 1024,
            "top_p": 0.8,
            "temperature": 0.8,
            "repetition_penalty": 1.1,
            "chunk_length": 200,
        }

        if self.reference_id:
            payload["reference_id"] = self.reference_id

        body = ormsgpack.packb(payload)

        try:
            t_start = time.monotonic()
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/tts",
                    content=body,
                    headers={"Content-Type": "application/msgpack"},
                    timeout=self.timeout,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        logger.error(
                            "TTS streaming failed (%d): %s",
                            resp.status_code,
                            error_body[:500],
                        )
                        return

                    # Fish Speech streaming returns raw PCM (no WAV
                    # header).  We know the format from the non-streaming
                    # endpoint: 44100 Hz, mono, 16-bit signed LE.
                    sample_rate = 44100
                    channels = 1
                    sample_width = 2  # bytes per sample
                    # ~0.2s of audio as minimum yield threshold
                    min_chunk_bytes = max(
                        sample_rate * channels * sample_width // 5,
                        3200,
                    )
                    pcm_buffer = b""
                    total_pcm = 0
                    t_first_byte: float | None = None
                    chunk_count = 0

                    async for raw_chunk in resp.aiter_bytes():
                        if t_first_byte is None:
                            t_first_byte = time.monotonic()
                        pcm_buffer += raw_chunk

                        # Yield when we have enough PCM data
                        while len(pcm_buffer) >= min_chunk_bytes:
                            chunk = pcm_buffer[:min_chunk_bytes]
                            pcm_buffer = pcm_buffer[min_chunk_bytes:]
                            wav_data = (
                                _make_wav_header(
                                    len(chunk),
                                    sample_rate,
                                    channels,
                                    sample_width,
                                )
                                + chunk
                            )
                            total_pcm += len(chunk)
                            chunk_count += 1
                            yield wav_data

                    # Flush remaining PCM data
                    if pcm_buffer:
                        wav_data = (
                            _make_wav_header(
                                len(pcm_buffer),
                                sample_rate,
                                channels,
                                sample_width,
                            )
                            + pcm_buffer
                        )
                        total_pcm += len(pcm_buffer)
                        chunk_count += 1
                        yield wav_data

                    t_end = time.monotonic()
                    t_fb = (t_first_byte - t_start) if t_first_byte else -1
                    logger.info(
                        "TTS streamed %d bytes (%d chunks) for '%s' (%d chars) "
                        "— first byte %.2fs, total %.2fs",
                        total_pcm,
                        chunk_count,
                        text[:50],
                        len(text),
                        t_fb,
                        t_end - t_start,
                    )
        except Exception as e:
            logger.error("TTS streaming request failed: %s", e)

    async def upload_reference(
        self,
        reference_id: str,
        audio_bytes: bytes,
        audio_filename: str,
        transcript: str,
    ) -> bool:
        """Upload a voice reference to the TTS server.

        Args:
            reference_id: Unique ID for this voice reference.
            audio_bytes: WAV audio file bytes.
            audio_filename: Original filename of the audio.
            transcript: Text transcript of what was said in the audio.

        Returns:
            True if upload succeeded.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/v1/references/add",
                    files={"audio": (audio_filename, audio_bytes, "audio/wav")},
                    data={"id": reference_id, "text": transcript},
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    data = ormsgpack.unpackb(resp.content)
                    logger.info("Reference '%s' uploaded: %s", reference_id, data.get("message"))
                    return True
                else:
                    logger.error(
                        "Reference upload failed (%d): %s",
                        resp.status_code,
                        resp.content[:500],
                    )
                    return False
        except Exception as e:
            logger.error("Reference upload failed: %s", e)
            return False

    async def list_references(self) -> list[str]:
        """List available voice references on the server.

        Returns:
            List of reference IDs.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/v1/references/list",
                    timeout=10.0,
                )
                data = ormsgpack.unpackb(resp.content)
                ref_ids = data.get("reference_ids", [])
                logger.info("Available TTS references: %s", ref_ids)
                return ref_ids
        except Exception as e:
            logger.error("List references failed: %s", e)
            return []
