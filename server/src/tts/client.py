"""Async client for the Fish Speech TTS API.

Handles synthesis requests and voice reference management.
Mirrors the protocol from client/src/tts_test.py but async.
"""

import logging
import os

import httpx
import ormsgpack

logger = logging.getLogger("avatar.tts.client")

# TTS service URL — set by Docker Compose env or fallback
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8080")


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
