"""Async client for the MuseTalk face animation service.

Mirrors the TTSClient pattern: persistent httpx connection pool,
stateful session management (start/feed/end), health check.
"""

import base64
import logging
import os
import time
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger("avatar.face.client")

FACE_BASE_URL = os.environ.get("FACE_BASE_URL", "http://localhost:8002")


class FaceAnimationClient:
    """Async client for the MuseTalk face animation service.

    Manages stateful streaming sessions: start a session, feed audio
    chunks to receive JPEG frames, end the session to flush final frames.
    """

    def __init__(
        self,
        base_url: str = FACE_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        """Initialize face animation client.

        Args:
            base_url: MuseTalk service base URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def health_check(self) -> bool:
        """Check if the face animation service is healthy.

        Returns:
            True if service responds OK.
        """
        try:
            resp = await self._http.get(f"{self.base_url}/health", timeout=10.0)
            data = resp.json()
            return data.get("status") == "ok"
        except Exception as e:
            logger.error("Face health check failed: %s", e)
            return False

    async def prepare_avatar(
        self,
        video_bytes: bytes,
        avatar_id: str = "default",
        filename: str = "reference.mp4",
    ) -> dict:
        """Prepare avatar material from a reference video.

        Args:
            video_bytes: Reference video file contents.
            avatar_id: Unique avatar identifier.
            filename: Original filename for content-type detection.

        Returns:
            Response dict with avatar_id, frame_count, preparation_time_s.
        """
        t0 = time.monotonic()
        resp = await self._http.post(
            f"{self.base_url}/prepare",
            files={"video": (filename, video_bytes)},
            data={"avatar_id": avatar_id},
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info(
            "Avatar '%s' prepared: %d frames in %.1fs",
            avatar_id, result.get("frame_count", 0), time.monotonic() - t0,
        )
        return result

    async def start_session(self, avatar_id: str = "default") -> str:
        """Start a new streaming animation session.

        Args:
            avatar_id: Which prepared avatar to use.

        Returns:
            Session ID string.
        """
        resp = await self._http.post(
            f"{self.base_url}/session/start",
            data={"avatar_id": avatar_id},
        )
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        logger.info("Face session started: %s (avatar=%s)", session_id, avatar_id)
        return session_id

    async def feed_audio(self, session_id: str, pcm_chunk: bytes) -> list[bytes]:
        """Feed raw PCM audio and get JPEG frames back.

        Args:
            session_id: Active session ID.
            pcm_chunk: Raw PCM audio (44100Hz, mono, int16 LE).

        Returns:
            List of JPEG frame bytes (may be empty if not enough audio yet).
        """
        resp = await self._http.post(
            f"{self.base_url}/session/{session_id}/feed",
            content=pcm_chunk,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        data = resp.json()

        frames = []
        for b64_frame in data.get("frames", []):
            frames.append(base64.b64decode(b64_frame))
        return frames

    async def end_session(self, session_id: str) -> list[bytes]:
        """End a session and flush final frames.

        Args:
            session_id: Session to end.

        Returns:
            List of remaining JPEG frame bytes.
        """
        resp = await self._http.post(
            f"{self.base_url}/session/{session_id}/end",
        )
        resp.raise_for_status()
        data = resp.json()

        frames = []
        for b64_frame in data.get("frames", []):
            frames.append(base64.b64decode(b64_frame))

        logger.info("Face session '%s' ended, %d final frames", session_id, len(frames))
        return frames

    async def list_avatars(self) -> dict:
        """List prepared avatars.

        Returns:
            Dict with avatar IDs and their frame counts.
        """
        resp = await self._http.get(f"{self.base_url}/avatars")
        resp.raise_for_status()
        return resp.json()

    async def get_idle_frames(self, avatar_id: str, max_frames: int = 30) -> list[bytes]:
        """Get reference frames for client-side idle animation.

        Args:
            avatar_id: Avatar to get frames from.
            max_frames: Maximum number of frames to return.

        Returns:
            List of JPEG frame bytes.
        """
        resp = await self._http.get(
            f"{self.base_url}/avatars/{avatar_id}/idle_frames",
            params={"max_frames": max_frames},
        )
        resp.raise_for_status()
        data = resp.json()

        frames = []
        for b64_frame in data.get("frames", []):
            frames.append(base64.b64decode(b64_frame))

        logger.info("Got %d idle frames for avatar '%s'", len(frames), avatar_id)
        return frames

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
