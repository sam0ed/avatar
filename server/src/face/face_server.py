"""MuseTalk 1.5 streaming face animation service.

Standalone FastAPI service run from inside the MuseTalk checkout, in its own
virtualenv, on port 8002. Prepare an avatar once from a reference video, then
stream PCM through start/feed/end to receive JPEG frames.

Concurrency model:
  - One GPU, so all inference is serialised behind a single lock and executed
    off the event loop. /health and /avatars never take the lock.
  - A session captures the AvatarData object it started with, so re-preparing an
    avatar cannot mutate material out from under a stream already using it.
  - Abandoned sessions are swept on a timer; a dropped WebSocket cannot leak its
    accumulated audio buffer forever.
"""

import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from starlette.requests import Request

from musetalk_audio import (
    HOLDBACK_FRAMES,
    PCM_SAMPLE_RATE,
    append_pcm,
    total_frames,
    window_chunks,
)
from musetalk_avatar import AvatarData, build_avatar, load_cached_avatars
from musetalk_models import MuseTalkModels, load_models
from musetalk_profiling import StageTimer
from musetalk_render import CPU_POOL, encode_jpeg, render_frames

logger = logging.getLogger("avatar.face_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

AVATARS_DIR = Path(os.environ.get("AVATARS_DIR", "/app/avatars"))

MAX_ACTIVE_SESSIONS = 8
SESSION_IDLE_TIMEOUT_S = 300.0
SESSION_SWEEP_INTERVAL_S = 60.0

_models: MuseTalkModels | None = None
_avatars: dict[str, AvatarData] = {}
_sessions: dict[str, "AnimationSession"] = {}
_gpu = asyncio.Lock()
_render_warmed = False


@dataclass
class AnimationSession:
    """One streaming response: accumulated audio and how far we have rendered.

    Holds the AvatarData object rather than an avatar id, so the material a
    stream renders against is fixed for its lifetime.
    """

    session_id: str
    avatar: AvatarData = field(repr=False)
    audio_16k: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )
    pcm_remainder: bytes = field(default=b"", repr=False)
    next_frame_idx: int = 0
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Mark the session as still in use."""
        self.last_activity = time.monotonic()


async def _sweep_idle_sessions() -> None:
    """Drop sessions whose client vanished without calling /session/end."""
    while True:
        await asyncio.sleep(SESSION_SWEEP_INTERVAL_S)
        cutoff = time.monotonic() - SESSION_IDLE_TIMEOUT_S
        stale = [sid for sid, s in _sessions.items() if s.last_activity < cutoff]
        for session_id in stale:
            _sessions.pop(session_id, None)
            logger.warning("Swept abandoned session %s", session_id)


def _warm_render_path(models: MuseTalkModels, avatar: AvatarData) -> None:
    """Render one second of silence through the real feed path, once.

    The first feed of a cold process pays for lazy imports (librosa, MuseTalk's
    datagen and blending helpers) and cuDNN autotuning of the UNet/VAE shapes —
    measured at ~17s on an A6000, all of which landed on the first conversation
    turn. Paying it at startup instead makes the first real feed a steady-state
    feed.
    """
    global _render_warmed
    session = AnimationSession(session_id="warmup", avatar=avatar)
    silence = b"\x00" * (PCM_SAMPLE_RATE * 2)
    session.audio_16k, session.pcm_remainder = append_pcm(
        session.audio_16k, session.pcm_remainder, silence
    )
    started = time.monotonic()
    rendered, profile = _render_pending(session, models, HOLDBACK_FRAMES)
    _render_warmed = True
    logger.info(
        "Render path warmed: %d frames in %.1fs %s",
        len(rendered), time.monotonic() - started, profile,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load MuseTalk once, and run the session sweeper for the process lifetime."""
    global _models
    AVATARS_DIR.mkdir(parents=True, exist_ok=True)
    _models = await run_in_threadpool(load_models)
    _avatars.update(await run_in_threadpool(load_cached_avatars, AVATARS_DIR))
    if _avatars:
        first_avatar = next(iter(_avatars.values()))
        await run_in_threadpool(_warm_render_path, _models, first_avatar)
    sweeper = asyncio.create_task(_sweep_idle_sessions())
    logger.info("Face animation service ready on port 8002")
    try:
        yield
    finally:
        sweeper.cancel()


app = FastAPI(title="MuseTalk Face Animation", version="0.2.0", lifespan=lifespan)


async def _run_on_gpu(func, *args):
    """Run blocking GPU work off the event loop, one job at a time."""
    async with _gpu:
        return await run_in_threadpool(func, *args)


def _require_models() -> MuseTalkModels:
    """Return loaded models or fail with a clear error."""
    if _models is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    return _models


def _require_session(session_id: str) -> AnimationSession:
    """Look up an active session."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _render_pending(
    session: AnimationSession,
    models: MuseTalkModels,
    holdback: int,
) -> tuple[list[tuple[int, str]], dict[str, float]]:
    """Render every frame whose audio context is complete.

    Returns (frame_index, base64 JPEG) pairs so reported indices stay correct
    even when a frame is dropped, plus per-stage timings in milliseconds.
    """
    timer = StageTimer(models.device)
    available = max(0, total_frames(session.audio_16k) - holdback)
    if available <= session.next_frame_idx:
        return [], timer.rounded()

    chunks, window_start = window_chunks(
        session.audio_16k, session.next_frame_idx, models, timer
    )
    first = session.next_frame_idx - window_start
    last = min(available - window_start, len(chunks))
    if last <= first:
        return [], timer.rounded()

    pending = chunks[first:last]
    rendered = render_frames(session.avatar, pending, session.next_frame_idx, models, timer)
    session.next_frame_idx = window_start + last

    encoded: list[tuple[int, str]] = []
    with timer.stage("jpeg"):
        jpegs = CPU_POOL.map(lambda item: encode_jpeg(item[1]), rendered)
        for (frame_index, _), jpeg in zip(rendered, jpegs):
            if jpeg is not None:
                encoded.append((frame_index, base64.b64encode(jpeg).decode("ascii")))

    logger.info(
        "feed_profile frames=%d window_frames=%d %s",
        len(encoded), len(chunks), timer.as_log(),
    )
    return encoded, timer.rounded()


def _as_response(rendered: list[tuple[int, str]]) -> dict:
    """Split (index, frame) pairs into the wire format."""
    return {
        "frames": [frame for _, frame in rendered],
        "frame_indices": [index for index, _ in rendered],
    }


@app.get("/health")
async def health() -> dict:
    """Report readiness, prepared avatars, and VRAM in use."""
    import torch

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return {
        "status": "ok" if _models is not None else "loading",
        "avatars": list(_avatars.keys()),
        "active_sessions": len(_sessions),
        "vram_mb": round(vram_mb, 1),
    }


@app.get("/avatars")
async def list_avatars() -> dict:
    """List prepared avatars and their cycled frame counts."""
    return {
        "avatars": {
            avatar_id: {"frame_count": avatar.frame_count}
            for avatar_id, avatar in _avatars.items()
        }
    }


@app.post("/prepare")
async def prepare_avatar(
    video: UploadFile = File(..., description="Reference video (MP4/AVI)"),
    avatar_id: str = Form("default", description="Avatar identifier"),
) -> dict:
    """Build and cache avatar material from a reference video."""
    models = _require_models()
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video upload")

    work_dir = AVATARS_DIR / avatar_id
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "reference.mp4"
    video_path.write_bytes(video_bytes)

    started = time.monotonic()
    try:
        avatar = await _run_on_gpu(build_avatar, avatar_id, video_path, work_dir, models)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _avatars[avatar_id] = avatar
    if not _render_warmed:
        await _run_on_gpu(_warm_render_path, models, avatar)
    return {
        "avatar_id": avatar_id,
        "frame_count": avatar.frame_count,
        "preparation_time_s": round(time.monotonic() - started, 1),
    }


@app.post("/session/start")
async def start_session(avatar_id: str = Form("default")) -> dict:
    """Open a streaming animation session against a prepared avatar."""
    avatar = _avatars.get(avatar_id)
    if avatar is None:
        raise HTTPException(
            status_code=404, detail=f"Avatar '{avatar_id}' not found. Call /prepare first."
        )
    if len(_sessions) >= MAX_ACTIVE_SESSIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many active sessions ({len(_sessions)}); try again shortly",
        )

    session_id = str(uuid.uuid4())
    _sessions[session_id] = AnimationSession(session_id=session_id, avatar=avatar)
    logger.info("Session %s started for avatar '%s'", session_id, avatar_id)
    return {"session_id": session_id, "avatar_id": avatar_id}


@app.post("/session/{session_id}/feed")
async def feed_audio(session_id: str, request: Request) -> dict:
    """Accept raw PCM (44100 Hz, mono, int16 LE) and return new frames."""
    models = _require_models()
    session = _require_session(session_id)

    pcm_bytes = await request.body()
    if not pcm_bytes:
        return {"frames": [], "frame_indices": []}

    started = time.monotonic()
    session.audio_16k, session.pcm_remainder = append_pcm(
        session.audio_16k, session.pcm_remainder, pcm_bytes
    )
    session.touch()

    rendered, profile = await _run_on_gpu(_render_pending, session, models, HOLDBACK_FRAMES)
    session.touch()

    elapsed_ms = (time.monotonic() - started) * 1000
    logger.debug(
        "Session %s: +%d bytes PCM, %d frames in %.0fms",
        session_id, len(pcm_bytes), len(rendered), elapsed_ms,
    )
    return {
        **_as_response(rendered),
        "processing_ms": round(elapsed_ms, 1),
        "profile": profile,
    }


@app.post("/session/{session_id}/end")
async def end_session(session_id: str) -> dict:
    """Flush the held-back tail frames and discard the session."""
    models = _require_models()
    session = _require_session(session_id)

    try:
        rendered, _ = await _run_on_gpu(_render_pending, session, models, 0)
    finally:
        _sessions.pop(session_id, None)

    logger.info("Session %s ended: %d frames total", session_id, session.next_frame_idx)
    return _as_response(rendered)


@app.get("/avatars/{avatar_id}/idle_frames")
async def get_idle_frames(avatar_id: str, max_frames: int = 30) -> dict:
    """Return evenly sampled reference frames for client-side idle animation."""
    avatar = _avatars.get(avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")

    step = max(1, avatar.frame_count // max_frames)
    encoded = []
    for frame in avatar.frames[::step][:max_frames]:
        jpeg = encode_jpeg(frame)
        if jpeg is not None:
            encoded.append(base64.b64encode(jpeg).decode("ascii"))

    return {
        "avatar_id": avatar_id,
        "frame_count": len(encoded),
        "fps": 5,
        "frames": encoded,
    }
