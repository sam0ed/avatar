"""MuseTalk 1.5 streaming face animation service.

Standalone FastAPI service running inside the MuseTalk venv.
Provides a stateful session API: prepare avatar material once,
then stream audio chunks through start/feed/end to get JPEG frames.

Whisper-tiny is used as an **audio feature extractor** (encoder hidden
states drive the UNet), NOT for transcription.  Features are cached
across feed() calls to avoid redundant re-extraction.

Runs on port 8002 inside the Vast.ai container.
"""

import base64
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile

logger = logging.getLogger("avatar.face_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Global model handles (loaded once at startup) ────────────────

_models: dict[str, Any] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Avatar data directory
AVATARS_DIR = Path(os.environ.get("AVATARS_DIR", "/app/avatars"))
AVATARS_DIR.mkdir(parents=True, exist_ok=True)

# MuseTalk config
MUSETALK_DIR = Path(os.environ.get("MUSETALK_DIR", "/opt/musetalk"))
FPS = 25
FRAME_DURATION_S = 1.0 / FPS  # 40ms per frame
# Audio samples per video frame (TTS outputs 44100Hz)
AUDIO_SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // FPS  # 1764

# Whisper operates at 16kHz internally
WHISPER_SR = 16000

# MuseTalk audio padding: left=2, right=2 frames
AUDIO_PAD_LEFT = 2
AUDIO_PAD_RIGHT = 2

# JPEG quality for output frames
JPEG_QUALITY = 80

app = FastAPI(title="MuseTalk Face Animation", version="0.1.0")


# ── Data structures ──────────────────────────────────────────────

@dataclass
class AvatarData:
    """Prepared avatar material (one-time computation)."""

    avatar_id: str
    frame_count: int
    # Original frames at full resolution (list of numpy HWC uint8)
    frames: list[np.ndarray] = field(repr=False)
    # Cropped face regions (256x256 numpy arrays)
    crop_frames: list[np.ndarray] = field(repr=False)
    # Face crop coordinates [(y1, y2, x1, x2), ...]
    crop_coords: list[tuple[int, int, int, int]] = field(repr=False)
    # VAE latent space representations of masked faces
    latents: list[torch.Tensor] = field(repr=False)
    # Mask images for blending
    masks: list[np.ndarray] = field(repr=False)


@dataclass
class AnimationSession:
    """Stateful streaming session for one response."""

    session_id: str
    avatar_id: str
    # Accumulated raw PCM audio (44100Hz, mono, int16) as bytes
    audio_buffer: bytearray = field(default_factory=bytearray)
    # Accumulated audio resampled to 16kHz float32 for whisper
    whisper_audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    # Cached whisper features (tensor of shape [T, D])
    cached_features: torch.Tensor | None = None
    # Number of whisper feature frames already cached
    cached_feature_frames: int = 0
    # Next video frame index to generate
    next_frame_idx: int = 0
    # Creation time
    created_at: float = field(default_factory=time.time)


# In-memory stores
_avatars: dict[str, AvatarData] = {}
_sessions: dict[str, AnimationSession] = {}


# ── Model loading ────────────────────────────────────────────────

def _load_models() -> None:
    """Load all MuseTalk models into GPU memory."""
    import sys
    sys.path.insert(0, str(MUSETALK_DIR))

    from musetalk.utils.utils import load_all_model

    logger.info("Loading MuseTalk models (device=%s, dtype=%s)...", _device, _dtype)
    t0 = time.time()

    audio_processor, vae, unet, pe = load_all_model()

    # Store globally
    _models["audio_processor"] = audio_processor
    _models["vae"] = vae
    _models["unet"] = unet
    _models["pe"] = pe

    # Load whisper model for audio feature extraction
    import whisper
    whisper_model = whisper.load_model("tiny", device=_device)
    _models["whisper"] = whisper_model

    logger.info("All models loaded in %.1fs", time.time() - t0)


@app.on_event("startup")
async def startup() -> None:
    """Load models on service start."""
    _load_models()
    logger.info("MuseTalk face animation service ready on port 8002")


# ── Helper functions ─────────────────────────────────────────────

def _pcm_to_whisper_audio(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw PCM (44100Hz, mono, int16) to whisper-compatible float32 at 16kHz.

    Args:
        pcm_bytes: Raw PCM audio data.

    Returns:
        Float32 numpy array at 16kHz, normalized to [-1, 1].
    """
    import librosa

    # Decode int16 PCM
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # Resample from 44100 to 16000
    audio_16k = librosa.resample(audio_f32, orig_sr=AUDIO_SAMPLE_RATE, target_sr=WHISPER_SR)
    return audio_16k


def _extract_whisper_features(audio_16k: np.ndarray) -> torch.Tensor:
    """Extract whisper encoder features from 16kHz audio.

    Args:
        audio_16k: Float32 audio at 16kHz.

    Returns:
        Whisper encoder hidden states tensor.
    """
    whisper_model = _models["whisper"]

    # Pad or trim to 30 seconds (whisper's expected input length)
    audio_padded = whisper.pad_or_trim(audio_16k)

    # Compute log-mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_padded).to(_device)

    # Extract encoder features
    with torch.no_grad():
        features = whisper_model.encoder(mel.unsqueeze(0))

    return features.squeeze(0)  # [T, D]


def _extract_whisper_features_cached(
    session: AnimationSession,
    new_audio_16k: np.ndarray,
) -> torch.Tensor:
    """Extract whisper features with caching for incremental audio.

    Only processes new audio + a small overlap boundary, then concatenates
    with previously cached features.

    Args:
        session: Current animation session with cached state.
        new_audio_16k: New 16kHz float32 audio to process.

    Returns:
        Full whisper encoder features for all accumulated audio.
    """
    # Append new audio to session buffer
    session.whisper_audio = np.concatenate([session.whisper_audio, new_audio_16k])

    # For the first chunk or very short audio, do full extraction
    total_audio = session.whisper_audio
    if session.cached_features is None or len(total_audio) < WHISPER_SR:
        features = _extract_whisper_features(total_audio)
        session.cached_features = features
        return features

    # For subsequent chunks: extract features on new audio + 200ms overlap
    overlap_samples = int(0.2 * WHISPER_SR)  # 200ms boundary overlap
    cached_audio_len = int(session.cached_feature_frames * WHISPER_SR * FRAME_DURATION_S)
    start_from = max(0, cached_audio_len - overlap_samples)

    segment_audio = total_audio[start_from:]
    new_features = _extract_whisper_features(segment_audio)

    # The overlap region's features replace the tail of the cached features
    # to ensure continuity at boundaries
    overlap_feature_frames = int(overlap_samples / (WHISPER_SR * FRAME_DURATION_S))
    if overlap_feature_frames > 0 and session.cached_features is not None:
        # Trim cached features to remove the overlap region
        trim_point = max(0, session.cached_features.shape[0] - overlap_feature_frames)
        cached_trimmed = session.cached_features[:trim_point]
        features = torch.cat([cached_trimmed, new_features], dim=0)
    else:
        features = new_features

    session.cached_features = features
    session.cached_feature_frames = features.shape[0]
    return features


def _get_whisper_chunk(features: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Get whisper features for a specific video frame with audio padding.

    MuseTalk uses a window of audio context around each frame:
    [frame_idx - pad_left, frame_idx + pad_right] (inclusive).

    Args:
        features: Full whisper features tensor [T, D].
        frame_idx: Video frame index.

    Returns:
        Audio feature chunk for this frame.
    """
    audio_processor = _models["audio_processor"]
    # MuseTalk's get_whisper_chunk equivalent
    # Each video frame corresponds to a whisper feature index
    # (whisper produces 50 features/s = 2 per video frame at 25fps)
    whisper_fps = 50  # whisper produces 50 feature frames per second
    frames_per_video = whisper_fps // FPS  # 2

    center = frame_idx * frames_per_video
    pad_left = AUDIO_PAD_LEFT * frames_per_video
    pad_right = AUDIO_PAD_RIGHT * frames_per_video

    start = max(0, center - pad_left)
    end = min(features.shape[0], center + pad_right + frames_per_video)

    chunk = features[start:end]

    # Pad if needed (at boundaries)
    expected_len = (AUDIO_PAD_LEFT + AUDIO_PAD_RIGHT + 1) * frames_per_video
    if chunk.shape[0] < expected_len:
        pad_size = expected_len - chunk.shape[0]
        chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_size))

    return chunk.unsqueeze(0).to(_device, dtype=_dtype)


def _generate_frames(
    avatar: AvatarData,
    features: torch.Tensor,
    start_frame: int,
    num_frames: int,
) -> list[np.ndarray]:
    """Generate face animation frames using MuseTalk UNet.

    Args:
        avatar: Prepared avatar data.
        features: Whisper features for the full audio.
        start_frame: First frame index to generate.
        num_frames: Number of frames to generate.

    Returns:
        List of full-resolution numpy frames (HWC, uint8).
    """
    import sys
    sys.path.insert(0, str(MUSETALK_DIR))
    from musetalk.utils.blending import get_image_blending

    vae = _models["vae"]
    unet = _models["unet"]
    pe = _models["pe"]

    result_frames = []

    for i in range(num_frames):
        frame_idx = start_frame + i
        # Cycle through avatar frames if response is longer than reference video
        avatar_idx = frame_idx % avatar.frame_count

        # Get audio features for this frame
        audio_chunk = _get_whisper_chunk(features, frame_idx)

        # Get the masked face latent
        latent = avatar.latents[avatar_idx].unsqueeze(0).to(_device, dtype=_dtype)

        # MuseTalk single-step UNet inference
        with torch.no_grad():
            pred_latents = unet.model(
                latent,
                timesteps=torch.tensor([0], device=_device),
                encoder_hidden_states=audio_chunk,
            ).sample

            # VAE decode
            recon = vae.decode(pred_latents / vae.config.scaling_factor).sample
            recon = (recon.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
            recon = (recon * 255).byte().cpu().numpy()
            recon = recon.squeeze(0).transpose(1, 2, 0)  # CHW → HWC

        # Blend into original frame
        original_frame = avatar.frames[avatar_idx].copy()
        y1, y2, x1, x2 = avatar.crop_coords[avatar_idx]
        mask = avatar.masks[avatar_idx]

        # Resize reconstruction to face region size
        face_h = y2 - y1
        face_w = x2 - x1
        recon_resized = cv2.resize(recon, (face_w, face_h))

        # Apply blending with mask
        blended = get_image_blending(original_frame, recon_resized, (y1, y2, x1, x2), mask)
        result_frames.append(blended)

    return result_frames


def _frame_to_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a numpy frame as JPEG bytes.

    Args:
        frame: HWC uint8 numpy array (BGR).
        quality: JPEG quality (0-100).

    Returns:
        JPEG compressed bytes.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
    return jpeg_buf.tobytes()


# ── API endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Health check with status info."""
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return {
        "status": "ok" if _models else "loading",
        "avatars": list(_avatars.keys()),
        "active_sessions": len(_sessions),
        "vram_mb": round(vram_mb, 1),
    }


@app.post("/prepare")
async def prepare_avatar(
    video: UploadFile = File(..., description="Reference video (MP4/AVI)"),
    avatar_id: str = Form("default", description="Avatar identifier"),
) -> dict:
    """Prepare avatar material from a reference video.

    One-time computation: extracts frames, detects faces, computes
    DWPose landmarks, encodes VAE latents, generates masks.
    Results are cached in memory.

    Args:
        video: Reference video file.
        avatar_id: Unique identifier for this avatar.

    Returns:
        Avatar metadata (id, frame count).
    """
    import sys
    sys.path.insert(0, str(MUSETALK_DIR))

    video_bytes = await video.read()
    t0 = time.time()

    # Save video temporarily
    avatar_dir = AVATARS_DIR / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    video_path = avatar_dir / "reference.mp4"
    video_path.write_bytes(video_bytes)

    logger.info("Preparing avatar '%s' from %d bytes...", avatar_id, len(video_bytes))

    # Extract frames from video
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video")

    logger.info("Extracted %d frames from reference video", len(frames))

    # Use MuseTalk's preparation pipeline
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image_prepare_material

    vae = _models["vae"]

    # Detect face landmarks and get crop coordinates
    crop_coords = []
    crop_frames = []
    latents = []
    masks = []

    for idx, frame in enumerate(frames):
        # Get face bounding box and landmarks
        bbox, landmarks = get_landmark_and_bbox(frame, upperbondrange=0)
        if bbox is None:
            # If face not detected, use previous coords
            if crop_coords:
                bbox = crop_coords[-1]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"No face detected in frame {idx}",
                )

        y1, y2, x1, x2 = bbox
        crop_coords.append((y1, y2, x1, x2))

        # Crop and resize face region to 256x256
        face_crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
        crop_frames.append(face_crop)

        # Compute mask for blending
        mask = get_image_prepare_material(frame, landmarks)
        masks.append(mask)

        # Encode masked face to VAE latent space
        # Apply jaw mask (v1.5)
        masked_face = face_crop.copy()
        masked_face[face_crop.shape[0] // 2:, :] = 0  # mask lower half

        # Convert to tensor and encode
        face_tensor = torch.from_numpy(masked_face).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor * 2 - 1  # [0,1] → [-1,1]
        face_tensor = face_tensor.unsqueeze(0).to(_device, dtype=_dtype)

        with torch.no_grad():
            latent = vae.encode(face_tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor

        latents.append(latent.squeeze(0))

    # Store avatar data
    _avatars[avatar_id] = AvatarData(
        avatar_id=avatar_id,
        frame_count=len(frames),
        frames=frames,
        crop_frames=crop_frames,
        crop_coords=crop_coords,
        latents=latents,
        masks=masks,
    )

    elapsed = time.time() - t0
    logger.info(
        "Avatar '%s' prepared: %d frames in %.1fs",
        avatar_id, len(frames), elapsed,
    )

    return {
        "avatar_id": avatar_id,
        "frame_count": len(frames),
        "preparation_time_s": round(elapsed, 1),
    }


@app.post("/session/start")
async def start_session(avatar_id: str = Form("default")) -> dict:
    """Start a new streaming animation session.

    Args:
        avatar_id: Which prepared avatar to use.

    Returns:
        Session metadata (session_id).
    """
    if avatar_id not in _avatars:
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found. Call /prepare first.")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = AnimationSession(
        session_id=session_id,
        avatar_id=avatar_id,
    )

    logger.info("Session '%s' started for avatar '%s'", session_id, avatar_id)
    return {"session_id": session_id, "avatar_id": avatar_id}


from starlette.requests import Request


@app.post("/session/{session_id}/feed")
async def feed_audio_endpoint(session_id: str, request: Request) -> dict:
    """Feed raw PCM audio chunk and get animation frames back.

    Accepts raw PCM body (44100Hz, mono, int16 LE). Uses whisper feature
    caching to avoid redundant re-extraction.

    Args:
        session_id: Active session ID.
        request: Starlette request with raw PCM bytes as body.

    Returns:
        Dict with base64-encoded JPEG frames and frame indices.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _sessions[session_id]
    avatar = _avatars.get(session.avatar_id)
    if avatar is None:
        raise HTTPException(status_code=404, detail=f"Avatar '{session.avatar_id}' not found")

    pcm_bytes = await request.body()
    if not pcm_bytes:
        return {"frames": [], "frame_indices": []}

    t0 = time.time()

    # Append to session audio buffer
    session.audio_buffer.extend(pcm_bytes)

    # Convert new PCM to whisper-compatible audio
    new_audio_16k = _pcm_to_whisper_audio(pcm_bytes)

    # Extract whisper features with caching
    features = _extract_whisper_features_cached(session, new_audio_16k)

    # Calculate how many total frames we can generate
    total_audio_duration = len(session.audio_buffer) / (AUDIO_SAMPLE_RATE * 2)  # 2 bytes per sample (int16)
    total_possible_frames = int(total_audio_duration * FPS)

    # Account for lookahead: we need AUDIO_PAD_RIGHT frames of future audio
    max_frame = max(0, total_possible_frames - AUDIO_PAD_RIGHT)
    num_new_frames = max(0, max_frame - session.next_frame_idx)

    if num_new_frames == 0:
        return {"frames": [], "frame_indices": []}

    # Generate frames
    result_frames = _generate_frames(avatar, features, session.next_frame_idx, num_new_frames)

    # Encode to JPEG
    jpeg_frames = []
    frame_indices = []
    for i, frame in enumerate(result_frames):
        jpeg_bytes = _frame_to_jpeg(frame)
        jpeg_frames.append(base64.b64encode(jpeg_bytes).decode("ascii"))
        frame_indices.append(session.next_frame_idx + i)

    session.next_frame_idx += num_new_frames

    elapsed = time.time() - t0
    logger.debug(
        "Session '%s': fed %d bytes PCM, generated %d frames in %.1fms",
        session_id, len(pcm_bytes), num_new_frames, elapsed * 1000,
    )

    return {
        "frames": jpeg_frames,
        "frame_indices": frame_indices,
        "processing_ms": round(elapsed * 1000, 1),
    }


@app.post("/session/{session_id}/end")
async def end_session(session_id: str) -> dict:
    """End a streaming session, flush final frames, and clean up.

    Generates remaining frames using zero-padded audio to satisfy
    the lookahead requirement.

    Args:
        session_id: Session to end.

    Returns:
        Any remaining JPEG frames.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _sessions[session_id]
    avatar = _avatars.get(session.avatar_id)

    final_frames: list[str] = []
    final_indices: list[int] = []

    if avatar and session.cached_features is not None:
        # Calculate remaining frames
        total_audio_duration = len(session.audio_buffer) / (AUDIO_SAMPLE_RATE * 2)
        total_possible_frames = int(total_audio_duration * FPS)
        remaining = max(0, total_possible_frames - session.next_frame_idx)

        if remaining > 0:
            # Pad whisper features for the lookahead
            features = session.cached_features
            pad_frames = AUDIO_PAD_RIGHT * 2  # whisper feature frames
            if pad_frames > 0:
                padding = torch.zeros(pad_frames, features.shape[1], device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=0)

            result_frames = _generate_frames(avatar, features, session.next_frame_idx, remaining)
            for i, frame in enumerate(result_frames):
                jpeg_bytes = _frame_to_jpeg(frame)
                final_frames.append(base64.b64encode(jpeg_bytes).decode("ascii"))
                final_indices.append(session.next_frame_idx + i)

    # Clean up session
    del _sessions[session_id]
    logger.info("Session '%s' ended, total frames generated: %d", session_id, session.next_frame_idx + len(final_frames))

    return {
        "frames": final_frames,
        "frame_indices": final_indices,
    }


@app.get("/avatars")
async def list_avatars() -> dict:
    """List all prepared avatars."""
    return {
        "avatars": {
            aid: {"frame_count": a.frame_count}
            for aid, a in _avatars.items()
        }
    }


@app.get("/avatars/{avatar_id}/idle_frames")
async def get_idle_frames(avatar_id: str, max_frames: int = 30) -> dict:
    """Get reference frames for client-side idle animation.

    Returns downscaled JPEG frames from the reference video.

    Args:
        avatar_id: Avatar to get frames from.
        max_frames: Maximum number of frames to return.

    Returns:
        List of base64-encoded JPEG frames.
    """
    if avatar_id not in _avatars:
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")

    avatar = _avatars[avatar_id]
    # Subsample frames evenly
    step = max(1, avatar.frame_count // max_frames)
    selected = avatar.frames[::step][:max_frames]

    jpeg_frames = []
    for frame in selected:
        # Downscale to 480p
        h, w = frame.shape[:2]
        target_h = 480
        scale = target_h / h
        target_w = int(w * scale)
        resized = cv2.resize(frame, (target_w, target_h))
        jpeg_bytes = _frame_to_jpeg(resized)
        jpeg_frames.append(base64.b64encode(jpeg_bytes).decode("ascii"))

    return {
        "avatar_id": avatar_id,
        "frame_count": len(jpeg_frames),
        "fps": 5,  # recommended idle FPS
        "frames": jpeg_frames,
    }
