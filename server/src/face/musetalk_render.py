"""Whisper chunks plus avatar material to finished JPEG frames."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from musetalk_avatar import AvatarData
from musetalk_models import MuseTalkModels
from musetalk_profiling import StageTimer

logger = logging.getLogger("avatar.face.render")

BATCH_SIZE = 8
JPEG_QUALITY = 80
CPU_WORKERS = min(8, os.cpu_count() or 1)

CPU_POOL = ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix="face-cpu")


def encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes | None:
    """Compress a BGR frame to JPEG bytes.

    Avatar material is already prepared at stream resolution, so there is no
    resize here: the frame handed in is the frame that goes on the wire.
    """
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else None


def _blend_into_frame(
    avatar: AvatarData,
    frame_index: int,
    generated_face: np.ndarray,
) -> np.ndarray | None:
    """Paste a generated mouth region back into its original frame."""
    from musetalk.utils.blending import get_image_blending

    idx = frame_index % avatar.frame_count
    x1, y1, x2, y2 = avatar.coords[idx]
    try:
        resized = cv2.resize(generated_face.astype(np.uint8), (x2 - x1, y2 - y1))
    except (cv2.error, ValueError):
        logger.warning("Skipped frame %d: bad face box %s", frame_index, avatar.coords[idx])
        return None

    return get_image_blending(
        avatar.frames[idx].copy(),
        resized,
        [x1, y1, x2, y2],
        avatar.masks[idx],
        avatar.crop_boxes[idx],
    )


def _latents_for(avatar: AvatarData, start_index: int, count: int) -> list:
    """Select the avatar latent that pairs with each requested frame."""
    return [
        avatar.latents[(start_index + offset) % avatar.frame_count]
        for offset in range(count)
    ]


def render_frames(
    avatar: AvatarData,
    chunks: torch.Tensor,
    start_index: int,
    models: MuseTalkModels,
    timer: StageTimer | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Generate blended frames for a run of whisper chunks.

    Returns (frame_index, frame) pairs. A frame that fails to blend is omitted
    rather than shifting the indices of everything after it.
    """
    from musetalk.utils.utils import datagen

    if len(chunks) == 0:
        return []

    timer = timer if timer is not None else StageTimer(models.device)
    latents = _latents_for(avatar, start_index, len(chunks))
    timesteps = torch.tensor([0], device=models.device)
    frames: list[tuple[int, np.ndarray]] = []
    produced = 0

    for whisper_batch, latent_batch in datagen(chunks, latents, BATCH_SIZE):
        with timer.stage("pe"):
            audio_features = models.pe(whisper_batch.to(models.device, dtype=models.dtype))
            latent_batch = latent_batch.to(device=models.device, dtype=models.unet.model.dtype)

        with torch.no_grad():
            with timer.stage("unet"):
                predicted = models.unet.model(
                    latent_batch, timesteps, encoder_hidden_states=audio_features
                ).sample
            with timer.stage("vae"):
                recon = models.vae.decode_latents(predicted.to(dtype=models.vae.vae.dtype))

        with timer.stage("blend"):
            batch_first_index = start_index + produced
            produced += len(recon)
            blended_batch = CPU_POOL.map(
                lambda item: _blend_into_frame(avatar, batch_first_index + item[0], item[1]),
                enumerate(recon),
            )
            for offset, blended in enumerate(blended_batch):
                if blended is not None:
                    frames.append((batch_first_index + offset, blended))

    return frames
