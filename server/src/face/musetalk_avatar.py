"""Reference video to cached MuseTalk avatar material.

Run once per avatar, then reused from disk. Produces the per-frame material the
renderer needs: display frames, face boxes, VAE latents, blending masks.

Two things are done at stream resolution rather than the source video's:
blending and the stored frames. MuseTalk always generates a 256x256 face, so
compositing it into a full-resolution canvas only to downscale the result wastes
work on every frame and memory for the process lifetime. The VAE latents are
still encoded from the full-resolution crop, so conditioning quality is
unchanged.

Only the un-cycled half is computed and stored; the ping-pong is rebuilt on
load, since frame i and frame 2N-1-i are the same image and share a mask.
"""

import glob
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from musetalk_models import CACHE_VERSION, STREAM_HEIGHT, MuseTalkModels

logger = logging.getLogger("avatar.face.avatar")

CROP_SIZE = 256
EXTRA_MARGIN = 10
PARSING_MODE = "jaw"
COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)
MAX_REFERENCE_FRAMES = 900


@dataclass
class AvatarData:
    """Per-frame material for one prepared avatar, ping-pong cycled."""

    avatar_id: str
    frames: list[np.ndarray] = field(repr=False)
    coords: list[list[int]] = field(repr=False)
    latents: list[Any] = field(repr=False)
    masks: list[np.ndarray] = field(repr=False)
    crop_boxes: list[Any] = field(repr=False)

    @property
    def frame_count(self) -> int:
        """Number of frames in the cycle."""
        return len(self.frames)


def _ping_pong(items: list) -> list:
    """Append the reverse so looping never jumps discontinuously."""
    return items + items[::-1]


def _cycled(
    frames: list, coords: list, latents: list, masks: list, crop_boxes: list
) -> tuple[list, list, list, list, list]:
    """Expand the stored half into the full playback cycle."""
    return (
        _ping_pong(frames),
        _ping_pong(coords),
        _ping_pong(latents),
        _ping_pong(masks),
        _ping_pong(crop_boxes),
    )


def extract_frames(video_path: Path, out_dir: Path) -> list[str]:
    """Write each video frame to out_dir as PNG and return sorted paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    capture = cv2.VideoCapture(str(video_path))
    index = 0
    while index < MAX_REFERENCE_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{index:08d}.png"), frame)
        index += 1
    truncated = capture.read()[0]
    capture.release()

    if truncated:
        logger.warning(
            "Reference video exceeds %d frames; only the first %d will be used",
            MAX_REFERENCE_FRAMES, MAX_REFERENCE_FRAMES,
        )

    return sorted(glob.glob(str(out_dir / "*.png")))


def _scale_for(frame: np.ndarray) -> float:
    """Factor taking a source frame down to stream height."""
    height = frame.shape[0]
    return min(1.0, STREAM_HEIGHT / height)


def _downscale(frame: np.ndarray, scale: float) -> np.ndarray:
    """Resize a frame by the given factor, if it is not already small enough."""
    if scale >= 1.0:
        return frame
    height, width = frame.shape[:2]
    return cv2.resize(
        frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )


def _encode_faces(
    coord_list: list,
    frame_list: list,
    models: MuseTalkModels,
) -> tuple[list, list, list]:
    """Encode each face at full resolution, and keep the frame at stream size."""
    coords: list[list[int]] = []
    frames: list[np.ndarray] = []
    latents: list[Any] = []

    for bbox, frame in zip(coord_list, frame_list):
        if tuple(bbox) == COORD_PLACEHOLDER:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)
        latents.append(models.vae.get_latents_for_unet(resized))

        scale = _scale_for(frame)
        frames.append(_downscale(frame, scale))
        coords.append([int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)])

    return coords, frames, latents


def _build_masks(frames: list, coords: list, models: MuseTalkModels) -> tuple[list, list]:
    """Compute the soft blending mask and crop box for each frame."""
    from musetalk.utils.blending import get_image_prepare_material

    masks: list[np.ndarray] = []
    crop_boxes: list[Any] = []
    for frame, coord in zip(frames, coords):
        mask, crop_box = get_image_prepare_material(
            frame, coord, fp=models.face_parser, mode=PARSING_MODE
        )
        masks.append(mask)
        crop_boxes.append(crop_box)
    return masks, crop_boxes


def _cache_dir(work_dir: Path) -> Path:
    """Location of the reusable prepared material."""
    return work_dir / "cache"


def save_avatar(
    work_dir: Path,
    frames: list,
    coords: list,
    latents: list,
    masks: list,
    crop_boxes: list,
) -> None:
    """Persist the un-cycled half so a restart does not re-prepare."""
    cache = _cache_dir(work_dir)
    (cache / "frames").mkdir(parents=True, exist_ok=True)
    (cache / "masks").mkdir(parents=True, exist_ok=True)

    for index, (frame, mask) in enumerate(zip(frames, masks)):
        cv2.imwrite(str(cache / "frames" / f"{index:08d}.png"), frame)
        cv2.imwrite(str(cache / "masks" / f"{index:08d}.png"), mask)

    torch.save(latents, cache / "latents.pt")
    (cache / "meta.json").write_text(
        json.dumps({
            "version": CACHE_VERSION,
            "stream_height": STREAM_HEIGHT,
            "count": len(frames),
            "coords": coords,
            "crop_boxes": [list(box) for box in crop_boxes],
        }),
        encoding="utf-8",
    )
    logger.info("Cached prepared avatar material to %s", cache)


def load_avatar(avatar_id: str, work_dir: Path) -> AvatarData | None:
    """Rebuild an avatar from cache, or None if there is no usable cache."""
    cache = _cache_dir(work_dir)
    meta_path = cache / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != CACHE_VERSION or meta.get("stream_height") != STREAM_HEIGHT:
            logger.info("Ignoring stale avatar cache for '%s'", avatar_id)
            return None

        frame_paths = sorted(glob.glob(str(cache / "frames" / "*.png")))
        mask_paths = sorted(glob.glob(str(cache / "masks" / "*.png")))
        if len(frame_paths) != meta["count"] or len(mask_paths) != meta["count"]:
            logger.warning("Avatar cache for '%s' is incomplete; ignoring", avatar_id)
            return None

        frames = [cv2.imread(p) for p in frame_paths]
        masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
        latents = torch.load(cache / "latents.pt")
        coords = meta["coords"]
        crop_boxes = meta["crop_boxes"]
    except Exception:
        logger.exception("Failed to load avatar cache for '%s'", avatar_id)
        return None

    frames, coords, latents, masks, crop_boxes = _cycled(
        frames, coords, latents, masks, crop_boxes
    )
    logger.info("Loaded avatar '%s' from cache: %d cycled frames", avatar_id, len(frames))
    return AvatarData(
        avatar_id=avatar_id,
        frames=frames,
        coords=coords,
        latents=latents,
        masks=masks,
        crop_boxes=crop_boxes,
    )


def load_cached_avatars(avatars_dir: Path) -> dict[str, AvatarData]:
    """Load every avatar that has a usable cache on disk."""
    found: dict[str, AvatarData] = {}
    if not avatars_dir.is_dir():
        return found
    for work_dir in sorted(p for p in avatars_dir.iterdir() if p.is_dir()):
        avatar = load_avatar(work_dir.name, work_dir)
        if avatar is not None:
            found[work_dir.name] = avatar
    return found


def build_avatar(
    avatar_id: str,
    video_path: Path,
    work_dir: Path,
    models: MuseTalkModels,
) -> AvatarData:
    """Prepare all per-frame material for one reference video."""
    from musetalk.utils.preprocessing import get_landmark_and_bbox

    full_imgs = work_dir / "full_imgs"
    img_list = extract_frames(video_path, full_imgs)
    if not img_list:
        raise ValueError("No frames could be read from the reference video")
    logger.info("Avatar '%s': extracted %d frames", avatar_id, len(img_list))

    coord_list, frame_list = get_landmark_and_bbox(img_list, 0)
    coords, frames, latents = _encode_faces(coord_list, frame_list, models)
    if not frames:
        raise ValueError("No face was detected in any frame of the reference video")
    if len(frames) < len(img_list):
        logger.warning(
            "Avatar '%s': %d of %d frames had no usable face and were dropped",
            avatar_id, len(img_list) - len(frames), len(img_list),
        )

    masks, crop_boxes = _build_masks(frames, coords, models)
    save_avatar(work_dir, frames, coords, latents, masks, crop_boxes)

    for stale in full_imgs.glob("*.png"):
        stale.unlink()

    frames, coords, latents, masks, crop_boxes = _cycled(
        frames, coords, latents, masks, crop_boxes
    )
    height, width = frames[0].shape[:2]
    logger.info(
        "Avatar '%s' ready: %d cycled frames at %dx%d (~%.2f GB resident)",
        avatar_id, len(frames), width, height,
        len(frames) // 2 * height * width * 3 / 1e9,
    )

    return AvatarData(
        avatar_id=avatar_id,
        frames=frames,
        coords=coords,
        latents=latents,
        masks=masks,
        crop_boxes=crop_boxes,
    )
