import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from frame_cycle import pingpong_index
from musetalk_avatar import AvatarData
from musetalk_models import MuseTalkModels

logger = logging.getLogger("avatar.face.blend")

MASK_MAX = 255
DIV255_ROUNDING = 128
BYTE_SHIFT = 8
RGB_TO_BGR = [2, 1, 0]


@dataclass(frozen=True)
class FacePatch:
    face_box: tuple[int, int, int, int]
    paste_box: tuple[int, int, int, int]
    body: torch.Tensor
    mask: torch.Tensor


def _paste_box(face_box: tuple[int, int, int, int], frame: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = face_box
    height, width = frame.shape[:2]
    return max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)


def _face_patch(frame: np.ndarray, coord: list[int], mask: np.ndarray, crop_box: list[int],
                device: torch.device) -> FacePatch | None:
    face_box = tuple(int(v) for v in coord)
    px1, py1, px2, py2 = _paste_box(face_box, frame)
    if px2 <= px1 or py2 <= py1:
        logger.warning("Face box %s lies outside its frame; that frame will be skipped", face_box)
        return None
    cx, cy = int(crop_box[0]), int(crop_box[1])
    if px1 < cx or py1 < cy:
        raise ValueError(f"Face box {face_box} is not inside its crop box {crop_box}")
    return FacePatch(
        face_box=face_box,
        paste_box=(px1, py1, px2, py2),
        body=torch.from_numpy(frame[py1:py2, px1:px2].copy()).to(device),
        mask=torch.from_numpy(mask[py1 - cy:py2 - cy, px1 - cx:px2 - cx].copy()).to(device)[..., None],
    )


def build_blend_material(avatar: AvatarData, device: torch.device) -> list[FacePatch | None]:
    if avatar.frame_count % 2:
        raise ValueError(f"Avatar '{avatar.avatar_id}' is not ping-pong cycled")
    half = [
        _face_patch(avatar.frames[i], avatar.coords[i], avatar.masks[i], avatar.crop_boxes[i], device)
        for i in range(avatar.frame_count // 2)
    ]
    return half + half[::-1]


def decode_faces(models: MuseTalkModels, latents: torch.Tensor) -> torch.Tensor:
    vae = models.vae
    image = vae.vae.decode((1 / vae.scaling_factor) * latents.to(vae.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1).float()
    return (image * MASK_MAX).round()[:, RGB_TO_BGR]


def _blend_patch(patch: FacePatch, face: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = patch.face_box
    px1, py1, px2, py2 = patch.paste_box
    resized = F.interpolate(face[None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
    resized = resized[0, :, py1 - y1:py2 - y1, px1 - x1:px2 - x1].round().clamp(0, MASK_MAX)
    mask = patch.mask.int()
    mixed = patch.body.int() * (MASK_MAX - mask) + resized.permute(1, 2, 0).int() * mask + DIV255_ROUNDING
    return ((mixed + (mixed >> BYTE_SHIFT)) >> BYTE_SHIFT).to(torch.uint8)


def _to_host(patches: list[torch.Tensor]) -> list[np.ndarray]:
    flat = torch.cat([patch.flatten() for patch in patches]).cpu().numpy()
    sizes = np.cumsum([patch.numel() for patch in patches])[:-1]
    return [part.reshape(patch.shape) for part, patch in zip(np.split(flat, sizes), patches)]


def _pasted(frame: np.ndarray, paste_box: tuple[int, int, int, int], patch: np.ndarray) -> np.ndarray:
    px1, py1, px2, py2 = paste_box
    out = frame.copy()
    out[py1:py2, px1:px2] = patch
    return out


def blend_batch(avatar: AvatarData, faces: torch.Tensor, first_index: int) -> list[tuple[int, np.ndarray]]:
    if avatar.blend is None:
        raise RuntimeError(f"Avatar '{avatar.avatar_id}' has no GPU blend material")
    jobs = []
    for offset, face in enumerate(faces):
        idx = pingpong_index(first_index + offset, avatar.frame_count)
        if avatar.blend[idx] is not None:
            jobs.append((first_index + offset, idx, _blend_patch(avatar.blend[idx], face)))
    if not jobs:
        return []
    host = _to_host([patch for _, _, patch in jobs])
    return [
        (frame_index, _pasted(avatar.frames[idx], avatar.blend[idx].paste_box, patch))
        for (frame_index, idx, _), patch in zip(jobs, host)
    ]
