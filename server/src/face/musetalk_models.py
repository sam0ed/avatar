"""MuseTalk 1.5 model loading and shared output configuration.

Every model path inside MuseTalk is relative to its repository root, so this
module changes the working directory to MUSETALK_DIR before importing anything
from it.

STREAM_HEIGHT is the height everything downstream works at. Avatar material is
prepared at that size so blending and encoding happen once at the resolution
actually streamed, instead of compositing into a full-resolution canvas and
discarding most of the pixels. MuseTalk always generates a 256x256 face, so any
stream height at or above that loses nothing.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("avatar.face.models")

MUSETALK_DIR = Path(os.environ.get("MUSETALK_DIR", "/opt/musetalk"))

UNET_MODEL_PATH = "models/musetalkV15/unet.pth"
UNET_CONFIG_PATH = "models/musetalkV15/musetalk.json"
VAE_TYPE = "sd-vae"
WHISPER_DIR = "models/whisper"

LEFT_CHEEK_WIDTH = 90
RIGHT_CHEEK_WIDTH = 90

STREAM_HEIGHT = 480
CACHE_VERSION = 1


@dataclass
class MuseTalkModels:
    """Loaded MuseTalk handles shared by every request."""

    vae: Any
    unet: Any
    pe: Any
    whisper: Any
    audio_processor: Any
    face_parser: Any
    device: torch.device
    dtype: torch.dtype


def resolve_device() -> torch.device:
    """Return the device MuseTalk should run on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enter_musetalk_root() -> None:
    """Put MuseTalk on the import path and make its relative paths resolve."""
    if not MUSETALK_DIR.is_dir():
        raise RuntimeError(f"MUSETALK_DIR does not exist: {MUSETALK_DIR}")
    if str(MUSETALK_DIR) not in sys.path:
        sys.path.insert(0, str(MUSETALK_DIR))
    os.chdir(MUSETALK_DIR)


def _verify_weights() -> None:
    """Fail fast with an actionable message if a checkpoint is missing."""
    required = [
        UNET_MODEL_PATH,
        UNET_CONFIG_PATH,
        f"models/{VAE_TYPE}/config.json",
        f"models/{VAE_TYPE}/diffusion_pytorch_model.bin",
        f"{WHISPER_DIR}/config.json",
        f"{WHISPER_DIR}/pytorch_model.bin",
        f"{WHISPER_DIR}/preprocessor_config.json",
        "models/dwpose/dw-ll_ucoco_384.pth",
        "models/face-parse-bisent/79999_iter.pth",
        "models/face-parse-bisent/resnet18-5c106cde.pth",
    ]
    missing = [path for path in required if not (MUSETALK_DIR / path).exists()]
    if missing:
        raise RuntimeError(
            "Missing MuseTalk weights under "
            f"{MUSETALK_DIR}: {', '.join(missing)}. "
            "Run the model download step in entrypoint_stage2.sh."
        )


def load_models() -> MuseTalkModels:
    """Load MuseTalk 1.5 weights and the whisper feature extractor."""
    enter_musetalk_root()
    _verify_weights()

    from transformers import WhisperModel

    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import load_all_model

    device = resolve_device()
    logger.info("Loading MuseTalk 1.5 onto %s", device)

    vae, unet, pe = load_all_model(
        unet_model_path=UNET_MODEL_PATH,
        vae_type=VAE_TYPE,
        unet_config=UNET_CONFIG_PATH,
        device=device,
    )
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    dtype = unet.model.dtype

    whisper = WhisperModel.from_pretrained(WHISPER_DIR)
    whisper = whisper.to(device=device, dtype=dtype).eval()
    whisper.requires_grad_(False)

    models = MuseTalkModels(
        vae=vae,
        unet=unet,
        pe=pe,
        whisper=whisper,
        audio_processor=AudioProcessor(feature_extractor_path=WHISPER_DIR),
        face_parser=FaceParsing(
            left_cheek_width=LEFT_CHEEK_WIDTH,
            right_cheek_width=RIGHT_CHEEK_WIDTH,
        ),
        device=device,
        dtype=dtype,
    )
    logger.info("MuseTalk models ready (dtype=%s)", dtype)
    return models
