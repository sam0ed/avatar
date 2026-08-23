"""Raw TTS PCM to MuseTalk whisper feature chunks, over a sliding window.

MuseTalk's AudioProcessor.get_audio_feature reads a wav file from disk and its
realtime path consumes a finished recording. For a live stream we keep its
get_whisper_chunk (so the feature layout the UNet expects is unchanged: one
[50, 384] chunk per video frame, from five encoder layers) but feed it a bounded
window instead of the whole utterance.

The window carries LEFT_CONTEXT_FRAMES of already-rendered audio before the
first frame still to render, and runs to the end of what has arrived. That gives
two properties the naive whole-buffer approach lacked:

  - Cost per feed is constant rather than growing with the turn, because the
    encoder always sees a bounded span.
  - A frame is encoded once, with a fixed amount of context on both sides, and
    is never recomputed. Re-encoding the whole buffer meant whisper's global
    attention silently changed features for frames already sent.

Incoming PCM is resampled on exact 441-sample boundaries (44100:16000 reduces to
441:160), so the accumulated 16 kHz signal is sample-identical to resampling the
whole buffer at once and no chunk boundary introduces an artefact.
"""

import logging

import numpy as np
import torch

from musetalk_models import MuseTalkModels
from musetalk_profiling import StageTimer

logger = logging.getLogger("avatar.face.audio")

PCM_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000
WHISPER_SEGMENT_SECONDS = 30
FPS = 25
AUDIO_PADDING_LEFT = 2
AUDIO_PADDING_RIGHT = 2

SAMPLES_PER_VIDEO_FRAME = WHISPER_SAMPLE_RATE // FPS
RESAMPLE_GROUP_IN = 441
LEFT_CONTEXT_FRAMES = 50

HOLDBACK_FRAMES = AUDIO_PADDING_RIGHT
MIN_SAMPLES_FOR_A_FRAME = SAMPLES_PER_VIDEO_FRAME


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw mono int16 PCM to float32 in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def resample_for_whisper(audio: np.ndarray) -> np.ndarray:
    """Resample TTS-rate audio down to whisper's 16 kHz."""
    import librosa

    if audio.size == 0:
        return audio
    return librosa.resample(
        audio, orig_sr=PCM_SAMPLE_RATE, target_sr=WHISPER_SAMPLE_RATE
    )


def append_pcm(
    audio_16k: np.ndarray,
    remainder: bytes,
    pcm_bytes: bytes,
) -> tuple[np.ndarray, bytes]:
    """Resample and append new PCM, returning the buffer and unused tail bytes."""
    buffered = remainder + pcm_bytes
    total_samples = len(buffered) // 2
    usable_samples = (total_samples // RESAMPLE_GROUP_IN) * RESAMPLE_GROUP_IN
    if usable_samples == 0:
        return audio_16k, buffered

    split = usable_samples * 2
    resampled = resample_for_whisper(pcm_to_float32(buffered[:split]))
    return np.concatenate([audio_16k, resampled]), buffered[split:]


def total_frames(audio_16k: np.ndarray) -> int:
    """Number of video frames the accumulated audio can back."""
    return int(audio_16k.size // SAMPLES_PER_VIDEO_FRAME)


def _segment_features(audio_16k: np.ndarray, models: MuseTalkModels) -> list[torch.Tensor]:
    """Run the whisper feature extractor over 30-second segments."""
    segment_length = WHISPER_SEGMENT_SECONDS * WHISPER_SAMPLE_RATE
    features = []
    for start in range(0, len(audio_16k), segment_length):
        segment = audio_16k[start:start + segment_length]
        extracted = models.audio_processor.feature_extractor(
            segment,
            return_tensors="pt",
            sampling_rate=WHISPER_SAMPLE_RATE,
        ).input_features
        features.append(extracted.to(dtype=models.dtype))
    return features


def whisper_chunks_for(
    audio_16k: np.ndarray,
    models: MuseTalkModels,
    timer: StageTimer | None = None,
) -> torch.Tensor:
    """Return one whisper feature chunk per video frame, shaped [T, 50, 384]."""
    if audio_16k.size < MIN_SAMPLES_FOR_A_FRAME:
        return torch.empty(0)

    timer = timer if timer is not None else StageTimer(models.device)
    with timer.stage("whisper_mel"):
        features = _segment_features(audio_16k, models)
    with timer.stage("whisper_enc"):
        return models.audio_processor.get_whisper_chunk(
            features,
            models.device,
            models.dtype,
            models.whisper,
            len(audio_16k),
            fps=FPS,
            audio_padding_length_left=AUDIO_PADDING_LEFT,
            audio_padding_length_right=AUDIO_PADDING_RIGHT,
        )


def window_chunks(
    audio_16k: np.ndarray,
    from_frame: int,
    models: MuseTalkModels,
    timer: StageTimer | None = None,
) -> tuple[torch.Tensor, int]:
    """Encode a bounded window ending at the newest audio.

    Returns the per-frame chunks for that window and the global frame index its
    first chunk corresponds to.
    """
    window_start_frame = max(0, from_frame - LEFT_CONTEXT_FRAMES)
    window = audio_16k[window_start_frame * SAMPLES_PER_VIDEO_FRAME:]
    return whisper_chunks_for(window, models, timer), window_start_frame
