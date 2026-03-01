"""Smart Turn v3.2 — ML-based end-of-turn detection.

Uses the Pipecat Smart Turn v3.2 ONNX model (Whisper Tiny backbone,
~8M params) to determine whether the user has finished speaking or
is pausing mid-sentence.  Analyzes the last 8 seconds of raw audio
using prosody cues — intonation patterns, not just silence duration.

Model: pipecat-ai/smart-turn-v3 on HuggingFace (BSD-2-Clause license)
Inference: 10–65ms CPU, 8MB quantized model
Languages: 23 including English and Ukrainian
Reference: https://github.com/pipecat-ai/smart-turn
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from transformers import WhisperFeatureExtractor as _WhisperFE

logger = logging.getLogger("avatar.smart_turn")

# HuggingFace model repository
_HF_REPO_ID = "pipecat-ai/smart-turn-v3"
_MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"

# Audio settings
_SAMPLE_RATE = 16_000
_MAX_DURATION_SECONDS = 8
_MAX_SAMPLES = _MAX_DURATION_SECONDS * _SAMPLE_RATE  # 128_000


class SmartTurnAnalyzer:
    """End-of-turn detector using Smart Turn v3.2 ONNX model.

    Determines whether the user has finished their conversational turn
    based on audio prosody cues.  Designed to run after VAD detects
    silence, on up to 8 seconds of preceding audio.

    Usage::

        analyzer = SmartTurnAnalyzer()
        is_done, confidence = analyzer.is_turn_complete(audio_buffer)
    """

    def __init__(self, model_path: str | None = None, cpu_count: int = 1) -> None:
        """Initialize the Smart Turn analyzer.

        Args:
            model_path: Path to the ONNX model file.  If None, downloads
                from HuggingFace on first use (~8 MB).
            cpu_count: Number of CPU threads for ONNX inference.
        """
        if model_path is None:
            model_path = self._download_model()

        logger.info("Loading Smart Turn v3.2 from %s", model_path)

        from transformers import WhisperFeatureExtractor

        self._feature_extractor: _WhisperFE = WhisperFeatureExtractor(
            chunk_length=_MAX_DURATION_SECONDS,
        )

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cpu_count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options=so)

        logger.info("Smart Turn v3.2 loaded")

    def is_turn_complete(
        self, audio: np.ndarray, threshold: float = 0.5,
    ) -> tuple[bool, float]:
        """Analyze audio to determine if the user's turn is complete.

        Args:
            audio: Raw float32 mono PCM audio at 16 kHz.
            threshold: Probability threshold for "complete" classification.

        Returns:
            Tuple of (is_complete, probability).
        """
        # Too little audio — assume complete (don't block on noise)
        if len(audio) < _SAMPLE_RATE // 4:  # < 0.25 s
            return True, 1.0

        # Truncate to last 8 seconds (keep the end — most relevant for prosody)
        if len(audio) > _MAX_SAMPLES:
            audio = audio[-_MAX_SAMPLES:]

        # Extract Whisper-compatible mel spectrogram features
        inputs = self._feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )

        input_features: np.ndarray = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # batch dim

        # ONNX inference
        outputs = self._session.run(None, {"input_features": input_features})
        probability = float(outputs[0][0].item())

        is_complete = probability > threshold
        return is_complete, probability

    @staticmethod
    def _download_model() -> str:
        """Download Smart Turn v3.2 ONNX model from HuggingFace.

        Returns:
            Path to the downloaded model file (cached by huggingface_hub).
        """
        from huggingface_hub import hf_hub_download

        logger.info("Downloading Smart Turn v3.2 model from HuggingFace...")
        path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MODEL_FILENAME)
        logger.info("Model downloaded to %s", path)
        return path
