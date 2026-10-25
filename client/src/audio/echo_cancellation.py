import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from asr.transcriber import SpeechTranscriber
    from audio.voice_processing import VoiceProcessingIO
    from audio.voice_processing_player import VoiceProcessingPlayer

logger = logging.getLogger("avatar.audio")

MACOS = "darwin"


def uses_voice_processing(echo_cancellation: bool, meeting_audio: bool) -> bool:
    return echo_cancellation and not meeting_audio and sys.platform == MACOS


def build_voice_processing(
    push_audio: Callable[[np.ndarray], None], device: int | str | None
) -> tuple["VoiceProcessingIO", "VoiceProcessingPlayer"]:
    from audio.voice_processing import VoiceProcessingIO
    from audio.voice_processing_player import VoiceProcessingPlayer

    if device is not None:
        logger.warning("--device %s is ignored with echo cancellation; pass --no-aec to choose a device", device)
    voice_io = VoiceProcessingIO(push_audio)
    logger.info("Echo cancellation: Apple voice processing, AGC off")
    return voice_io, VoiceProcessingPlayer(voice_io)


async def start_voice_processing(voice_io: "VoiceProcessingIO", transcriber: "SpeechTranscriber") -> None:
    try:
        await voice_io.start()
    except Exception:
        transcriber.close()
        raise
