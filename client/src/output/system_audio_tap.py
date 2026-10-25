import logging
import os
from collections.abc import Callable

import catap
import numpy as np

logger = logging.getLogger("avatar.output.ears")

POLL_SECONDS = 0.5
FLOAT32_BITS = 32
MONO = 1
PERMISSION_MISSING = (
    "Meeting ears cannot hear the meeting: the app that launched the client (VS Code, Terminal) "
    "lacks the 'System Audio Recording Only' permission, so macOS would hand the Core Audio tap "
    "only silence. Grant it in System Settings > Privacy & Security > Screen & System Audio "
    "Recording, then restart that app."
)


def capture_system_audio(on_audio: Callable[[np.ndarray, int], None], running: Callable[[], bool]) -> None:
    own_process = catap.find_process_by_pid(os.getpid())
    if own_process is None:
        raise RuntimeError("Core Audio has no process object for the client; cannot exclude it from the tap")
    session = catap.record_system_audio(
        exclude=[own_process],
        mono=True,
        on_buffer=lambda buffer: on_audio(_samples(buffer), int(buffer.format.sample_rate)),
    )
    session.start()
    logger.info("Meeting ears on a Core Audio tap of all system audio except pid %d", own_process.pid)
    try:
        _require_capture_permission(session)
        _watch(session, running)
    finally:
        session.stop()


def _require_capture_permission(session: catap.RecordingSession) -> None:
    try:
        session.set_mute_behavior(False)
    except PermissionError as exc:
        raise PermissionError(PERMISSION_MISSING) from exc


def _watch(session: catap.RecordingSession, running: Callable[[], bool]) -> None:
    while running():
        if session.wait_for_capture_failure(POLL_SECONDS):
            raise RuntimeError("Core Audio system tap capture failed")


def _samples(buffer: catap.AudioBuffer) -> np.ndarray:
    stream_format = buffer.format
    expected = stream_format.is_float and stream_format.bits_per_sample == FLOAT32_BITS
    if not expected or stream_format.num_channels != MONO:
        raise RuntimeError(f"Unexpected Core Audio tap format: {stream_format}")
    return np.frombuffer(buffer.data, dtype=np.float32)
