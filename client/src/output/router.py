"""Meeting-mode switchboard: whose face and voice the meeting receives."""

import logging
from collections.abc import Callable
from enum import Enum

import numpy as np

from output.mic_passthrough import MicPassthrough
from output.virtual_camera import VirtualCameraSink
from output.webcam_source import WebcamSource

logger = logging.getLogger("avatar.output.router")


class OutputMode(Enum):
    AVATAR = "avatar"
    ME = "me"


class OutputRouter:
    """Feeds the virtual camera and VB-Cable from either the avatar or the real user.

    AVATAR mode: avatar frames go to the virtual camera, avatar speech is played
    into VB-Cable by the audio player, the ASR listens for the conversation.
    ME mode: the physical webcam and microphone pass through, and the avatar
    pipeline is muted so it cannot react to the user's own words.
    """

    def __init__(
        self,
        avatar_frame: Callable[[], np.ndarray | None],
        cable_device: int,
        on_mode_change: Callable[[OutputMode], None] | None = None,
    ) -> None:
        self._avatar_frame = avatar_frame
        self._webcam = WebcamSource()
        self._mic_loop = MicPassthrough(cable_device)
        self._camera = VirtualCameraSink(self._current_frame)
        self._mode = OutputMode.AVATAR
        self._on_mode_change = on_mode_change

    @property
    def mode(self) -> OutputMode:
        return self._mode

    def start(self) -> None:
        self._camera.start()
        logger.info("Output router started in %s mode", self._mode.value)

    def stop(self) -> None:
        self._camera.stop()
        self._webcam.stop()
        self._mic_loop.stop()

    def toggle(self) -> OutputMode:
        target = OutputMode.ME if self._mode == OutputMode.AVATAR else OutputMode.AVATAR
        self.set_mode(target)
        return target

    def set_mode(self, mode: OutputMode) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        if mode == OutputMode.ME:
            self._webcam.start()
            self._mic_loop.start()
        else:
            self._mic_loop.stop()
            self._webcam.stop()
        logger.info("Output switched to %s", mode.value)
        if self._on_mode_change is not None:
            self._on_mode_change(mode)

    def _current_frame(self) -> np.ndarray | None:
        if self._mode == OutputMode.ME:
            return self._webcam.frame()
        return self._avatar_frame()
