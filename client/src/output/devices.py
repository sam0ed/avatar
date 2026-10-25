"""Discovery of the virtual meeting devices (VB-Cable, OBS Virtual Camera)."""

import logging
import sys

import sounddevice as sd

logger = logging.getLogger("avatar.output.devices")

MACOS = "darwin"
CABLE_INPUT_MARKER = "CABLE Input"
BLACKHOLE_MARKER = "BlackHole 2ch"
VB_CABLE_MISSING = (
    "VB-Cable playback device not found. Install VB-Audio Virtual Cable "
    "(https://vb-audio.com/Cable/) and reboot, then select 'CABLE Output' "
    "as the microphone in your meeting app."
)
BLACKHOLE_MISSING = (
    "BlackHole 2ch output device not found. Install it with "
    "'brew install --cask blackhole-2ch' and reboot, then select 'BlackHole 2ch' "
    "as the microphone in your meeting app."
)


def find_cable_output_device() -> int:
    """Index of the VB-Cable playback endpoint; raises if the driver is missing."""
    on_macos = sys.platform == MACOS
    marker = BLACKHOLE_MARKER if on_macos else CABLE_INPUT_MARKER
    missing = BLACKHOLE_MISSING if on_macos else VB_CABLE_MISSING
    for index, device in enumerate(sd.query_devices()):
        if marker in device["name"] and device["max_output_channels"] > 0:
            logger.info("Virtual cable found: [%d] %s", index, device["name"])
            return index
    raise RuntimeError(missing)
