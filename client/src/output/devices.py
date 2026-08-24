"""Discovery of the virtual meeting devices (VB-Cable, OBS Virtual Camera)."""

import logging

import sounddevice as sd

logger = logging.getLogger("avatar.output.devices")

CABLE_INPUT_MARKER = "CABLE Input"


def find_cable_output_device() -> int:
    """Index of the VB-Cable playback endpoint; raises if the driver is missing."""
    for index, device in enumerate(sd.query_devices()):
        if CABLE_INPUT_MARKER in device["name"] and device["max_output_channels"] > 0:
            logger.info("VB-Cable found: [%d] %s", index, device["name"])
            return index
    raise RuntimeError(
        "VB-Cable playback device not found. Install VB-Audio Virtual Cable "
        "(https://vb-audio.com/Cable/) and reboot, then select 'CABLE Output' "
        "as the microphone in your meeting app."
    )
