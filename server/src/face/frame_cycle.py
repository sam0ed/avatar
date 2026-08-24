"""Shared ping-pong mapping over the avatar frame cycle; dependency-free."""


def pingpong_index(position: int, frame_count: int) -> int:
    """Map a monotonically growing position onto a forward-backward frame sweep.

    Plain modulo jumps from the last frame back to the first every cycle;
    ping-pong reverses direction instead, so long turns and the idle loop
    never show a seam. The client display uses the identical mapping.
    """
    if frame_count <= 1:
        return 0
    period = 2 * frame_count - 2
    pos = position % period
    return pos if pos < frame_count else period - pos
