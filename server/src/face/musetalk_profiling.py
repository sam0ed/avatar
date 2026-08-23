"""Per-stage wall-clock attribution for the render path.

Stage boundaries synchronise CUDA before and after timing. Without that,
kernel launches are asynchronous and their cost lands on whichever later
stage first blocks on the GPU (usually the `.cpu()` copy inside VAE decode),
which misattributes almost everything. The syncs serialise work the driver
could overlap, so the numbers answer "where does the time go" — the per-feed
total remains the authoritative throughput figure.
"""

import time
from contextlib import contextmanager

import torch


class StageTimer:
    """Accumulates milliseconds per named stage across one feed."""

    def __init__(self, device: object) -> None:
        self._sync = str(device).startswith("cuda") and torch.cuda.is_available()
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        """Time a block, attributing all GPU work launched inside it."""
        if self._sync:
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self._sync:
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            self.ms[name] = self.ms.get(name, 0.0) + elapsed

    def rounded(self) -> dict[str, float]:
        """Stage times rounded for wire/log use."""
        return {name: round(value, 1) for name, value in self.ms.items()}

    def as_log(self) -> str:
        """Render stages as sorted key=value pairs for structured logging."""
        return " ".join(
            f"{name}_ms={value:.1f}" for name, value in sorted(self.ms.items())
        )
