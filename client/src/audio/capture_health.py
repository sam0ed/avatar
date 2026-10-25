import math

import numpy as np

CONSTANT_RANGE = 1e-6


class CaptureHealth:
    def __init__(self) -> None:
        self._frames = 0
        self._first_signal: int | None = None
        self._low = math.inf
        self._high = -math.inf

    def observe(self, block: np.ndarray) -> None:
        start = self._frames
        self._frames += len(block)
        if self._first_signal is None:
            nonzero = np.flatnonzero(block)
            if nonzero.size == 0:
                return
            self._first_signal = start + int(nonzero[0])
            block = block[nonzero[0] :]
        self._low = min(self._low, float(block.min()))
        self._high = max(self._high, float(block.max()))

    def problem(self) -> str | None:
        if self._frames == 0:
            return "no microphone audio arrived"
        if self._first_signal is None:
            return "the microphone delivered only zeros"
        if self._high - self._low < CONSTANT_RANGE:
            return "the microphone delivered a constant signal"
        return None

    def describe(self, rate: float) -> str:
        leading = self._frames if self._first_signal is None else self._first_signal
        return f"{self._frames / rate:.2f} s captured, leading zeros {leading / rate:.2f} s"
