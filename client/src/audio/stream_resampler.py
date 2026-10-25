from math import gcd

import numpy as np
from scipy.signal import firwin, upfirdn

HALF_LENGTH_PER_RATE_STEP = 10
KAISER_BETA = 5.0
IDENTITY_TAPS = np.ones(1)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


class StreamResampler:
    def __init__(self, rate_in: int, rate_out: int) -> None:
        common = gcd(rate_in, rate_out)
        self._up = rate_out // common
        self._down = rate_in // common
        self._taps = IDENTITY_TAPS if self._up == self._down else self._design_taps()
        support = _ceil_div(len(self._taps) - 1, self._up)
        self._history = np.zeros(support + self._down, dtype=np.float32)
        self._consumed = 0
        self._emitted = 0

    def _design_taps(self) -> np.ndarray:
        widest = max(self._up, self._down)
        taps = 2 * HALF_LENGTH_PER_RATE_STEP * widest + 1
        return firwin(taps, 1.0 / widest, window=("kaiser", KAISER_BETA)) * self._up

    def process(self, block: np.ndarray) -> np.ndarray:
        window = np.concatenate([self._history, block.astype(np.float32)])
        window_start = self._consumed - len(self._history)
        skip = -window_start % self._down
        aligned_start = window_start + skip
        filtered = upfirdn(self._taps, window[skip:], self._up, self._down)
        first = aligned_start * self._up // self._down
        end = _ceil_div((self._consumed + len(block)) * self._up, self._down)
        output = filtered[self._emitted - first : end - first].astype(np.float32)
        self._history = window[-len(self._history) :]
        self._consumed += len(block)
        self._emitted = end
        return output
