import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    player_start: int
    content_start: int
    frames: int
    buffer: Any

    @property
    def player_end(self) -> int:
        return self.player_start + self.frames


class PlayTimeline:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segments: list[Segment] = []
        self._content_base = 0

    def reset(self, content_base: int = 0) -> None:
        with self._lock:
            self._segments = []
            self._content_base = content_base

    def place(self, buffer: Any, frames: int, earliest: int) -> int:
        with self._lock:
            last = self._segments[-1] if self._segments else None
            start = earliest if last is None else max(earliest, last.player_end)
            content = self._content_base if last is None else last.content_start + last.frames
            self._segments.append(Segment(start, content, frames, buffer))
            return start

    def content_at(self, player_time: float) -> float:
        with self._lock:
            reached = [s for s in self._segments if s.player_start <= player_time]
            if not reached:
                return float(self._content_base)
            current = reached[-1]
            return current.content_start + min(player_time - current.player_start, current.frames)

    def end(self) -> int | None:
        with self._lock:
            return self._segments[-1].player_end if self._segments else None

    def not_started(self, player_time: float) -> list[Segment]:
        with self._lock:
            return [s for s in self._segments if s.player_start > player_time]
