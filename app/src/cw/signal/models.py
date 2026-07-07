from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalState(Enum):
    MARK = "mark"
    SPACE = "space"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignalRun:
    state: SignalState
    duration_s: float


@dataclass(frozen=True)
class SignalTrack:
    analyzer: str
    runs: tuple[SignalRun, ...] = ()
    unknown_ratio: float = 0.0


__all__ = ["SignalState", "SignalRun", "SignalTrack"]
