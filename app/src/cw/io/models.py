from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class AudioBlock:
    """One continuous block from an audio source.

    The IO layer exposes every input backend through this common shape. The
    block contains mono float samples plus timing metadata; upper layers should
    not need to know whether the samples came from a WAV file, raw PCM stream,
    or an in-memory replay array.
    """

    samples: np.ndarray
    sample_rate: int
    start_s: float
    duration_s: float
    index: int

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class AudioSource(Protocol):
    """Iterable source of mono float audio blocks."""

    sample_rate: int
    duration_s: float | None

    def __iter__(self) -> Iterator[AudioBlock]: ...
