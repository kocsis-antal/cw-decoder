from __future__ import annotations

from typing import Iterator

import numpy as np

from cw.io.audio import to_mono_float
from cw.io.models import AudioBlock


class ArrayAudioSource:
    """Replay an in-memory signal as timed audio blocks."""

    def __init__(self, samples: np.ndarray, sample_rate: int, block_ms: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if block_ms <= 0:
            raise ValueError("block_ms must be positive")

        self.samples = to_mono_float(np.asarray(samples))
        self.sample_rate = int(sample_rate)
        self.block_ms = float(block_ms)
        self.block_size = max(1, round(self.sample_rate * self.block_ms / 1000))
        self.duration_s = len(self.samples) / self.sample_rate

    def __iter__(self) -> Iterator[AudioBlock]:
        index = 0
        for start in range(0, len(self.samples), self.block_size):
            block = self.samples[start : start + self.block_size]
            yield AudioBlock(
                samples=block,
                sample_rate=self.sample_rate,
                start_s=start / self.sample_rate,
                duration_s=len(block) / self.sample_rate,
                index=index,
            )
            index += 1
