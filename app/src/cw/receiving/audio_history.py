from __future__ import annotations

from collections import deque

import numpy as np


class AudioRingBuffer:
    """Append-only streaming audio history with bounded retention.

    The stream processor repeatedly asks for recent or channel-retained audio.
    Keeping chunks in a deque avoids rebuilding one large NumPy array on every
    push; materialization happens only when a decode window is requested.

    ``max_history_s`` is a policy limit, not an automatic blind cutter.  The
    application/receiver calls ``trim_before()`` only after it has a safe audio
    boundary, so the ring buffer itself does not cut into a Morse character just
    because the wall clock advanced.
    """

    def __init__(self, sample_rate: int, max_history_s: float | None) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(sample_rate)
        self.max_history_s = max_history_s
        self.start_s = 0.0
        self._chunks: deque[np.ndarray] = deque()
        self._sample_count = 0
        self._dropped_samples = 0

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def duration_s(self) -> float:
        return self._sample_count / self.sample_rate

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    def append(self, samples: np.ndarray) -> int:
        block = np.asarray(samples, dtype=np.float32)
        if len(block) == 0:
            return 0
        self._chunks.append(block.copy())
        self._sample_count += len(block)
        return 0

    def recent(self, target_s: float) -> tuple[np.ndarray, float]:
        if self._sample_count == 0:
            return np.asarray([], dtype=np.float32), self.start_s
        target_samples = max(1, int(round(float(target_s) * self.sample_rate)))
        if target_samples >= self._sample_count:
            return self.as_array(), self.start_s
        skip = self._sample_count - target_samples
        return self.from_sample_offset(skip)

    def from_time(self, start_s: float) -> tuple[np.ndarray, float]:
        offset = int(round((float(start_s) - self.start_s) * self.sample_rate))
        return self.from_sample_offset(offset)

    def from_sample_offset(self, offset: int) -> tuple[np.ndarray, float]:
        offset = min(max(0, int(offset)), self._sample_count)
        if offset == 0:
            return self.as_array(), self.start_s
        remaining_skip = offset
        parts: list[np.ndarray] = []
        for chunk in self._chunks:
            if remaining_skip >= len(chunk):
                remaining_skip -= len(chunk)
                continue
            parts.append(chunk[remaining_skip:])
            remaining_skip = 0
        if not parts:
            return np.asarray([], dtype=np.float32), self.start_s + offset / self.sample_rate
        return np.concatenate(parts).astype(np.float32, copy=False), self.start_s + offset / self.sample_rate

    def trim_before(self, before_s: float) -> int:
        """Drop audio before ``before_s`` and return dropped samples.

        Callers are responsible for choosing a safe boundary.  The buffer clamps
        the request to its current contents and never invents a trim point from
        ``max_history_s`` on its own.
        """

        target = max(self.start_s, min(float(before_s), self.end_s))
        samples_to_drop = int(round((target - self.start_s) * self.sample_rate))
        return self._drop_samples(samples_to_drop)

    def as_array(self) -> np.ndarray:
        if not self._chunks:
            return np.asarray([], dtype=np.float32)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return np.concatenate(tuple(self._chunks)).astype(np.float32, copy=False)

    def _drop_samples(self, samples_to_drop: int) -> int:
        samples_to_drop = min(max(0, int(samples_to_drop)), self._sample_count)
        if samples_to_drop <= 0:
            return 0
        dropped = 0
        remaining = samples_to_drop
        while remaining > 0 and self._chunks:
            first = self._chunks[0]
            if remaining >= len(first):
                self._chunks.popleft()
                self._sample_count -= len(first)
                dropped += len(first)
                remaining -= len(first)
                continue
            self._chunks[0] = first[remaining:].copy()
            self._sample_count -= remaining
            dropped += remaining
            remaining = 0
        if dropped:
            self.start_s += dropped / self.sample_rate
            self._dropped_samples += dropped
        return dropped
