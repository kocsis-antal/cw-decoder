from __future__ import annotations

import numpy as np

from cw.decoder import _to_mono_float
from cw.stream_models import SpectrumFrame


class StreamingSTFT:
    """Incremental, overlapping STFT over a continuous sample stream."""

    def __init__(self, sample_rate: int, frame_ms: float, hop_ms: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if hop_ms <= 0:
            raise ValueError("hop_ms must be positive")

        self.sample_rate = sample_rate
        self.frame_length = max(1, round(sample_rate * frame_ms / 1000))
        self.hop_length = max(1, round(sample_rate * hop_ms / 1000))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._next_frame_start_sample = 0
        self._window = np.hanning(self.frame_length).astype(np.float32)
        self._freqs = np.fft.rfftfreq(self.frame_length, 1 / sample_rate)

    def push(self, samples: np.ndarray) -> list[SpectrumFrame]:
        samples = _to_mono_float(np.asarray(samples))
        if len(samples) == 0:
            return []

        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        frames: list[SpectrumFrame] = []

        while self._next_frame_start_sample + self.frame_length <= self._buffer_start_sample + len(self._buffer):
            offset = self._next_frame_start_sample - self._buffer_start_sample
            frame = self._buffer[offset : offset + self.frame_length]
            spectrum = np.abs(np.fft.rfft(frame * self._window)) ** 2
            frames.append(
                SpectrumFrame(
                    start_s=self._next_frame_start_sample / self.sample_rate,
                    spectrum=spectrum.astype(np.float32, copy=False),
                    freqs=self._freqs,
                )
            )
            self._next_frame_start_sample += self.hop_length

        self._drop_obsolete_samples()
        return frames

    def _drop_obsolete_samples(self) -> None:
        drop_count = self._next_frame_start_sample - self._buffer_start_sample
        if drop_count <= 0:
            return
        self._buffer = self._buffer[drop_count:]
        self._buffer_start_sample += drop_count
