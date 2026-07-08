from __future__ import annotations

import numpy as np

from cw.dsp.envelope import baseband_envelope


def envelope_energy_frames(envelope: np.ndarray, sample_rate: int, *, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(sample_rate * hop_ms / 1000)))
    if len(envelope) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    starts = np.arange(0, len(envelope), hop, dtype=np.int64)
    energy = np.empty(len(starts), dtype=np.float32)
    for out_index, start in enumerate(starts):
        frame = envelope[start : min(start + hop, len(envelope))]
        energy[out_index] = float(np.mean(frame * frame)) if len(frame) else 0.0
    times = starts.astype(np.float32) / float(sample_rate)
    return energy, times
