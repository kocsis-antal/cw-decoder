from __future__ import annotations

import numpy as np


def to_mono_float(signal: np.ndarray) -> np.ndarray:
    """Normalize source audio into the IO layer's mono float sample contract."""

    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal.astype(np.float32, copy=False)
