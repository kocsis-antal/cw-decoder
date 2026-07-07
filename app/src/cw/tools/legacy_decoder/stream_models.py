from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cw.tools.legacy_decoder.base import DecodeResult
from cw.tools.legacy_decoder.quality import QualityScore


@dataclass(frozen=True)
class StreamSessionResult:
    session_id: int
    first_seen_s: float
    last_seen_s: float
    hits: int
    final_time_s: float
    final_reason: str
    quality: QualityScore
    decoded: DecodeResult


@dataclass(frozen=True)
class SpectrumFrame:
    start_s: float
    spectrum: np.ndarray
    freqs: np.ndarray
