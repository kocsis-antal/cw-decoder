from __future__ import annotations

import numpy as np

from cw.tools.legacy_decoder.base import DecodeResult

def _cost_prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(values.astype(np.float64, copy=False))])

def _segment_mean_cost(prefix: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return 20.0
    return float(prefix[end] - prefix[start]) / (end - start)

def _advance_over_idle_frames(position: int, end: int, probabilities: np.ndarray, *, max_skip_frames: int) -> int:
    skipped = 0
    while position < end and probabilities[position] < 0.18 and skipped < max_skip_frames:
        position += 1
        skipped += 1
    return position

def _remaining_tone_probability(probabilities: np.ndarray, position: int, end: int) -> float:
    if position >= end:
        return 0.0
    return float(np.max(probabilities[position:end]))

def _duration_options(unit_frames: float, units: float, *, relative_width: float, max_options: int = 5) -> tuple[tuple[int, float], ...]:
    center = max(1.0, unit_frames * units)
    low = max(1, int(round(center * (1.0 - relative_width))))
    high = max(low, int(round(center * (1.0 + relative_width))))
    if high == low:
        candidates = [low]
    else:
        raw = np.linspace(low, high, num=min(max_options, high - low + 1))
        candidates = sorted({max(1, int(round(value))) for value in raw})
    output: list[tuple[int, float]] = []
    for frames in candidates:
        actual_units = frames / max(unit_frames, 1e-6)
        penalty = abs(actual_units - units) / max(units, 1e-6)
        output.append((frames, float(penalty)))
    return tuple(sorted(output, key=lambda item: item[1]))

def _symbol_hmm_candidate_is_plausible(
    decoded: DecodeResult,
    *,
    unit_s: float,
    confidence: float,
    quality_score: float,
    detector: str,
) -> bool:
    """Reject generic duration-HMM overfits before they reach session scoring.

    The direct HMM is deliberately powerful: it can explain probability frames
    without pre-cut runs.  That also means it can over-explain short noisy
    stretches as absurdly fast E/T-like Morse.  This validation is not a content
    prior; it only enforces physically plausible keying speed and a minimum
    amount of signal support.
    """

    if not 0.025 <= unit_s <= 0.250:
        return False
    compact = "".join(char for char in decoded.text if not char.isspace())
    known_chars = sum(1 for char in compact if char != "?")
    if known_chars < 2:
        return False
    token_list = [token for token in decoded.tokens if token != "/"]
    if not token_list:
        return False
    short_tokens = sum(1 for token in token_list if len(token) <= 1)
    short_density = short_tokens / max(1, len(token_list))
    # Long strings consisting mostly of one-element characters are a common HMM
    # failure mode on noise.  Permit short fragments, but reject long degeneracy.
    if len(token_list) >= 8 and short_density > 0.72:
        return False
    # High quality_score is bad.  The threshold is deliberately loose because
    # this is only an overfit guard; normal candidate ranking remains responsible
    # for choosing between plausible alternatives.
    if quality_score > 42.0 and detector == "symbol-hmm":
        return False
    if quality_score > 48.0 and detector == "char-hmm":
        return False
    if confidence < 0.42 and known_chars < 5:
        return False
    return True
