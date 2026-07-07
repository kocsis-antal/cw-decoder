from __future__ import annotations

import numpy as np
from cw.tools.legacy_decoder.base import DecodeResult
from cw.tools.legacy_decoder.models import SignalRun

def _baseband_envelope(signal: np.ndarray, sample_rate: int, carrier_hz: float, *, lowpass_ms: float) -> np.ndarray:
    mono = signal.astype(np.float32, copy=False)
    if len(mono) == 0:
        return np.asarray([], dtype=np.float32)
    time = np.arange(len(mono), dtype=np.float32) / float(sample_rate)
    mixed = mono * np.exp(-2j * np.pi * float(carrier_hz) * time)
    window_len = max(5, int(round(sample_rate * lowpass_ms / 1000)))
    if window_len % 2 == 0:
        window_len += 1
    window = np.hanning(window_len).astype(np.float32)
    if float(np.sum(window)) <= 0:
        window = np.ones(window_len, dtype=np.float32)
    window = window / np.sum(window)
    filtered_real = np.convolve(mixed.real, window, mode="same")
    filtered_imag = np.convolve(mixed.imag, window, mode="same")
    envelope = np.sqrt(filtered_real * filtered_real + filtered_imag * filtered_imag)
    return envelope.astype(np.float32, copy=False)

def _envelope_energy_frames(envelope: np.ndarray, sample_rate: int, *, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
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

def _activity_probability(energy: np.ndarray, noise_floor: float, signal_floor: float) -> np.ndarray:
    contrast = max(signal_floor - noise_floor, 1e-12)
    linear = np.clip((energy - noise_floor) / contrast, 0.0, 1.0)
    return (linear * linear * (3.0 - 2.0 * linear)).astype(np.float32)

def _signal_runs(
    decoded: DecodeResult,
    probabilities: np.ndarray,
    frame_times: np.ndarray,
    start_s: float,
    hop_s: float,
) -> list[SignalRun]:
    output: list[SignalRun] = []
    for classified in decoded.classified_runs:
        start = classified.start_s - start_s
        end = start + classified.duration_s
        if len(frame_times):
            mask = (frame_times >= start) & (frame_times < end)
            values = probabilities[mask]
        else:
            values = np.asarray([], dtype=np.float32)
        if len(values) == 0:
            center_index = int(round(start / hop_s)) if hop_s > 0 else 0
            if 0 <= center_index < len(probabilities):
                values = probabilities[center_index : center_index + 1]
        if len(values) == 0:
            confidence = 0.0
        elif classified.kind == "tone":
            confidence = float(np.mean(values))
        else:
            confidence = float(np.mean(1.0 - values))
        output.append(
            SignalRun(
                kind=classified.kind,
                start_s=round(float(classified.start_s), 6),
                duration_s=round(float(classified.duration_s), 6),
                confidence=round(max(0.0, min(1.0, confidence)), 6),
                units=round(float(classified.units), 3),
                symbol=classified.symbol,
            )
        )
    return output

def _mean_run_confidence(runs: list[SignalRun]) -> float:
    if not runs:
        return 0.0
    weights = [1.5 if run.kind == "tone" else 1.0 for run in runs]
    return float(sum(run.confidence * weight for run, weight in zip(runs, weights)) / sum(weights))

def _candidate_evidence_score(decoded: DecodeResult, quality_score: float, confidence: float) -> float:
    known_chars = sum(1 for char in decoded.text if not char.isspace() and char != "?")
    unknowns = decoded.text.count("?")
    punctuation = sum(1 for char in decoded.text if not char.isspace() and not char.isalnum() and char != "?")
    token_count = len([token for token in decoded.tokens if token != "/"])
    word_gap_count = decoded.tokens.count("/")
    duration_bonus = min(18.0, len(decoded.classified_runs) * 0.18)
    fragmentation_penalty = _text_fragmentation_penalty(decoded.text)
    return (
        known_chars * 1.6
        + token_count * 0.8
        + min(word_gap_count, 8) * 0.8
        + confidence * 18.0
        + duration_bonus
        - unknowns * 2.0
        - punctuation * 0.6
        - quality_score * 0.45
        - fragmentation_penalty
    )

def _text_fragmentation_penalty(text: str) -> float:
    """Penalize over-fragmented text-neutral hypotheses.

    Weak/high-threshold decodes sometimes explain a continuous keyed burst as a
    stream of isolated one-character "words" (for example ``Q C Q C Q``).
    That is usually a symptom of false word-gap decisions, not better signal
    evidence.  Keep short exchanges such as ``R R`` or ``E E`` possible, but
    make longer, heavily fragmented candidates compete on their actual timing
    quality instead of winning by simply producing more separated characters.
    """

    words = [word for word in text.split() if word]
    if len(words) < 4:
        return 0.0

    cleaned = ["".join(char for char in word if char.isalnum() or char == "?") for word in words]
    cleaned = [word for word in cleaned if word]
    if len(cleaned) < 4:
        return 0.0

    one_char_words = sum(1 for word in cleaned if len(word) == 1)
    one_char_density = one_char_words / max(1, len(cleaned))
    # Four or more isolated single-character words in one session is strong
    # evidence of gap fragmentation.  Density catches alternating patterns even
    # when a few longer tokens are present.
    excess_singletons = max(0, one_char_words - 3)
    density_penalty = max(0.0, one_char_density - 0.38) * len(cleaned) * 7.0
    return excess_singletons * 2.8 + density_penalty
