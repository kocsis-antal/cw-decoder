from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from cw.morse_table import decode_tokens


@dataclass(frozen=True)
class DecoderConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    min_tone_hz: float = 200.0
    max_tone_hz: float = 2000.0
    bandwidth_hz: float = 40.0
    threshold_ratio: float = 0.35
    target_tone_hz: float | None = None


@dataclass(frozen=True)
class DetectedRun:
    kind: str
    start_s: float
    duration_s: float


@dataclass(frozen=True)
class ClassifiedRun:
    kind: str
    start_s: float
    duration_s: float
    symbol: str
    units: float


@dataclass(frozen=True)
class DecodeResult:
    text: str
    tokens: list[str]
    runs: list[DetectedRun]
    classified_runs: list[ClassifiedRun]
    carrier_hz: float
    unit_s: float
    threshold: float


def decode_wav(path: Path, config: DecoderConfig | None = None) -> DecodeResult:
    config = config or DecoderConfig()
    signal, sample_rate = read_wav_mono(path)
    return decode_signal(signal, sample_rate, config)


def decode_signal(signal: np.ndarray, sample_rate: int, config: DecoderConfig | None = None) -> DecodeResult:
    config = config or DecoderConfig()
    signal = _to_mono_float(signal)

    energy, carrier_hz = _carrier_energy(signal, sample_rate, config)
    threshold = _energy_threshold(energy, config)
    active = energy > threshold

    runs = _runs_from_activity(active, config.hop_ms / 1000)
    unit_s = _estimate_unit_s(runs)
    classified_runs = classify_runs(runs, unit_s)
    tokens = _classified_runs_to_tokens(classified_runs)
    text = decode_tokens(tokens)

    return DecodeResult(text, tokens, runs, classified_runs, carrier_hz, unit_s, threshold)


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    signal, sample_rate = sf.read(path)
    return _to_mono_float(signal), int(sample_rate)


def classify_runs(runs: list[DetectedRun], unit_s: float) -> list[ClassifiedRun]:
    classified_runs: list[ClassifiedRun] = []

    for run in runs:
        units = run.duration_s / unit_s

        if run.kind == "tone":
            symbol = "." if units < 2 else "-"
        elif units < 2:
            symbol = "element_gap"
        elif units < 5:
            symbol = "letter_gap"
        else:
            symbol = "word_gap"

        classified_runs.append(
            ClassifiedRun(
                kind=run.kind,
                start_s=run.start_s,
                duration_s=run.duration_s,
                symbol=symbol,
                units=round(units, 3),
            )
        )

    return classified_runs


def _to_mono_float(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal.astype(np.float32, copy=False)


def _carrier_energy(
    signal: np.ndarray,
    sample_rate: int,
    config: DecoderConfig,
) -> tuple[np.ndarray, float]:
    spectrum, freqs = _power_spectrum_frames(signal, sample_rate, config.frame_ms, config.hop_ms)

    if config.target_tone_hz is None:
        carrier_hz = _dominant_carrier_hz(spectrum, freqs, config.min_tone_hz, config.max_tone_hz)
    else:
        carrier_hz = _nearest_frequency(freqs, config.target_tone_hz)

    carrier_mask = np.abs(freqs - carrier_hz) <= config.bandwidth_hz
    if not np.any(carrier_mask):
        carrier_mask[np.argmin(np.abs(freqs - carrier_hz))] = True

    energy = spectrum[:, carrier_mask].sum(axis=1)
    return energy, carrier_hz


def _power_spectrum_frames(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    hop_length = max(1, round(sample_rate * hop_ms / 1000))
    frames = _frame_signal(signal, frame_length, hop_length)

    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_length, 1 / sample_rate)
    return spectrum, freqs


def _dominant_carrier_hz(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    min_tone_hz: float,
    max_tone_hz: float,
) -> float:
    search_mask = (freqs >= min_tone_hz) & (freqs <= max_tone_hz)
    if not np.any(search_mask):
        raise ValueError("No FFT bins in configured carrier search range")

    summed = spectrum[:, search_mask].sum(axis=0)
    search_freqs = freqs[search_mask]
    return float(search_freqs[int(np.argmax(summed))])


def _nearest_frequency(freqs: np.ndarray, target_hz: float) -> float:
    return float(freqs[int(np.argmin(np.abs(freqs - target_hz)))])


def _frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))

    starts = range(0, len(signal) - frame_length + 1, hop_length)
    frames = [signal[start : start + frame_length] for start in starts]
    return np.stack(frames).astype(np.float32, copy=False)


def _energy_threshold(energy: np.ndarray, config: DecoderConfig) -> float:
    noise = float(np.percentile(energy, 10))
    signal = float(np.percentile(energy, 95))
    return noise + (signal - noise) * config.threshold_ratio


def _runs_from_activity(active: np.ndarray, hop_s: float) -> list[DetectedRun]:
    if len(active) == 0:
        return []

    runs: list[DetectedRun] = []
    current = bool(active[0])
    start = 0

    for index, value in enumerate(active[1:], start=1):
        value = bool(value)
        if value == current:
            continue

        runs.append(_make_run(current, start, index, hop_s))
        current = value
        start = index

    runs.append(_make_run(current, start, len(active), hop_s))
    return runs


def _make_run(active: bool, start: int, end: int, hop_s: float) -> DetectedRun:
    return DetectedRun(
        kind="tone" if active else "gap",
        start_s=round(start * hop_s, 10),
        duration_s=round((end - start) * hop_s, 10),
    )


def _estimate_unit_s(runs: list[DetectedRun]) -> float:
    durations = [run.duration_s for run in runs if run.kind == "tone" and run.duration_s > 0]
    if not durations:
        raise ValueError("No tone runs found")

    if len(durations) == 1:
        return durations[0]

    minimum = min(durations)
    maximum = max(durations)
    lower = max(minimum * 0.6, 0.001)
    upper = max(min(maximum / 2, maximum), lower)
    candidates = np.linspace(lower, upper, 300)

    def cost(unit_s: float) -> float:
        return sum(min(abs(duration - unit_s), abs(duration - 3 * unit_s)) for duration in durations)

    return round(float(min(candidates, key=cost)), 10)


def _classified_runs_to_tokens(runs: list[ClassifiedRun]) -> list[str]:
    tokens: list[str] = []
    current = ""

    for run in runs:
        if run.kind == "tone":
            current += run.symbol
            continue

        if run.symbol == "element_gap":
            continue

        if current:
            tokens.append(current)
            current = ""

        if run.symbol == "word_gap":
            tokens.append("/")

    if current:
        tokens.append(current)

    return _trim_word_separators(tokens)


def _trim_word_separators(tokens: list[str]) -> list[str]:
    while tokens and tokens[0] == "/":
        tokens.pop(0)
    while tokens and tokens[-1] == "/":
        tokens.pop()
    return tokens
