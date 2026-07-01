from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cw.contest import ContestGrid, LiveConsensusResult, LiveContestResult, run_live_contest, summarize_live_consensus
from cw.decoder import _power_spectrum_frames, read_wav_mono


@dataclass(frozen=True)
class CarrierDetectionConfig:
    frame_ms: float = 100.0
    hop_ms: float = 20.0
    min_tone_hz: float = 200.0
    max_tone_hz: float = 2000.0
    max_carriers: int = 5
    min_separation_hz: float = 80.0
    relative_threshold: float = 0.15


@dataclass(frozen=True)
class DetectedCarrier:
    rank: int
    frequency_hz: float
    relative_power: float
    power: float


@dataclass(frozen=True)
class MultiLiveContestResult:
    rank: int
    carrier: DetectedCarrier
    live_results: list[LiveContestResult]
    consensus: list[LiveConsensusResult]

    @property
    def best(self) -> LiveContestResult:
        return self.live_results[0]

    @property
    def best_consensus(self) -> LiveConsensusResult:
        return self.consensus[0]


def detect_carriers(path: Path, config: CarrierDetectionConfig | None = None) -> list[DetectedCarrier]:
    config = config or CarrierDetectionConfig()
    signal, sample_rate = read_wav_mono(path)
    return detect_carriers_from_signal(signal, sample_rate, config)


def detect_carriers_from_signal(
    signal: np.ndarray,
    sample_rate: int,
    config: CarrierDetectionConfig | None = None,
) -> list[DetectedCarrier]:
    config = config or CarrierDetectionConfig()
    _validate_detection_config(config)

    spectrum, freqs = _power_spectrum_frames(signal, sample_rate, config.frame_ms, config.hop_ms)
    search_mask = (freqs >= config.min_tone_hz) & (freqs <= config.max_tone_hz)
    if not np.any(search_mask):
        raise ValueError("No FFT bins in configured carrier search range")

    search_freqs = freqs[search_mask]
    powers = spectrum[:, search_mask].sum(axis=0)
    if len(powers) == 0 or float(np.max(powers)) <= 0:
        return []

    candidates = _local_peak_indices(powers)
    if not candidates:
        candidates = list(range(len(powers)))
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)

    max_power = float(powers[candidates[0]])
    selected: list[tuple[float, float, float]] = []
    for index in candidates:
        power = float(powers[index])
        relative_power = power / max_power if max_power > 0 else 0.0
        if relative_power < config.relative_threshold:
            continue

        frequency_hz = float(search_freqs[index])
        if any(abs(frequency_hz - selected_frequency) < config.min_separation_hz for selected_frequency, _rel, _power in selected):
            continue

        selected.append((frequency_hz, relative_power, power))
        if len(selected) >= config.max_carriers:
            break

    return [
        DetectedCarrier(
            rank=index + 1,
            frequency_hz=round(frequency_hz, 3),
            relative_power=relative_power,
            power=power,
        )
        for index, (frequency_hz, relative_power, power) in enumerate(selected)
    ]


def run_multi_live_contest(
    wav_path: Path,
    grid: ContestGrid,
    detection_config: CarrierDetectionConfig | None = None,
    *,
    decoder_min_tone_hz: float = 200.0,
    decoder_max_tone_hz: float = 2000.0,
) -> list[MultiLiveContestResult]:
    carriers = detect_carriers(wav_path, detection_config)
    results: list[MultiLiveContestResult] = []

    for carrier in carriers:
        live_results = run_live_contest(
            wav_path,
            grid,
            min_tone_hz=decoder_min_tone_hz,
            max_tone_hz=decoder_max_tone_hz,
            target_tone_hz=carrier.frequency_hz,
        )
        consensus = summarize_live_consensus(live_results)
        results.append(
            MultiLiveContestResult(
                rank=carrier.rank,
                carrier=carrier,
                live_results=live_results,
                consensus=consensus,
            )
        )

    return results


def _local_peak_indices(values: np.ndarray) -> list[int]:
    if len(values) == 1:
        return [0]

    peaks: list[int] = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index < len(values) - 1 else -np.inf
        if value >= left and value >= right and (value > left or value > right):
            peaks.append(index)
    return peaks


def _validate_detection_config(config: CarrierDetectionConfig) -> None:
    if config.frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if config.hop_ms <= 0:
        raise ValueError("hop_ms must be positive")
    if config.min_tone_hz >= config.max_tone_hz:
        raise ValueError("min_tone_hz must be lower than max_tone_hz")
    if config.max_carriers <= 0:
        raise ValueError("max_carriers must be positive")
    if config.min_separation_hz <= 0:
        raise ValueError("min_separation_hz must be positive")
    if not 0 < config.relative_threshold <= 1:
        raise ValueError("relative_threshold must be in the (0, 1] range")
