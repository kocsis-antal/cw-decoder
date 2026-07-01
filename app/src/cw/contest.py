from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

from cw.decoder import DecodeResult, DecoderConfig, decode_wav
from cw.evaluation import EvaluationResult, evaluate_wav
from cw.quality import QualityScore, score_decode_result


@dataclass(frozen=True)
class ContestGrid:
    frame_ms: list[float]
    hop_ms: list[float]
    bandwidth_hz: list[float]
    threshold_ratio: list[float]


@dataclass(frozen=True)
class ContestResult:
    rank: int
    score: float
    config: DecoderConfig
    evaluation: EvaluationResult


@dataclass(frozen=True)
class LiveContestResult:
    rank: int
    quality: QualityScore
    config: DecoderConfig
    decoded: DecodeResult


@dataclass(frozen=True)
class LiveConsensusResult:
    rank: int
    text: str
    count: int
    share: float
    best_score: float
    best_rank: int
    best_config: DecoderConfig
    best_decoded: DecodeResult


def run_contest(
    wav_path: Path,
    labels_path: Path,
    grid: ContestGrid,
    min_tone_hz: float = 200.0,
    max_tone_hz: float = 2000.0,
    target_tone_hz: float | None = None,
) -> list[ContestResult]:
    _validate_grid(grid)
    scored: list[tuple[float, DecoderConfig, EvaluationResult]] = []

    for frame_ms, hop_ms, bandwidth_hz, threshold_ratio in product(
        grid.frame_ms,
        grid.hop_ms,
        grid.bandwidth_hz,
        grid.threshold_ratio,
    ):
        config = DecoderConfig(
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            min_tone_hz=min_tone_hz,
            max_tone_hz=max_tone_hz,
            bandwidth_hz=bandwidth_hz,
            threshold_ratio=threshold_ratio,
            target_tone_hz=target_tone_hz,
        )
        evaluation = evaluate_wav(wav_path, labels_path, config)
        scored.append((_score_evaluation(evaluation), config, evaluation))

    scored.sort(key=lambda item: item[0])
    return [
        ContestResult(rank=index + 1, score=score, config=config, evaluation=evaluation)
        for index, (score, config, evaluation) in enumerate(scored)
    ]


def run_live_contest(
    wav_path: Path,
    grid: ContestGrid,
    min_tone_hz: float = 200.0,
    max_tone_hz: float = 2000.0,
    target_tone_hz: float | None = None,
) -> list[LiveContestResult]:
    _validate_grid(grid)
    scored: list[tuple[QualityScore, DecoderConfig, DecodeResult]] = []

    for frame_ms, hop_ms, bandwidth_hz, threshold_ratio in product(
        grid.frame_ms,
        grid.hop_ms,
        grid.bandwidth_hz,
        grid.threshold_ratio,
    ):
        config = DecoderConfig(
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            min_tone_hz=min_tone_hz,
            max_tone_hz=max_tone_hz,
            bandwidth_hz=bandwidth_hz,
            threshold_ratio=threshold_ratio,
            target_tone_hz=target_tone_hz,
        )
        decoded = decode_wav(wav_path, config)
        scored.append((score_decode_result(decoded), config, decoded))

    scored.sort(key=lambda item: item[0].score)
    return [
        LiveContestResult(rank=index + 1, quality=quality, config=config, decoded=decoded)
        for index, (quality, config, decoded) in enumerate(scored)
    ]


def summarize_live_consensus(results: list[LiveContestResult]) -> list[LiveConsensusResult]:
    if not results:
        return []

    groups: dict[str, list[LiveContestResult]] = {}
    for result in results:
        groups.setdefault(result.decoded.text, []).append(result)

    summaries: list[tuple[str, int, float, LiveContestResult]] = []
    total_count = len(results)
    for text, group in groups.items():
        best = min(group, key=lambda result: (result.quality.score, result.rank))
        summaries.append((text, len(group), len(group) / total_count, best))

    summaries.sort(key=lambda item: (-item[1], item[3].quality.score, item[3].rank, item[0]))
    return [
        LiveConsensusResult(
            rank=index + 1,
            text=text,
            count=count,
            share=share,
            best_score=best.quality.score,
            best_rank=best.rank,
            best_config=best.config,
            best_decoded=best.decoded,
        )
        for index, (text, count, share, best) in enumerate(summaries)
    ]


def _score_evaluation(result: EvaluationResult) -> float:
    timing = result.timing

    return (
        (0 if result.text_ok else 10000)
        + (1 - result.token_accuracy) * 5000
        + (1 - timing.symbol_accuracy) * 3000
        + abs(timing.count_delta) * 100
        + abs(result.carrier_error_hz) * 2
        + abs(result.unit_error_ms) * 5
        + timing.avg_start_error_ms
        + timing.avg_duration_error_ms
        + timing.max_start_error_ms * 0.25
        + timing.max_duration_error_ms * 0.25
    )


def _validate_grid(grid: ContestGrid) -> None:
    values = {
        "frame_ms": grid.frame_ms,
        "hop_ms": grid.hop_ms,
        "bandwidth_hz": grid.bandwidth_hz,
        "threshold_ratio": grid.threshold_ratio,
    }
    for name, items in values.items():
        if not items:
            raise ValueError(f"Contest grid value list must not be empty: {name}")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]
