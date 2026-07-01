from __future__ import annotations

from dataclasses import dataclass

from cw.decoder import DecodeResult


@dataclass(frozen=True)
class QualityScore:
    score: float
    unknown_count: int
    token_count: int
    dot_count: int
    dash_count: int
    tone_ratio_error: float
    gap_min_error: float
    unit_cv: float


def score_decode_result(result: DecodeResult) -> QualityScore:
    tones = [run for run in result.classified_runs if run.kind == "tone"]
    gaps = [run for run in result.classified_runs if run.kind == "gap"]
    dots = [run for run in tones if run.symbol == "."]
    dashes = [run for run in tones if run.symbol == "-"]
    unknown_count = result.text.count("?")
    token_count = len([token for token in result.tokens if token != "/"])

    tone_ratio_error = _tone_ratio_error(dots, dashes)
    gap_min_error = _gap_min_error(gaps)
    unit_cv = _unit_cv(dots)

    score = (
        unknown_count * 300
        + _invalid_token_penalty(result.tokens)
        + _too_few_tokens_penalty(token_count)
        + tone_ratio_error * 120
        + gap_min_error * 80
        + unit_cv * 60
    )

    return QualityScore(
        score=score,
        unknown_count=unknown_count,
        token_count=token_count,
        dot_count=len(dots),
        dash_count=len(dashes),
        tone_ratio_error=tone_ratio_error,
        gap_min_error=gap_min_error,
        unit_cv=unit_cv,
    )


def _tone_ratio_error(dots, dashes) -> float:
    if not dots or not dashes:
        return 0.0

    dot_mean = sum(run.duration_s for run in dots) / len(dots)
    dash_mean = sum(run.duration_s for run in dashes) / len(dashes)
    if dot_mean <= 0:
        return 10.0

    ratio = dash_mean / dot_mean
    return _soft_ratio_error(ratio, target=3.0, tolerance=0.75)


def _gap_min_error(gaps) -> float:
    error = 0.0
    for gap in gaps:
        if gap.symbol == "element_gap":
            error += _below_min_error(gap.units, minimum=0.75)
        elif gap.symbol == "letter_gap":
            error += _below_min_error(gap.units, minimum=2.5)
        elif gap.symbol == "word_gap":
            error += _below_min_error(gap.units, minimum=5.5)
    return error / max(len(gaps), 1)


def _unit_cv(dots) -> float:
    if len(dots) < 2:
        return 0.0

    durations = [run.duration_s for run in dots]
    mean = sum(durations) / len(durations)
    if mean <= 0:
        return 10.0

    variance = sum((duration - mean) ** 2 for duration in durations) / len(durations)
    return variance**0.5 / mean


def _invalid_token_penalty(tokens: list[str]) -> float:
    return sum(80 for token in tokens if token and token != "/" and len(token) > 6)


def _too_few_tokens_penalty(token_count: int) -> float:
    if token_count == 0:
        return 1000
    if token_count == 1:
        return 100
    return 0


def _below_min_error(value: float, minimum: float) -> float:
    if value >= minimum:
        return 0.0
    return (minimum - value) / minimum


def _soft_ratio_error(value: float, target: float, tolerance: float) -> float:
    distance = abs(value - target)
    if distance <= tolerance:
        return 0.0
    return (distance - tolerance) / target
