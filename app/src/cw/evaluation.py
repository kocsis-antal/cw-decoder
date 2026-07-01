from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cw.decoder import DecodeResult, DecoderConfig, decode_wav
from cw.morse_table import encode_text, normalize_text


@dataclass(frozen=True)
class TimingStats:
    compared_count: int
    count_delta: int
    avg_start_error_ms: float
    max_start_error_ms: float
    avg_duration_error_ms: float
    max_duration_error_ms: float
    symbol_accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    expected_text: str
    decoded_text: str
    text_ok: bool
    expected_tokens: list[str]
    decoded_tokens: list[str]
    token_accuracy: float
    expected_carrier_hz: float
    detected_carrier_hz: float
    carrier_error_hz: float
    expected_unit_s: float
    detected_unit_s: float
    unit_error_ms: float
    timing: TimingStats


def evaluate_wav(
    wav_path: Path,
    labels_path: Path,
    config: DecoderConfig | None = None,
) -> EvaluationResult:
    labels = _read_labels(labels_path)
    decoded = decode_wav(wav_path, config)
    expected_text = normalize_text(str(labels["text"]))
    expected_tokens = encode_text(expected_text)
    decoded_tokens = decoded.tokens

    expected_carrier_hz = float(labels["tone_hz"])
    expected_unit_s = float(labels["unit_s"])

    return EvaluationResult(
        expected_text=expected_text,
        decoded_text=decoded.text,
        text_ok=normalize_text(decoded.text) == expected_text,
        expected_tokens=expected_tokens,
        decoded_tokens=decoded_tokens,
        token_accuracy=_sequence_accuracy(expected_tokens, decoded_tokens),
        expected_carrier_hz=expected_carrier_hz,
        detected_carrier_hz=decoded.carrier_hz,
        carrier_error_hz=decoded.carrier_hz - expected_carrier_hz,
        expected_unit_s=expected_unit_s,
        detected_unit_s=decoded.unit_s,
        unit_error_ms=(decoded.unit_s - expected_unit_s) * 1000,
        timing=_timing_stats(labels, decoded),
    )


def _read_labels(labels_path: Path) -> dict[str, Any]:
    return json.loads(labels_path.read_text(encoding="utf-8"))


def _sequence_accuracy(expected: list[str], actual: list[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected:
        return 0.0

    compared = zip(expected, actual)
    matches = sum(1 for expected_item, actual_item in compared if expected_item == actual_item)
    return matches / max(len(expected), len(actual))


def _timing_stats(labels: dict[str, Any], decoded: DecodeResult) -> TimingStats:
    expected_events = labels["events"]
    actual_runs = decoded.classified_runs
    compared_count = min(len(expected_events), len(actual_runs))

    if compared_count == 0:
        return TimingStats(
            compared_count=0,
            count_delta=len(actual_runs) - len(expected_events),
            avg_start_error_ms=0.0,
            max_start_error_ms=0.0,
            avg_duration_error_ms=0.0,
            max_duration_error_ms=0.0,
            symbol_accuracy=0.0,
        )

    start_errors_ms: list[float] = []
    duration_errors_ms: list[float] = []
    symbol_matches = 0

    for expected, actual in zip(expected_events, actual_runs):
        start_errors_ms.append(abs(actual.start_s - float(expected["start_s"])) * 1000)
        duration_errors_ms.append(abs(actual.duration_s - float(expected["duration_s"])) * 1000)
        if actual.kind == expected["kind"] and actual.symbol == expected["symbol"]:
            symbol_matches += 1

    return TimingStats(
        compared_count=compared_count,
        count_delta=len(actual_runs) - len(expected_events),
        avg_start_error_ms=sum(start_errors_ms) / compared_count,
        max_start_error_ms=max(start_errors_ms),
        avg_duration_error_ms=sum(duration_errors_ms) / compared_count,
        max_duration_error_ms=max(duration_errors_ms),
        symbol_accuracy=symbol_matches / compared_count,
    )
