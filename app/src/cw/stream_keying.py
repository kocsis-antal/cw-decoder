from __future__ import annotations

from dataclasses import dataclass

from cw.decoder import DecodeResult, DetectedRun
from cw.stream_models import StreamSessionResult, StreamingConfig


@dataclass(frozen=True)
class KeyingMetrics:
    """Small, deterministic CW-like activity summary for a decoded session.

    Carrier SNR tells us whether there is a spectral peak.  It does not tell us
    whether the peak is actually keyed like CW.  These metrics are intentionally
    based on the already detected tone/gap runs so they can be used both in WAV
    replay and live stdin without keeping additional audio history.
    """

    text: str
    text_chars: int
    known_chars: int
    tone_runs: int
    gap_runs: int
    active_duration_s: float
    span_s: float
    duty_cycle: float
    score: float
    unit_s: float
    et_only: bool


def keying_metrics(session: StreamSessionResult) -> KeyingMetrics:
    decoded = session.decoded
    runs = decoded.runs
    tone_runs = [run for run in runs if run.kind == "tone"]
    gap_runs = [run for run in runs if run.kind == "gap"]
    active_duration_s = sum(run.duration_s for run in tone_runs)
    span_s = _tone_span_s(tone_runs)
    duty_cycle = active_duration_s / span_s if span_s > 0 else 0.0
    chars = [char for char in decoded.text.replace(" ", "") if char]
    known_chars = [char for char in chars if char != "?"]
    known_alpha = {char.upper() for char in known_chars if char.isalpha()}
    et_only = bool(known_alpha) and known_alpha <= {"E", "T"}
    return KeyingMetrics(
        text=decoded.text,
        text_chars=len(chars),
        known_chars=len(known_chars),
        tone_runs=len(tone_runs),
        gap_runs=len(gap_runs),
        active_duration_s=active_duration_s,
        span_s=span_s,
        duty_cycle=duty_cycle,
        score=session.quality.score,
        unit_s=decoded.unit_s,
        et_only=et_only,
    )


def passes_keying_gate(session: StreamSessionResult, config: StreamingConfig) -> bool:
    """Return True when a session looks like keyed CW, not just a noise peak.

    The gate is disabled by default for laboratory replay because old tests and
    experiments should remain byte-for-byte permissive.  ``stream-stdin`` enables
    it through safer defaults.
    """

    metrics = keying_metrics(session)
    if metrics.tone_runs < config.min_keying_tone_runs:
        return False
    if metrics.text_chars < config.min_keying_chars:
        return False
    if metrics.known_chars < config.min_keying_known_chars:
        return False
    if metrics.active_duration_s < config.min_keying_active_duration_s:
        return False
    if metrics.unit_s < config.min_keying_unit_s:
        return False
    if config.max_keying_unit_s is not None and metrics.unit_s > config.max_keying_unit_s:
        return False
    if config.max_keying_score is not None and metrics.score > config.max_keying_score:
        return False
    if config.min_keying_duty_cycle is not None and metrics.duty_cycle < config.min_keying_duty_cycle:
        return False
    if config.max_keying_duty_cycle is not None and metrics.duty_cycle > config.max_keying_duty_cycle:
        return False
    if config.reject_et_only_sessions and metrics.et_only and metrics.text_chars >= config.et_only_min_chars:
        return False
    return True


def filter_keyed_sessions(
    sessions: list[StreamSessionResult],
    config: StreamingConfig,
) -> list[StreamSessionResult]:
    if not keying_gate_enabled(config):
        return sessions
    return [session for session in sessions if passes_keying_gate(session, config)]


def keying_gate_enabled(config: StreamingConfig) -> bool:
    return any(
        [
            config.min_keying_tone_runs > 0,
            config.min_keying_chars > 0,
            config.min_keying_known_chars > 0,
            config.min_keying_active_duration_s > 0,
            config.min_keying_unit_s > 0,
            config.max_keying_unit_s is not None,
            config.max_keying_score is not None,
            config.min_keying_duty_cycle is not None,
            config.max_keying_duty_cycle is not None,
            config.reject_et_only_sessions,
        ]
    )


def _tone_span_s(tone_runs: list[DetectedRun]) -> float:
    if not tone_runs:
        return 0.0
    start_s = tone_runs[0].start_s
    end_s = tone_runs[-1].start_s + tone_runs[-1].duration_s
    return max(0.0, end_s - start_s)
