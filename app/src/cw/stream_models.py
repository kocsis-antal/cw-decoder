from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cw.decoder import DecodeResult
from cw.quality import QualityScore


@dataclass(frozen=True)
class StreamingConfig:
    input_block_ms: float = 10.0
    frame_ms: float = 30.0
    hop_ms: float = 5.0
    tracker_frame_ms: float | None = None
    tracker_hop_ms: float | None = None
    min_tone_hz: float = 200.0
    max_tone_hz: float = 2000.0
    bandwidth_hz: float = 40.0
    threshold_ratio: float = 0.35
    peak_relative_threshold: float = 0.25
    track_relative_threshold: float = 0.10
    min_peak_snr_db: float = 0.0
    min_keying_tone_runs: int = 0
    min_keying_chars: int = 0
    min_keying_known_chars: int = 0
    min_keying_active_duration_s: float = 0.0
    min_keying_duty_cycle: float | None = None
    max_keying_duty_cycle: float | None = None
    min_keying_unit_s: float = 0.0
    max_keying_unit_s: float | None = None
    max_keying_score: float | None = None
    reject_et_only_sessions: bool = False
    et_only_min_chars: int = 3
    merge_short_gaps_ms: float = 0.0
    drop_short_tones_ms: float = 0.0
    unit_candidate_spread: float = 0.0
    unit_candidate_steps: int = 1
    punctuation_penalty: float = 0.0
    min_separation_hz: float = 80.0
    peak_min_separation_hz: float | None = None
    track_match_hz: float | None = None
    channel_merge_hz: float | None = None
    max_tracks: int = 5
    max_track_gap_s: float = 2.0
    carrier_smoothing: float = 0.20
    min_track_hits: int = 2
    emit_interval_s: float = 0.50
    stable_updates: bool = True
    min_update_score: float = 25.0
    max_final_score: float | None = 30.0
    shadow_suppression_hz: float | None = None
    shadow_score_margin: float = 15.0
    session_gap_units: float = 20.0
    min_session_gap_s: float = 1.20
    final_event_reason: str = "end_of_stream"
    prune_finalized_sessions: bool = True
    prune_committed_active_sessions: bool = False
    history_margin_s: float = 0.25
    active_history_margin_s: float | None = None


@dataclass(frozen=True)
class StreamUpdate:
    time_s: float
    track_id: int
    session_id: int
    carrier_hz: float
    score: float
    text: str


@dataclass(frozen=True)
class StreamEvent:
    time_s: float
    kind: str
    channel_id: int
    session_id: int | None
    carrier_hz: float
    text: str = ""
    score: float = 0.0
    reason: str = ""


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
class StreamTrackResult:
    track_id: int
    carrier_hz: float
    first_seen_s: float
    last_seen_s: float
    hits: int
    quality: QualityScore
    decoded: DecodeResult
    sessions: list[StreamSessionResult] = field(default_factory=list)


@dataclass(frozen=True)
class StreamChunkResult:
    time_s: float
    updates: list[StreamUpdate] = field(default_factory=list)
    events: list[StreamEvent] = field(default_factory=list)
    frames_processed: int = 0
    tracker_frames_processed: int = 0
    retained_frames: int = 0
    pruned_frames: int = 0
    active_pruned_frames: int = 0
    finalized_pruned_frames: int = 0


@dataclass(frozen=True)
class StreamSimulationResult:
    duration_s: float
    updates: list[StreamUpdate]
    tracks: list[StreamTrackResult]
    events: list[StreamEvent]
    frames_processed: int = 0
    tracker_frames_processed: int = 0
    retained_frames: int = 0
    pruned_frames: int = 0
    active_pruned_frames: int = 0
    finalized_pruned_frames: int = 0


@dataclass(frozen=True)
class SpectrumFrame:
    start_s: float
    spectrum: np.ndarray
    freqs: np.ndarray


def effective_tracker_frame_ms(config: StreamingConfig) -> float:
    return config.tracker_frame_ms if config.tracker_frame_ms is not None else config.frame_ms


def effective_tracker_hop_ms(config: StreamingConfig) -> float:
    return config.tracker_hop_ms if config.tracker_hop_ms is not None else config.hop_ms


def peak_min_separation_hz(config: StreamingConfig) -> float:
    return config.peak_min_separation_hz or config.min_separation_hz


def track_match_hz(config: StreamingConfig) -> float:
    return config.track_match_hz or max(config.bandwidth_hz, config.min_separation_hz)


def channel_merge_hz(config: StreamingConfig) -> float:
    return config.channel_merge_hz or config.min_separation_hz


def channel_match_hz(config: StreamingConfig) -> float:
    return max(channel_merge_hz(config) / 2.0, config.bandwidth_hz)


def validate_streaming_config(config: StreamingConfig) -> None:
    if config.input_block_ms <= 0:
        raise ValueError("input_block_ms must be positive")
    if config.frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if config.hop_ms <= 0:
        raise ValueError("hop_ms must be positive")
    if config.tracker_frame_ms is not None and config.tracker_frame_ms <= 0:
        raise ValueError("tracker_frame_ms must be positive when set")
    if config.tracker_hop_ms is not None and config.tracker_hop_ms <= 0:
        raise ValueError("tracker_hop_ms must be positive when set")
    if config.min_tone_hz >= config.max_tone_hz:
        raise ValueError("min_tone_hz must be lower than max_tone_hz")
    if config.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if not 0 < config.threshold_ratio < 1:
        raise ValueError("threshold_ratio must be in the (0, 1) range")
    if not 0 < config.peak_relative_threshold <= 1:
        raise ValueError("peak_relative_threshold must be in the (0, 1] range")
    if not 0 < config.track_relative_threshold <= 1:
        raise ValueError("track_relative_threshold must be in the (0, 1] range")
    if config.min_peak_snr_db < 0:
        raise ValueError("min_peak_snr_db must not be negative")
    if config.min_keying_tone_runs < 0:
        raise ValueError("min_keying_tone_runs must not be negative")
    if config.min_keying_chars < 0:
        raise ValueError("min_keying_chars must not be negative")
    if config.min_keying_known_chars < 0:
        raise ValueError("min_keying_known_chars must not be negative")
    if config.min_keying_active_duration_s < 0:
        raise ValueError("min_keying_active_duration_s must not be negative")
    if config.min_keying_duty_cycle is not None and not 0 <= config.min_keying_duty_cycle <= 1:
        raise ValueError("min_keying_duty_cycle must be in the [0, 1] range when set")
    if config.max_keying_duty_cycle is not None and not 0 <= config.max_keying_duty_cycle <= 1:
        raise ValueError("max_keying_duty_cycle must be in the [0, 1] range when set")
    if (
        config.min_keying_duty_cycle is not None
        and config.max_keying_duty_cycle is not None
        and config.min_keying_duty_cycle > config.max_keying_duty_cycle
    ):
        raise ValueError("min_keying_duty_cycle must not be greater than max_keying_duty_cycle")
    if config.min_keying_unit_s < 0:
        raise ValueError("min_keying_unit_s must not be negative")
    if config.max_keying_unit_s is not None and config.max_keying_unit_s <= 0:
        raise ValueError("max_keying_unit_s must be positive when set")
    if config.max_keying_unit_s is not None and config.min_keying_unit_s > config.max_keying_unit_s:
        raise ValueError("min_keying_unit_s must not be greater than max_keying_unit_s")
    if config.max_keying_score is not None and config.max_keying_score <= 0:
        raise ValueError("max_keying_score must be positive when set")
    if config.et_only_min_chars < 1:
        raise ValueError("et_only_min_chars must be positive")
    if config.merge_short_gaps_ms < 0:
        raise ValueError("merge_short_gaps_ms must not be negative")
    if config.drop_short_tones_ms < 0:
        raise ValueError("drop_short_tones_ms must not be negative")
    if config.unit_candidate_spread < 0:
        raise ValueError("unit_candidate_spread must not be negative")
    if config.unit_candidate_steps < 1:
        raise ValueError("unit_candidate_steps must be positive")
    if config.punctuation_penalty < 0:
        raise ValueError("punctuation_penalty must not be negative")
    if config.min_separation_hz <= 0:
        raise ValueError("min_separation_hz must be positive")
    if config.peak_min_separation_hz is not None and config.peak_min_separation_hz <= 0:
        raise ValueError("peak_min_separation_hz must be positive when set")
    if config.track_match_hz is not None and config.track_match_hz <= 0:
        raise ValueError("track_match_hz must be positive when set")
    if config.channel_merge_hz is not None and config.channel_merge_hz <= 0:
        raise ValueError("channel_merge_hz must be positive when set")
    if config.max_tracks <= 0:
        raise ValueError("max_tracks must be positive")
    if not 0 <= config.carrier_smoothing <= 1:
        raise ValueError("carrier_smoothing must be in the [0, 1] range")
    if config.min_track_hits <= 0:
        raise ValueError("min_track_hits must be positive")
    if config.emit_interval_s <= 0:
        raise ValueError("emit_interval_s must be positive")
    if config.min_update_score <= 0:
        raise ValueError("min_update_score must be positive")
    if config.max_final_score is not None and config.max_final_score <= 0:
        raise ValueError("max_final_score must be positive when set")
    if config.shadow_suppression_hz is not None and config.shadow_suppression_hz < 0:
        raise ValueError("shadow_suppression_hz must not be negative when set")
    if config.shadow_score_margin < 0:
        raise ValueError("shadow_score_margin must not be negative")
    if config.session_gap_units <= 0:
        raise ValueError("session_gap_units must be positive")
    if config.min_session_gap_s <= 0:
        raise ValueError("min_session_gap_s must be positive")
    if config.history_margin_s < 0:
        raise ValueError("history_margin_s must not be negative")
    if config.active_history_margin_s is not None and config.active_history_margin_s < 0:
        raise ValueError("active_history_margin_s must not be negative when set")
