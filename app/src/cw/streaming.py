from __future__ import annotations

from pathlib import Path

import numpy as np

from cw.decoder import (
    ClassifiedRun,
    DecodeResult,
    DetectedRun,
    _to_mono_float,
    read_wav_mono,
)
from cw.stream_decode import decode_carrier_sessions_from_frames, detect_accumulated_carriers
from cw.stream_filter import filter_final_tracks
from cw.stream_state import ChannelRegistry, ChannelState
from cw.stream_models import (
    SpectrumFrame,
    StreamEvent,
    StreamingConfig,
    effective_tracker_frame_ms,
    effective_tracker_hop_ms,
    StreamSessionResult,
    StreamSimulationResult,
    StreamTrackResult,
    StreamUpdate,
    channel_match_hz,
    validate_streaming_config,
)
from cw.stream_stft import StreamingSTFT
from cw.stream_tracker import CarrierTracker

__all__ = [
    "SpectrumFrame",
    "StreamEvent",
    "StreamingConfig",
    "StreamingSTFT",
    "CarrierTracker",
    "StreamSessionResult",
    "StreamSimulationResult",
    "StreamTrackResult",
    "StreamUpdate",
    "simulate_stream",
    "simulate_stream_from_wav",
]


def simulate_stream_from_wav(path: Path, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    signal, sample_rate = read_wav_mono(path)
    return simulate_stream(signal, sample_rate, config)


def simulate_stream(signal: np.ndarray, sample_rate: int, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    validate_streaming_config(config)
    signal = _to_mono_float(signal)

    decode_stft = StreamingSTFT(sample_rate, config.frame_ms, config.hop_ms)
    tracker_stft = StreamingSTFT(
        sample_rate,
        effective_tracker_frame_ms(config),
        effective_tracker_hop_ms(config),
    )
    registry = ChannelRegistry(config)
    tracker = CarrierTracker(config)
    frames: list[SpectrumFrame] = []
    updates: list[StreamUpdate] = []
    events: list[StreamEvent] = []
    last_emit_s = 0.0
    frames_processed = 0
    tracker_frames_processed = 0
    pruned_frames = 0

    block_length = max(1, round(sample_rate * config.input_block_ms / 1000))
    for start in range(0, len(signal), block_length):
        block = signal[start : start + block_length]
        decode_frames = [("decode", frame) for frame in decode_stft.push(block)]
        tracker_frames = [("track", frame) for frame in tracker_stft.push(block)]
        frame_events = sorted(
            [*decode_frames, *tracker_frames],
            key=lambda item: (item[1].start_s, 0 if item[0] == "track" else 1),
        )
        for kind, frame in frame_events:
            if kind == "track":
                tracker.update(frame)
                tracker_frames_processed += 1
                continue

            frames.append(frame)
            frames_processed += 1
            if frame.start_s - last_emit_s < config.emit_interval_s:
                continue
            last_emit_s = frame.start_s
            new_updates = _updates_from_frames(
                frames,
                registry,
                frame.start_s,
                config,
                tracker.candidate_carriers(frame.start_s),
            )
            events.extend(registry.pop_pending_events())
            updates.extend(new_updates)
            events.extend(_events_from_updates(new_updates))
            if config.prune_finalized_sessions:
                frames, dropped_count = _prune_finalized_frames(frames, registry, config)
                pruned_frames += dropped_count

    duration_s = len(signal) / sample_rate if sample_rate else 0.0
    tracks = _final_tracks_from_frames(
        frames,
        registry,
        config,
        duration_s,
        tracker.candidate_carriers(duration_s),
    )
    events.extend(registry.pop_pending_events())
    events.extend(_final_events_from_tracks(tracks, registry, duration_s, config))
    return StreamSimulationResult(
        duration_s=duration_s,
        updates=updates,
        tracks=tracks,
        events=events,
        frames_processed=frames_processed,
        tracker_frames_processed=tracker_frames_processed,
        retained_frames=len(frames),
        pruned_frames=pruned_frames,
    )


def _prune_finalized_frames(
    frames: list[SpectrumFrame],
    registry: ChannelRegistry,
    config: StreamingConfig,
) -> tuple[list[SpectrumFrame], int]:
    prune_before_s = registry.prune_before_s()
    if prune_before_s is None:
        return frames, 0

    keep_from_s = max(0.0, prune_before_s - config.history_margin_s)
    first_keep_index = 0
    while first_keep_index < len(frames) and frames[first_keep_index].start_s < keep_from_s:
        first_keep_index += 1
    if first_keep_index <= 0:
        return frames, 0
    return frames[first_keep_index:], first_keep_index

def _updates_from_frames(
    frames: list[SpectrumFrame],
    registry: ChannelRegistry,
    time_s: float,
    config: StreamingConfig,
    carriers: list[tuple[float, float, float]] | None = None,
) -> list[StreamUpdate]:
    updates: list[StreamUpdate] = []
    candidates = carriers if carriers is not None else detect_accumulated_carriers(frames, config)
    for carrier_hz, _relative_power, _power in candidates:
        track = registry.channel_for(carrier_hz, time_s)
        sessions = decode_carrier_sessions_from_frames(frames, track.carrier_hz, config, time_s)
        active_session = registry.sync_sessions(track, sessions)
        if active_session is None or not active_session.decoded.text:
            continue
        quality = active_session.quality
        text_to_emit = track.commit_text_candidate(active_session.decoded.text, quality.score, config)
        if not text_to_emit:
            continue
        updates.append(
            StreamUpdate(
                time_s=round(time_s, 3),
                track_id=track.track_id,
                session_id=track.session_id,
                carrier_hz=round(track.carrier_hz, 3),
                score=quality.score,
                text=text_to_emit,
            )
        )
    return updates



def _events_from_updates(updates: list[StreamUpdate]) -> list[StreamEvent]:
    return [
        StreamEvent(
            time_s=update.time_s,
            kind="TEXT_COMMITTED",
            channel_id=update.track_id,
            session_id=update.session_id,
            carrier_hz=update.carrier_hz,
            text=update.text,
            score=update.score,
        )
        for update in updates
    ]


def _final_tracks_from_frames(
    frames: list[SpectrumFrame],
    registry: ChannelRegistry,
    config: StreamingConfig,
    final_time_s: float,
    carriers: list[tuple[float, float, float]] | None = None,
) -> list[StreamTrackResult]:
    active_sessions_by_track_id: dict[int, StreamSessionResult] = {}

    candidates = _merge_final_carriers(registry, carriers if carriers is not None else detect_accumulated_carriers(frames, config))
    for carrier_hz, _relative_power, _power in candidates:
        track = registry.channel_for(carrier_hz, final_time_s)
        sessions = decode_carrier_sessions_from_frames(
            frames,
            track.carrier_hz,
            config,
            final_time_s,
        )
        active_session = registry.sync_sessions(track, sessions)
        if active_session is not None and active_session.decoded.text:
            active_sessions_by_track_id[track.track_id] = active_session

    results: list[StreamTrackResult] = []
    for track in registry.channels:
        sessions = list(track.finalized_sessions)
        active_session = active_sessions_by_track_id.get(track.track_id)
        if active_session is not None and active_session.session_id not in {session.session_id for session in sessions}:
            sessions.append(active_session)
        sessions.sort(key=lambda session: session.session_id)
        if not sessions:
            continue

        decoded = _combined_decode_from_sessions(sessions, track.carrier_hz)
        representative = sessions[-1]
        first_seen_s = min(session.first_seen_s for session in sessions)
        last_seen_s = max(session.last_seen_s for session in sessions)
        hits = sum(session.hits for session in sessions)
        results.append(
            StreamTrackResult(
                track_id=track.track_id,
                carrier_hz=round(track.carrier_hz, 3),
                first_seen_s=round(first_seen_s, 3),
                last_seen_s=round(last_seen_s, 3),
                hits=hits,
                quality=representative.quality,
                decoded=decoded,
                sessions=sessions,
            )
        )
    results.sort(key=lambda result: (result.track_id, result.quality.score))
    return filter_final_tracks(results, config)


def _merge_final_carriers(
    registry: ChannelRegistry,
    carriers: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = list(carriers)
    for channel in registry.channels:
        if any(abs(channel.carrier_hz - carrier_hz) <= channel_match_hz(registry.config) for carrier_hz, _r, _p in merged):
            continue
        merged.append((channel.carrier_hz, 0.0, 0.0))
    return merged


def _combined_decode_from_sessions(sessions: list[StreamSessionResult], carrier_hz: float) -> DecodeResult:
    text = " ".join(session.decoded.text for session in sessions if session.decoded.text).strip()
    tokens: list[str] = []
    runs: list[DetectedRun] = []
    classified_runs: list[ClassifiedRun] = []
    unit_values = [session.decoded.unit_s for session in sessions if session.decoded.unit_s > 0]
    threshold_values = [session.decoded.threshold for session in sessions]
    for index, session in enumerate(sessions):
        if index > 0 and tokens and tokens[-1] != "/":
            tokens.append("/")
        tokens.extend(session.decoded.tokens)
        runs.extend(session.decoded.runs)
        classified_runs.extend(session.decoded.classified_runs)
    return DecodeResult(
        text=text,
        tokens=tokens,
        runs=runs,
        classified_runs=classified_runs,
        carrier_hz=carrier_hz,
        unit_s=unit_values[-1] if unit_values else 0.0,
        threshold=threshold_values[-1] if threshold_values else 0.0,
    )


def _final_events_from_tracks(
    tracks: list[StreamTrackResult],
    registry: ChannelRegistry,
    final_time_s: float,
    config: StreamingConfig,
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for result in tracks:
        track = registry.channel_by_id(result.track_id)
        sessions = result.sessions or [
            StreamSessionResult(
                session_id=1,
                first_seen_s=result.first_seen_s,
                last_seen_s=result.last_seen_s,
                hits=result.hits,
                final_time_s=final_time_s,
                final_reason=config.final_event_reason,
                quality=result.quality,
                decoded=result.decoded,
            )
        ]
        for session in sessions:
            if track is not None and session.session_id in track.finalized_session_ids:
                continue
            events.append(
                StreamEvent(
                    time_s=round(session.final_time_s, 3),
                    kind="SESSION_FINAL",
                    channel_id=result.track_id,
                    session_id=session.session_id,
                    carrier_hz=result.carrier_hz,
                    text=session.decoded.text,
                    score=session.quality.score,
                    reason=session.final_reason,
                )
            )
            if track is not None:
                track.finalized_session_ids.add(session.session_id)
        events.append(
            StreamEvent(
                time_s=round(final_time_s, 3),
                kind="CHANNEL_DORMANT",
                channel_id=result.track_id,
                session_id=None,
                carrier_hz=result.carrier_hz,
                text=result.decoded.text,
                score=result.quality.score,
                reason=config.final_event_reason,
            )
        )
    return events
