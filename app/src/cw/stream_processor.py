from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from cw.decoder import (
    ClassifiedRun,
    DecodeResult,
    DetectedRun,
    _to_mono_float,
)
from cw.stream_decode import decode_carrier_sessions_from_frames, detect_accumulated_carriers
from cw.stream_filter import filter_final_tracks
from cw.stream_keying import filter_keyed_sessions
from cw.stream_models import (
    SpectrumFrame,
    StreamChunkResult,
    StreamEvent,
    StreamSessionResult,
    StreamSimulationResult,
    StreamTrackResult,
    StreamUpdate,
    StreamingConfig,
    channel_match_hz,
    effective_tracker_frame_ms,
    effective_tracker_hop_ms,
    validate_streaming_config,
)
from cw.stream_state import ChannelRegistry
from cw.stream_sources import ArrayAudioSource, AudioSource, WavFileSource
from cw.stream_stft import StreamingSTFT
from cw.stream_tracker import CarrierTracker


class StreamProcessor:
    """Stateful online stream processor.

    ``simulate_stream`` is still convenient for tests and WAV replay, but this
    class is the live-facing API: feed it audio chunks with :meth:`push` and call
    :meth:`finish` when the source ends.  All channel/session state, carrier
    tracking, rolling frame history, and pruning counters live here instead of
    being hidden inside one large function.
    """

    def __init__(self, sample_rate: int, config: StreamingConfig | None = None) -> None:
        self.config = config or StreamingConfig()
        validate_streaming_config(self.config)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.sample_rate = sample_rate
        self.decode_stft = StreamingSTFT(sample_rate, self.config.frame_ms, self.config.hop_ms)
        self.tracker_stft = StreamingSTFT(
            sample_rate,
            effective_tracker_frame_ms(self.config),
            effective_tracker_hop_ms(self.config),
        )
        self.registry = ChannelRegistry(self.config)
        self.tracker = CarrierTracker(self.config)
        self.frames: list[SpectrumFrame] = []
        self.updates: list[StreamUpdate] = []
        self.events: list[StreamEvent] = []
        self.last_emit_s = 0.0
        self.frames_processed = 0
        self.tracker_frames_processed = 0
        self.pruned_frames = 0
        self.active_pruned_frames = 0
        self.finalized_pruned_frames = 0
        self.samples_processed = 0
        self._finished_result: StreamSimulationResult | None = None

    @property
    def processed_duration_s(self) -> float:
        return self.samples_processed / self.sample_rate if self.sample_rate else 0.0

    @property
    def retained_frames(self) -> int:
        return len(self.frames)

    def push(self, samples: np.ndarray) -> StreamChunkResult:
        """Feed one audio block and return only the newly emitted live output."""

        if self._finished_result is not None:
            raise RuntimeError("cannot push samples after finish()")

        samples = _to_mono_float(np.asarray(samples))
        if len(samples) == 0:
            return StreamChunkResult(time_s=round(self.processed_duration_s, 3), retained_frames=len(self.frames))

        start_update_count = len(self.updates)
        start_event_count = len(self.events)
        start_frames_processed = self.frames_processed
        start_tracker_frames_processed = self.tracker_frames_processed
        start_pruned_frames = self.pruned_frames
        start_active_pruned_frames = self.active_pruned_frames
        start_finalized_pruned_frames = self.finalized_pruned_frames

        decode_frames = [("decode", frame) for frame in self.decode_stft.push(samples)]
        tracker_frames = [("track", frame) for frame in self.tracker_stft.push(samples)]
        frame_events = sorted(
            [*decode_frames, *tracker_frames],
            key=lambda item: (item[1].start_s, 0 if item[0] == "track" else 1),
        )
        for kind, frame in frame_events:
            if kind == "track":
                self.tracker.update(frame)
                self.tracker_frames_processed += 1
                continue
            self._process_decode_frame(frame)

        self.samples_processed += len(samples)
        return StreamChunkResult(
            time_s=round(self.processed_duration_s, 3),
            updates=self.updates[start_update_count:],
            events=self.events[start_event_count:],
            frames_processed=self.frames_processed - start_frames_processed,
            tracker_frames_processed=self.tracker_frames_processed - start_tracker_frames_processed,
            retained_frames=len(self.frames),
            pruned_frames=self.pruned_frames - start_pruned_frames,
            active_pruned_frames=self.active_pruned_frames - start_active_pruned_frames,
            finalized_pruned_frames=self.finalized_pruned_frames - start_finalized_pruned_frames,
        )

    def finish(self, final_time_s: float | None = None) -> StreamSimulationResult:
        """Flush final sessions and return the accumulated stream result."""

        if self._finished_result is not None:
            return self._finished_result

        duration_s = self.processed_duration_s if final_time_s is None else final_time_s
        tracks = _final_tracks_from_frames(
            self.frames,
            self.registry,
            self.config,
            duration_s,
            self.tracker.candidate_carriers(duration_s),
        )
        self.events.extend(self.registry.pop_pending_events())
        self.events.extend(_final_events_from_tracks(tracks, self.registry, duration_s, self.config))
        self._finished_result = StreamSimulationResult(
            duration_s=duration_s,
            updates=list(self.updates),
            tracks=tracks,
            events=list(self.events),
            frames_processed=self.frames_processed,
            tracker_frames_processed=self.tracker_frames_processed,
            retained_frames=len(self.frames),
            pruned_frames=self.pruned_frames,
            active_pruned_frames=self.active_pruned_frames,
            finalized_pruned_frames=self.finalized_pruned_frames,
        )
        return self._finished_result

    def _process_decode_frame(self, frame: SpectrumFrame) -> None:
        self.frames.append(frame)
        self.frames_processed += 1
        if frame.start_s - self.last_emit_s < self.config.emit_interval_s:
            return

        self.last_emit_s = frame.start_s
        new_updates = _updates_from_frames(
            self.frames,
            self.registry,
            frame.start_s,
            self.config,
            self.tracker.candidate_carriers(frame.start_s),
        )
        self.events.extend(self.registry.pop_pending_events())
        self.updates.extend(new_updates)
        self.events.extend(_events_from_updates(new_updates))
        if self.config.prune_finalized_sessions or self.config.prune_committed_active_sessions:
            self._prune_frames()

    def _prune_frames(self) -> None:
        self.frames, dropped_count, prune_reason = _prune_stream_frames(self.frames, self.registry, self.config)
        self.pruned_frames += dropped_count
        if prune_reason == "active":
            self.active_pruned_frames += dropped_count
        elif prune_reason == "finalized":
            self.finalized_pruned_frames += dropped_count


def simulate_stream_from_wav(path: Path, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    return process_audio_source(WavFileSource(path, config.input_block_ms), config)


def simulate_stream(signal: np.ndarray, sample_rate: int, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    return process_audio_source(ArrayAudioSource(signal, sample_rate, config.input_block_ms), config)


def process_audio_source(
    source: AudioSource,
    config: StreamingConfig | None = None,
    on_chunk: Callable[[StreamChunkResult], None] | None = None,
) -> StreamSimulationResult:
    """Feed any block-based audio source into ``StreamProcessor``.

    ``on_chunk`` is called after each push with only the newly emitted updates
    and events.  This keeps replay, future microphone input, and test fixtures on
    the same path while still returning the final accumulated result at EOF.
    """

    config = config or StreamingConfig()
    processor = StreamProcessor(source.sample_rate, config)

    for block in source:
        if block.sample_rate != source.sample_rate:
            raise ValueError("audio block sample_rate changed during stream")
        chunk = processor.push(block.samples)
        if on_chunk is not None:
            on_chunk(chunk)

    return processor.finish(final_time_s=source.duration_s)


def _prune_stream_frames(
    frames: list[SpectrumFrame],
    registry: ChannelRegistry,
    config: StreamingConfig,
) -> tuple[list[SpectrumFrame], int, str | None]:
    cutoff = registry.prune_cutoff_s()
    if cutoff is None:
        return frames, 0, None

    prune_before_s, reason = cutoff
    if reason == "active" and not config.prune_committed_active_sessions:
        return frames, 0, None
    if reason == "finalized" and not config.prune_finalized_sessions:
        return frames, 0, None

    margin_s = config.history_margin_s
    if reason == "active" and config.active_history_margin_s is not None:
        margin_s = config.active_history_margin_s

    keep_from_s = max(0.0, prune_before_s - margin_s)
    first_keep_index = 0
    while first_keep_index < len(frames) and frames[first_keep_index].start_s < keep_from_s:
        first_keep_index += 1
    if first_keep_index <= 0:
        return frames, 0, reason
    return frames[first_keep_index:], first_keep_index, reason


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
        sessions = filter_keyed_sessions(
            decode_carrier_sessions_from_frames(frames, carrier_hz, config, time_s),
            config,
        )
        if not sessions:
            continue
        track = registry.channel_for(carrier_hz, sessions[0].first_seen_s)
        if abs(track.carrier_hz - carrier_hz) > 1e-9:
            sessions = filter_keyed_sessions(
                decode_carrier_sessions_from_frames(frames, track.carrier_hz, config, time_s),
                config,
            )
            if not sessions:
                continue
        active_session = registry.sync_sessions(track, sessions)
        if active_session is None or not active_session.decoded.text:
            continue
        quality = active_session.quality
        text_to_emit = track.commit_session_candidate(active_session, config)
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
        sessions = filter_keyed_sessions(
            decode_carrier_sessions_from_frames(
                frames,
                carrier_hz,
                config,
                final_time_s,
            ),
            config,
        )
        if not sessions:
            continue
        track = registry.channel_for(carrier_hz, sessions[0].first_seen_s)
        if abs(track.carrier_hz - carrier_hz) > 1e-9:
            sessions = filter_keyed_sessions(
                decode_carrier_sessions_from_frames(
                    frames,
                    track.carrier_hz,
                    config,
                    final_time_s,
                ),
                config,
            )
            if not sessions:
                continue
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
