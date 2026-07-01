from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from cw.decoder import (
    ClassifiedRun,
    DecodeResult,
    DecoderConfig,
    DetectedRun,
    _classified_runs_to_tokens,
    _energy_threshold,
    _runs_from_activity,
    _to_mono_float,
    classify_runs,
    read_wav_mono,
)
from cw.morse_table import decode_tokens
from cw.multi_decoder import _local_peak_indices
from cw.quality import QualityScore, score_decode_result


@dataclass(frozen=True)
class StreamingConfig:
    input_block_ms: float = 10.0
    frame_ms: float = 30.0
    hop_ms: float = 5.0
    min_tone_hz: float = 200.0
    max_tone_hz: float = 2000.0
    bandwidth_hz: float = 40.0
    threshold_ratio: float = 0.35
    peak_relative_threshold: float = 0.25
    min_separation_hz: float = 80.0
    max_tracks: int = 5
    max_track_gap_s: float = 2.0
    carrier_smoothing: float = 0.20
    min_track_hits: int = 2
    emit_interval_s: float = 0.50
    stable_updates: bool = True
    min_update_score: float = 25.0
    session_gap_units: float = 20.0
    min_session_gap_s: float = 1.20
    final_event_reason: str = "end_of_stream"
    prune_finalized_sessions: bool = True
    history_margin_s: float = 0.25


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
class StreamSimulationResult:
    duration_s: float
    updates: list[StreamUpdate]
    tracks: list[StreamTrackResult]
    events: list[StreamEvent]
    frames_processed: int = 0
    retained_frames: int = 0
    pruned_frames: int = 0


@dataclass(frozen=True)
class SpectrumFrame:
    start_s: float
    spectrum: np.ndarray
    freqs: np.ndarray


@dataclass
class _TrackState:
    track_id: int
    carrier_hz: float
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    total_peak_power: float = 0.0
    times: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    last_emitted_text: str = ""
    last_emit_s: float = 0.0


class StreamingSTFT:
    def __init__(self, sample_rate: int, frame_ms: float, hop_ms: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if hop_ms <= 0:
            raise ValueError("hop_ms must be positive")

        self.sample_rate = sample_rate
        self.frame_length = max(1, round(sample_rate * frame_ms / 1000))
        self.hop_length = max(1, round(sample_rate * hop_ms / 1000))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._next_frame_start_sample = 0
        self._window = np.hanning(self.frame_length).astype(np.float32)
        self._freqs = np.fft.rfftfreq(self.frame_length, 1 / sample_rate)

    def push(self, samples: np.ndarray) -> list[SpectrumFrame]:
        samples = _to_mono_float(np.asarray(samples))
        if len(samples) == 0:
            return []

        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        frames: list[SpectrumFrame] = []

        while self._next_frame_start_sample + self.frame_length <= self._buffer_start_sample + len(self._buffer):
            offset = self._next_frame_start_sample - self._buffer_start_sample
            frame = self._buffer[offset : offset + self.frame_length]
            spectrum = np.abs(np.fft.rfft(frame * self._window)) ** 2
            frames.append(
                SpectrumFrame(
                    start_s=self._next_frame_start_sample / self.sample_rate,
                    spectrum=spectrum.astype(np.float32, copy=False),
                    freqs=self._freqs,
                )
            )
            self._next_frame_start_sample += self.hop_length

        self._drop_obsolete_samples()
        return frames

    def _drop_obsolete_samples(self) -> None:
        drop_count = self._next_frame_start_sample - self._buffer_start_sample
        if drop_count <= 0:
            return
        self._buffer = self._buffer[drop_count:]
        self._buffer_start_sample += drop_count


class CarrierTracker:
    def __init__(self, config: StreamingConfig) -> None:
        _validate_streaming_config(config)
        self.config = config
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> list[_TrackState]:
        return self._tracks

    def process_frame(self, frame: SpectrumFrame) -> None:
        peaks = self._find_peaks(frame)
        self._append_track_energies(frame)
        self._assign_peaks(frame.start_s, peaks)

    def _find_peaks(self, frame: SpectrumFrame) -> list[tuple[float, float]]:
        freqs = frame.freqs
        spectrum = frame.spectrum
        search_mask = (freqs >= self.config.min_tone_hz) & (freqs <= self.config.max_tone_hz)
        if not np.any(search_mask):
            return []

        search_freqs = freqs[search_mask]
        powers = spectrum[search_mask]
        if len(powers) == 0:
            return []

        max_power = float(np.max(powers))
        if max_power <= 0:
            return []

        peak_indices = _local_peak_indices(powers)
        peak_indices.sort(key=lambda index: float(powers[index]), reverse=True)

        selected: list[tuple[float, float]] = []
        for index in peak_indices:
            power = float(powers[index])
            if power < max_power * self.config.peak_relative_threshold:
                continue
            frequency_hz = float(search_freqs[index])
            if any(abs(frequency_hz - existing_hz) < self.config.min_separation_hz for existing_hz, _p in selected):
                continue
            selected.append((frequency_hz, power))
            if len(selected) >= self.config.max_tracks:
                break
        return selected

    def _append_track_energies(self, frame: SpectrumFrame) -> None:
        for track in self._tracks:
            track.times.append(frame.start_s)
            track.energies.append(_band_energy(frame.spectrum, frame.freqs, track.carrier_hz, self.config.bandwidth_hz))

    def _assign_peaks(self, time_s: float, peaks: list[tuple[float, float]]) -> None:
        matched_track_ids: set[int] = set()
        max_match_hz = max(self.config.min_separation_hz / 2, self.config.bandwidth_hz)

        for frequency_hz, power in peaks:
            track = self._nearest_track(frequency_hz, matched_track_ids, max_match_hz)
            if track is None:
                self._create_track(time_s, frequency_hz, power)
                continue

            matched_track_ids.add(track.track_id)
            smoothing = self.config.carrier_smoothing
            track.carrier_hz = (1 - smoothing) * track.carrier_hz + smoothing * frequency_hz
            track.last_seen_s = time_s
            track.hits += 1
            track.total_peak_power += power

    def _nearest_track(self, frequency_hz: float, used_track_ids: set[int], max_match_hz: float) -> _TrackState | None:
        candidates = [
            track
            for track in self._tracks
            if track.track_id not in used_track_ids
            and abs(track.carrier_hz - frequency_hz) <= max_match_hz
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda track: abs(track.carrier_hz - frequency_hz))

    def _create_track(self, time_s: float, frequency_hz: float, power: float) -> None:
        if len(self._tracks) >= self.config.max_tracks:
            return
        self._tracks.append(
            _TrackState(
                track_id=self._next_track_id,
                carrier_hz=frequency_hz,
                first_seen_s=time_s,
                last_seen_s=time_s,
                hits=1,
                total_peak_power=power,
                times=[time_s],
                energies=[power],
                last_emit_s=time_s,
            )
        )
        self._next_track_id += 1


def simulate_stream_from_wav(path: Path, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    signal, sample_rate = read_wav_mono(path)
    return simulate_stream(signal, sample_rate, config)


def simulate_stream(signal: np.ndarray, sample_rate: int, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    _validate_streaming_config(config)
    signal = _to_mono_float(signal)

    stft = StreamingSTFT(sample_rate, config.frame_ms, config.hop_ms)
    registry = _TrackRegistry(config)
    frames: list[SpectrumFrame] = []
    updates: list[StreamUpdate] = []
    events: list[StreamEvent] = []
    last_emit_s = 0.0
    frames_processed = 0
    pruned_frames = 0

    block_length = max(1, round(sample_rate * config.input_block_ms / 1000))
    for start in range(0, len(signal), block_length):
        block = signal[start : start + block_length]
        for frame in stft.push(block):
            frames.append(frame)
            frames_processed += 1
            if frame.start_s - last_emit_s < config.emit_interval_s:
                continue
            last_emit_s = frame.start_s
            new_updates = _updates_from_frames(frames, registry, frame.start_s, config)
            events.extend(registry.pop_pending_events())
            updates.extend(new_updates)
            events.extend(_events_from_updates(new_updates))
            if config.prune_finalized_sessions:
                frames, dropped_count = _prune_finalized_frames(frames, registry, config)
                pruned_frames += dropped_count

    duration_s = len(signal) / sample_rate if sample_rate else 0.0
    tracks = _final_tracks_from_frames(frames, registry, config, duration_s)
    events.extend(registry.pop_pending_events())
    events.extend(_final_events_from_tracks(tracks, registry, duration_s, config))
    return StreamSimulationResult(
        duration_s=duration_s,
        updates=updates,
        tracks=tracks,
        events=events,
        frames_processed=frames_processed,
        retained_frames=len(frames),
        pruned_frames=pruned_frames,
    )


@dataclass
class _RegisteredTrack:
    track_id: int
    carrier_hz: float
    session_id: int = 1
    last_candidate_text: str = ""
    last_emitted_text: str = ""
    finalized_session_ids: set[int] = field(default_factory=set)
    started_session_ids: set[int] = field(default_factory=set)
    finalized_sessions: list[StreamSessionResult] = field(default_factory=list)
    active_session_first_seen_s: float | None = None


class _TrackRegistry:
    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self._next_track_id = 1
        self._tracks: list[_RegisteredTrack] = []
        self._pending_events: list[StreamEvent] = []

    def track_for(self, carrier_hz: float, time_s: float = 0.0) -> _RegisteredTrack:
        max_match_hz = max(self.config.min_separation_hz / 2, self.config.bandwidth_hz)
        candidates = [track for track in self._tracks if abs(track.carrier_hz - carrier_hz) <= max_match_hz]
        if candidates:
            track = min(candidates, key=lambda existing: abs(existing.carrier_hz - carrier_hz))
            smoothing = self.config.carrier_smoothing
            track.carrier_hz = (1 - smoothing) * track.carrier_hz + smoothing * carrier_hz
            return track

        track = _RegisteredTrack(track_id=self._next_track_id, carrier_hz=carrier_hz, started_session_ids={1})
        self._tracks.append(track)
        self._next_track_id += 1
        self._pending_events.extend(
            [
                StreamEvent(
                    time_s=round(time_s, 3),
                    kind="CHANNEL_STARTED",
                    channel_id=track.track_id,
                    session_id=None,
                    carrier_hz=round(carrier_hz, 3),
                ),
                StreamEvent(
                    time_s=round(time_s, 3),
                    kind="SESSION_STARTED",
                    channel_id=track.track_id,
                    session_id=track.session_id,
                    carrier_hz=round(carrier_hz, 3),
                ),
            ]
        )
        return track

    @property
    def tracks(self) -> list[_RegisteredTrack]:
        return self._tracks

    def sync_sessions(self, track: _RegisteredTrack, sessions: list[StreamSessionResult]) -> StreamSessionResult | None:
        active_session: StreamSessionResult | None = None
        track.active_session_first_seen_s = None

        for session in sessions:
            existing = self._matching_finalized_session(track, session)
            if existing is not None:
                track.session_id = max(track.session_id, existing.session_id + 1)
                continue

            session = self._with_current_session_id(track, session)
            self._ensure_session_started(track, session)

            if session.final_reason == "silence_gap":
                self._finalize_session(track, session)
            elif session.final_reason == self.config.final_event_reason:
                active_session = session
                track.active_session_first_seen_s = session.first_seen_s

        return active_session

    def _with_current_session_id(
        self, track: _RegisteredTrack, session: StreamSessionResult
    ) -> StreamSessionResult:
        return replace(session, session_id=track.session_id)

    def _ensure_session_started(self, track: _RegisteredTrack, session: StreamSessionResult) -> None:
        if session.session_id in track.started_session_ids:
            return
        self._pending_events.append(
            StreamEvent(
                time_s=round(session.first_seen_s, 3),
                kind="SESSION_STARTED",
                channel_id=track.track_id,
                session_id=session.session_id,
                carrier_hz=round(track.carrier_hz, 3),
            )
        )
        track.started_session_ids.add(session.session_id)
        track.last_candidate_text = ""
        track.last_emitted_text = ""

    def _matching_finalized_session(
        self, track: _RegisteredTrack, session: StreamSessionResult
    ) -> StreamSessionResult | None:
        tolerance_s = max(self.config.hop_ms / 1000 * 3, 0.03)
        for existing in track.finalized_sessions:
            if (
                abs(existing.first_seen_s - session.first_seen_s) <= tolerance_s
                and abs(existing.last_seen_s - session.last_seen_s) <= tolerance_s
                and existing.decoded.text == session.decoded.text
            ):
                return existing
        return None

    def _finalize_session(self, track: _RegisteredTrack, session: StreamSessionResult) -> None:
        if session.session_id in track.finalized_session_ids:
            return
        self._pending_events.append(
            StreamEvent(
                time_s=round(session.final_time_s, 3),
                kind="SESSION_FINAL",
                channel_id=track.track_id,
                session_id=session.session_id,
                carrier_hz=round(track.carrier_hz, 3),
                text=session.decoded.text,
                score=session.quality.score,
                reason=session.final_reason,
            )
        )
        track.finalized_session_ids.add(session.session_id)
        track.finalized_sessions.append(session)
        track.finalized_sessions.sort(key=lambda item: item.session_id)
        if track.session_id <= session.session_id:
            track.session_id = session.session_id + 1
            track.active_session_first_seen_s = None
            track.last_candidate_text = ""
            track.last_emitted_text = ""

    def prune_before_s(self) -> float | None:
        active_starts = [
            track.active_session_first_seen_s
            for track in self._tracks
            if track.active_session_first_seen_s is not None
        ]
        if active_starts:
            return min(active_starts)

        finalized_times = [
            session.final_time_s
            for track in self._tracks
            for session in track.finalized_sessions
        ]
        if finalized_times:
            return max(finalized_times)
        return None

    def track_by_id(self, track_id: int) -> _RegisteredTrack | None:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def pop_pending_events(self) -> list[StreamEvent]:
        events = self._pending_events
        self._pending_events = []
        return events



def _prune_finalized_frames(
    frames: list[SpectrumFrame],
    registry: _TrackRegistry,
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
    registry: _TrackRegistry,
    time_s: float,
    config: StreamingConfig,
) -> list[StreamUpdate]:
    updates: list[StreamUpdate] = []
    for carrier_hz, _relative_power, _power in _detect_accumulated_carriers(frames, config):
        track = registry.track_for(carrier_hz, time_s)
        sessions = _decode_carrier_sessions_from_frames(frames, track.carrier_hz, config, time_s)
        active_session = registry.sync_sessions(track, sessions)
        if active_session is None or not active_session.decoded.text:
            continue
        quality = active_session.quality
        text_to_emit = _text_to_emit(track, active_session.decoded.text, quality.score, config)
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


def _text_to_emit(
    track: _RegisteredTrack,
    current_text: str,
    score: float,
    config: StreamingConfig,
) -> str | None:
    if not config.stable_updates:
        if current_text == track.last_emitted_text:
            track.last_candidate_text = current_text
            return None
        track.last_candidate_text = current_text
        track.last_emitted_text = current_text
        return current_text

    if score > config.min_update_score:
        return None

    previous_text = track.last_candidate_text
    track.last_candidate_text = current_text
    if not previous_text:
        return None

    stable_prefix = _common_text_prefix(previous_text, current_text).rstrip()
    if not stable_prefix:
        return None
    if not stable_prefix.startswith(track.last_emitted_text):
        return None
    if len(stable_prefix) <= len(track.last_emitted_text):
        return None

    track.last_emitted_text = stable_prefix
    return stable_prefix


def _common_text_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def _final_tracks_from_frames(
    frames: list[SpectrumFrame],
    registry: _TrackRegistry,
    config: StreamingConfig,
    final_time_s: float,
) -> list[StreamTrackResult]:
    active_sessions_by_track_id: dict[int, StreamSessionResult] = {}

    for carrier_hz, _relative_power, _power in _detect_accumulated_carriers(frames, config):
        track = registry.track_for(carrier_hz, final_time_s)
        sessions = _decode_carrier_sessions_from_frames(
            frames,
            track.carrier_hz,
            config,
            final_time_s,
        )
        active_session = registry.sync_sessions(track, sessions)
        if active_session is not None and active_session.decoded.text:
            active_sessions_by_track_id[track.track_id] = active_session

    results: list[StreamTrackResult] = []
    for track in registry.tracks:
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
    return results


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
    registry: _TrackRegistry,
    final_time_s: float,
    config: StreamingConfig,
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for result in tracks:
        track = registry.track_by_id(result.track_id)
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


def _detect_accumulated_carriers(
    frames: list[SpectrumFrame],
    config: StreamingConfig,
) -> list[tuple[float, float, float]]:
    if not frames:
        return []

    freqs = frames[-1].freqs
    summed = np.sum([frame.spectrum for frame in frames], axis=0)
    search_mask = (freqs >= config.min_tone_hz) & (freqs <= config.max_tone_hz)
    if not np.any(search_mask):
        return []

    search_freqs = freqs[search_mask]
    powers = summed[search_mask]
    if len(powers) == 0:
        return []
    max_power = float(np.max(powers))
    if max_power <= 0:
        return []

    candidates = _local_peak_indices(powers)
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)

    selected: list[tuple[float, float, float]] = []
    for index in candidates:
        power = float(powers[index])
        relative_power = power / max_power
        if relative_power < config.peak_relative_threshold:
            continue
        frequency_hz = float(search_freqs[index])
        if any(abs(frequency_hz - existing_hz) < config.min_separation_hz for existing_hz, _r, _p in selected):
            continue
        selected.append((frequency_hz, relative_power, power))
        if len(selected) >= config.max_tracks:
            break
    return selected


def _decode_carrier_from_frames(
    frames: list[SpectrumFrame],
    carrier_hz: float,
    config: StreamingConfig,
) -> tuple[DecodeResult, int, float, float]:
    if not frames:
        return _empty_decode(carrier_hz), 0, 0.0, 0.0

    energy = np.asarray(
        [_band_energy(frame.spectrum, frame.freqs, carrier_hz, config.bandwidth_hz) for frame in frames],
        dtype=np.float32,
    )
    if len(energy) == 0 or float(np.max(energy)) <= 0:
        return _empty_decode(carrier_hz), 0, 0.0, 0.0

    decoder_config = DecoderConfig(
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
        min_tone_hz=config.min_tone_hz,
        max_tone_hz=config.max_tone_hz,
        bandwidth_hz=config.bandwidth_hz,
        threshold_ratio=config.threshold_ratio,
        target_tone_hz=carrier_hz,
    )
    threshold = _energy_threshold(energy, decoder_config)
    active = energy > threshold
    active_count = int(np.sum(active))
    runs = _offset_runs(_runs_from_activity(active, config.hop_ms / 1000), frames[0].start_s)
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return _empty_decode(carrier_hz, threshold=threshold, runs=runs), active_count, 0.0, 0.0

    classified_runs = classify_runs(runs, unit_s)
    tokens = _classified_runs_to_tokens(classified_runs)
    text = decode_tokens(tokens)
    first_seen_s, last_seen_s = _active_time_bounds(runs)
    return (
        DecodeResult(
            text=text,
            tokens=tokens,
            runs=runs,
            classified_runs=classified_runs,
            carrier_hz=carrier_hz,
            unit_s=unit_s,
            threshold=threshold,
        ),
        active_count,
        first_seen_s,
        last_seen_s,
    )


def _decode_carrier_sessions_from_frames(
    frames: list[SpectrumFrame],
    carrier_hz: float,
    config: StreamingConfig,
    final_time_s: float,
) -> list[StreamSessionResult]:
    if not frames:
        return []

    energy = np.asarray(
        [_band_energy(frame.spectrum, frame.freqs, carrier_hz, config.bandwidth_hz) for frame in frames],
        dtype=np.float32,
    )
    if len(energy) == 0 or float(np.max(energy)) <= 0:
        return []

    decoder_config = DecoderConfig(
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
        min_tone_hz=config.min_tone_hz,
        max_tone_hz=config.max_tone_hz,
        bandwidth_hz=config.bandwidth_hz,
        threshold_ratio=config.threshold_ratio,
        target_tone_hz=carrier_hz,
    )
    threshold = _energy_threshold(energy, decoder_config)
    active = energy > threshold
    runs = _offset_runs(_runs_from_activity(active, config.hop_ms / 1000), frames[0].start_s)
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return []

    gap_threshold_s = max(config.min_session_gap_s, config.session_gap_units * unit_s)
    segments = _split_runs_by_session_gap(runs, gap_threshold_s, final_time_s)
    sessions: list[StreamSessionResult] = []
    for index, (segment_runs, session_final_time_s, reason) in enumerate(segments, start=1):
        decoded = _decode_run_segment(segment_runs, carrier_hz, threshold)
        if not decoded.text:
            continue
        quality = score_decode_result(decoded)
        first_seen_s, last_seen_s = _active_time_bounds(segment_runs)
        hits = _tone_hit_count(segment_runs, config.hop_ms / 1000)
        sessions.append(
            StreamSessionResult(
                session_id=index,
                first_seen_s=round(first_seen_s, 3),
                last_seen_s=round(last_seen_s, 3),
                hits=hits,
                final_time_s=round(session_final_time_s, 3),
                final_reason=reason,
                quality=quality,
                decoded=decoded,
            )
        )
    return sessions



def _offset_runs(runs: list[DetectedRun], offset_s: float) -> list[DetectedRun]:
    if not runs or abs(offset_s) < 1e-12:
        return runs
    return [
        DetectedRun(
            kind=run.kind,
            start_s=run.start_s + offset_s,
            duration_s=run.duration_s,
        )
        for run in runs
    ]

def _split_runs_by_session_gap(
    runs: list[DetectedRun],
    gap_threshold_s: float,
    final_time_s: float,
) -> list[tuple[list[DetectedRun], float, str]]:
    segments: list[tuple[list[DetectedRun], float, str]] = []
    current: list[DetectedRun] = []

    for run in runs:
        if run.kind == "gap" and run.duration_s >= gap_threshold_s and _has_tone(current):
            segments.append((current, run.start_s + gap_threshold_s, "silence_gap"))
            current = []
            continue
        if current or run.kind == "tone":
            current.append(run)

    if _has_tone(current):
        segments.append((current, final_time_s, "end_of_stream"))
    return segments


def _decode_run_segment(
    runs: list[DetectedRun],
    carrier_hz: float,
    threshold: float,
) -> DecodeResult:
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return _empty_decode(carrier_hz, threshold=threshold, runs=runs)
    classified_runs = classify_runs(runs, unit_s)
    tokens = _classified_runs_to_tokens(classified_runs)
    text = decode_tokens(tokens)
    return DecodeResult(
        text=text,
        tokens=tokens,
        runs=runs,
        classified_runs=classified_runs,
        carrier_hz=carrier_hz,
        unit_s=unit_s,
        threshold=threshold,
    )


def _has_tone(runs: list[DetectedRun]) -> bool:
    return any(run.kind == "tone" for run in runs)


def _tone_hit_count(runs: list[DetectedRun], hop_s: float) -> int:
    if hop_s <= 0:
        return sum(1 for run in runs if run.kind == "tone")
    return sum(max(1, round(run.duration_s / hop_s)) for run in runs if run.kind == "tone")


def _active_time_bounds(runs: list[DetectedRun]) -> tuple[float, float]:
    tones = [run for run in runs if run.kind == "tone"]
    if not tones:
        return 0.0, 0.0
    return tones[0].start_s, tones[-1].start_s + tones[-1].duration_s


def _maybe_emit_updates(tracks: list[_TrackState], time_s: float, config: StreamingConfig) -> list[StreamUpdate]:
    updates: list[StreamUpdate] = []
    for track in tracks:
        if track.hits < config.min_track_hits:
            continue
        if time_s - track.last_emit_s < config.emit_interval_s:
            continue
        decoded = _decode_track(track, config)
        if not decoded.text or decoded.text == track.last_emitted_text:
            track.last_emit_s = time_s
            continue
        quality = score_decode_result(decoded)
        updates.append(
            StreamUpdate(
                time_s=round(time_s, 3),
                track_id=track.track_id,
                session_id=1,
                carrier_hz=round(track.carrier_hz, 3),
                score=quality.score,
                text=decoded.text,
            )
        )
        track.last_emitted_text = decoded.text
        track.last_emit_s = time_s
    return updates


def _finalize_tracks(tracks: list[_TrackState], config: StreamingConfig) -> list[StreamTrackResult]:
    results: list[StreamTrackResult] = []
    for track in tracks:
        if track.hits < config.min_track_hits:
            continue
        decoded = _decode_track(track, config)
        if not decoded.text:
            continue
        quality = score_decode_result(decoded)
        results.append(
            StreamTrackResult(
                track_id=track.track_id,
                carrier_hz=round(track.carrier_hz, 3),
                first_seen_s=round(track.first_seen_s, 3),
                last_seen_s=round(track.last_seen_s, 3),
                hits=track.hits,
                quality=quality,
                decoded=decoded,
            )
        )
    results.sort(key=lambda result: (-result.hits, result.quality.score, result.track_id))
    return results


def _decode_track(track: _TrackState, config: StreamingConfig) -> DecodeResult:
    energy = np.asarray(track.energies, dtype=np.float32)
    if len(energy) == 0 or float(np.max(energy)) <= 0:
        return _empty_decode(track.carrier_hz)

    decoder_config = DecoderConfig(
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
        min_tone_hz=config.min_tone_hz,
        max_tone_hz=config.max_tone_hz,
        bandwidth_hz=config.bandwidth_hz,
        threshold_ratio=config.threshold_ratio,
        target_tone_hz=track.carrier_hz,
    )
    threshold = _energy_threshold(energy, decoder_config)
    active = energy > threshold
    runs = _runs_from_activity(active, config.hop_ms / 1000)
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return _empty_decode(track.carrier_hz, threshold=threshold, runs=runs)

    classified_runs = classify_runs(runs, unit_s)
    tokens = _classified_runs_to_tokens(classified_runs)
    text = decode_tokens(tokens)
    return DecodeResult(
        text=text,
        tokens=tokens,
        runs=runs,
        classified_runs=classified_runs,
        carrier_hz=track.carrier_hz,
        unit_s=unit_s,
        threshold=threshold,
    )


def _estimate_unit_from_runs(runs: list[DetectedRun]) -> float:
    from cw.decoder import _estimate_unit_s

    return _estimate_unit_s(runs)


def _empty_decode(
    carrier_hz: float,
    *,
    threshold: float = 0.0,
    runs: list[DetectedRun] | None = None,
) -> DecodeResult:
    return DecodeResult(
        text="",
        tokens=[],
        runs=runs or [],
        classified_runs=[],
        carrier_hz=carrier_hz,
        unit_s=0.0,
        threshold=threshold,
    )


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray, carrier_hz: float, bandwidth_hz: float) -> float:
    mask = np.abs(freqs - carrier_hz) <= bandwidth_hz
    if not np.any(mask):
        mask[np.argmin(np.abs(freqs - carrier_hz))] = True
    return float(spectrum[mask].sum())


def _validate_streaming_config(config: StreamingConfig) -> None:
    if config.input_block_ms <= 0:
        raise ValueError("input_block_ms must be positive")
    if config.frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if config.hop_ms <= 0:
        raise ValueError("hop_ms must be positive")
    if config.min_tone_hz >= config.max_tone_hz:
        raise ValueError("min_tone_hz must be lower than max_tone_hz")
    if config.bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    if not 0 < config.threshold_ratio < 1:
        raise ValueError("threshold_ratio must be in the (0, 1) range")
    if not 0 < config.peak_relative_threshold <= 1:
        raise ValueError("peak_relative_threshold must be in the (0, 1] range")
    if config.min_separation_hz <= 0:
        raise ValueError("min_separation_hz must be positive")
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
    if config.session_gap_units <= 0:
        raise ValueError("session_gap_units must be positive")
    if config.min_session_gap_s <= 0:
        raise ValueError("min_session_gap_s must be positive")
    if config.history_margin_s < 0:
        raise ValueError("history_margin_s must not be negative")
