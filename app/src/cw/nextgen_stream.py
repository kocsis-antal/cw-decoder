from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cw.nextgen import NextgenCarrierResult, NextgenSession, _detect_carriers_nextgen, decode_signal_carrier_nextgen
from cw.stream_models import StreamChunkResult, StreamEvent, StreamingConfig, channel_match_hz, effective_tracker_frame_ms, effective_tracker_hop_ms


@dataclass
class _LiveSessionState:
    session_id: int
    start_s: float
    end_s: float
    last_candidate_text: str = ""
    committed_text: str = ""
    committed_score: float | None = None
    final_text: str = ""
    final_score: float | None = None
    final_evidence: float | None = None
    last_observed_s: float = 0.0
    last_commit_s: float = 0.0
    pending_final_since_s: float | None = None
    finalized: bool = False


@dataclass
class _LiveChannelState:
    channel_id: int
    carrier_hz: float
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    channel_started: bool = False
    next_session_id: int = 1
    sessions: list[_LiveSessionState] = field(default_factory=list)
    dormant: bool = False


class NextgenStreamProcessor:
    """Incremental JSON-event streamer backed by the carrier-centric decoder.

    This is intentionally a thin live layer over ``cw.nextgen``: carrier
    detection and text decoding use the same signal-domain path as ``decode-raw``.
    The live layer only buffers audio, periodically re-decodes the retained
    window, and turns timed nextgen sessions into stable stream events.
    """

    def __init__(self, sample_rate: int, config: StreamingConfig) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(sample_rate)
        self.config = config
        self.processed_duration_s = 0.0
        self.frames_processed = 0
        self.tracker_frames_processed = 0
        self.retained_frames = 0
        self.pruned_frames = 0
        self.last_input_rms = 0.0
        self.last_input_peak = 0.0

        self._window = np.asarray([], dtype=np.float32)
        self._window_start_s = 0.0
        self._last_decode_s = 0.0
        self._events: list[StreamEvent] = []
        self._channels: list[_LiveChannelState] = []
        self._next_channel_id = 1

    def push(self, samples: np.ndarray) -> StreamChunkResult:
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) == 0:
            return StreamChunkResult(time_s=self.processed_duration_s)

        self.last_input_rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        self.last_input_peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        self._window = np.concatenate([self._window, samples]) if len(self._window) else samples.copy()
        self.processed_duration_s += len(samples) / self.sample_rate
        self._update_counters()
        self._prune_window_if_needed()

        before = len(self._events)
        if self.processed_duration_s - self._last_decode_s >= max(0.05, self.config.emit_interval_s):
            self._last_decode_s = self.processed_duration_s
            self._decode_and_emit(final=False)
        return StreamChunkResult(
            time_s=self.processed_duration_s,
            events=self._events[before:],
            frames_processed=self.frames_processed,
            tracker_frames_processed=self.tracker_frames_processed,
            retained_frames=self.retained_frames,
            pruned_frames=self.pruned_frames,
        )

    def finish(self, *, final_time_s: float | None = None) -> StreamChunkResult:
        if final_time_s is not None:
            self.processed_duration_s = max(self.processed_duration_s, float(final_time_s))
        before = len(self._events)
        self._decode_and_emit(final=True)
        for channel in self._channels:
            for session in channel.sessions:
                if not session.finalized and (session.final_text or session.committed_text):
                    self._emit_session_final(channel, session, reason=self.config.final_event_reason)
            if not channel.dormant:
                self._events.append(
                    StreamEvent(
                        time_s=self.processed_duration_s,
                        kind="CHANNEL_DORMANT",
                        channel_id=channel.channel_id,
                        session_id=None,
                        carrier_hz=channel.carrier_hz,
                        text="",
                        score=None,
                        reason=self.config.final_event_reason,
                    )
                )
                channel.dormant = True
        return StreamChunkResult(
            time_s=self.processed_duration_s,
            events=self._events,
            frames_processed=self.frames_processed,
            tracker_frames_processed=self.tracker_frames_processed,
            retained_frames=self.retained_frames,
            pruned_frames=self.pruned_frames,
        )

    def _decode_and_emit(self, *, final: bool) -> None:
        signal, signal_start_s = self._decode_window()
        if len(signal) < max(1, int(self.sample_rate * 0.20)):
            return
        detected = _detect_carriers_nextgen(
            signal,
            self.sample_rate,
            min_tone_hz=self.config.min_tone_hz,
            max_tone_hz=self.config.max_tone_hz,
            max_carriers=self.config.max_tracks,
            min_separation_hz=self.config.min_separation_hz,
            relative_threshold=self.config.peak_relative_threshold,
            frame_ms=effective_tracker_frame_ms(self.config),
            hop_ms=effective_tracker_hop_ms(self.config),
        )
        if not detected:
            self._finalize_inactive_channels()
            return

        seen_channels: set[int] = set()
        for detected_carrier in detected:
            carrier_hz = float(detected_carrier.carrier_hz)
            channel = self._channel_for_carrier(carrier_hz)
            seen_channels.add(channel.channel_id)
            if channel.hits < self.config.min_track_hits:
                continue
            result = decode_signal_carrier_nextgen(
                signal,
                self.sample_rate,
                carrier_hz=carrier_hz,
                start_s=signal_start_s,
                threshold_ratios=self.config.threshold_ratios or (self.config.threshold_ratio,),
                lowpass_ms=max(5.0, self.config.frame_ms / 2.5),
                envelope_hop_ms=self.config.hop_ms,
                session_gap_s=self.config.min_session_gap_s,
                min_session_evidence_score=0.0,
                config=self.config,
                max_candidates=self.config.max_tracks,
                max_candidates_per_session=4,
            )
            self._publish_carrier_result(channel, result, final=final)

        self._finalize_inactive_channels(seen_channel_ids=seen_channels)


    def _decode_window(self) -> tuple[np.ndarray, float]:
        # The carrier-centric decoder is intentionally richer than the old STFT
        # path, but repeatedly decoding an ever-growing live buffer would be
        # quadratic.  Decode a recent rolling window and keep the state machine
        # responsible for carrying session ids and finalization forward.
        target_s = 12.0
        if self.config.max_history_s is not None:
            target_s = min(target_s, float(self.config.max_history_s))
        max_samples = max(1, int(round(target_s * self.sample_rate)))
        if len(self._window) <= max_samples:
            return self._window, self._window_start_s
        return self._window[-max_samples:], self._window_start_s + (len(self._window) - max_samples) / self.sample_rate

    def _publish_carrier_result(self, channel: _LiveChannelState, result: NextgenCarrierResult, *, final: bool) -> None:
        channel.carrier_hz = _smooth_carrier(channel.carrier_hz, result.carrier_hz, self.config.carrier_smoothing)
        channel.last_seen_s = self.processed_duration_s
        for decoded_session in result.sessions:
            if not decoded_session.text or not self._session_passes_publish_filters(decoded_session):
                continue
            session = self._session_for_decoded(channel, decoded_session)
            if session is None or session.finalized:
                continue
            previous_end_s = session.end_s
            session.end_s = max(session.end_s, decoded_session.end_s)
            session.last_observed_s = self.processed_duration_s
            score = decoded_session.best.quality_score if decoded_session.best is not None else None
            self._remember_final_candidate(session, decoded_session, score)
            text_to_emit = self._text_to_commit(session, decoded_session.text, score)
            if text_to_emit is not None:
                self._events.append(
                    StreamEvent(
                        time_s=self.processed_duration_s,
                        kind="TEXT_COMMITTED",
                        channel_id=channel.channel_id,
                        session_id=session.session_id,
                        carrier_hz=channel.carrier_hz,
                        text=text_to_emit,
                        score=score,
                    )
                )
            if session.end_s > previous_end_s + max(0.05, self.config.hop_ms / 1000 * 4):
                session.pending_final_since_s = None
            if final:
                self._emit_session_final(channel, session, reason=self.config.final_event_reason)
            elif decoded_session.end_s <= self.processed_duration_s - self.config.finalization_delay_s:
                self._maybe_emit_pending_final(channel, session)


    def _finalize_inactive_channels(self, *, seen_channel_ids: set[int] | None = None) -> None:
        seen_channel_ids = seen_channel_ids or set()
        inactive_after_s = _channel_inactive_after_s(self.config)
        for channel in self._channels:
            if channel.channel_id in seen_channel_ids:
                continue
            if channel.dormant:
                continue
            if self.processed_duration_s - channel.last_seen_s < inactive_after_s:
                continue
            for session in channel.sessions:
                if not session.finalized and (session.final_text or session.committed_text):
                    self._emit_session_final(channel, session, reason="channel_inactive")
            if channel.channel_started:
                self._events.append(
                    StreamEvent(
                        time_s=self.processed_duration_s,
                        kind="CHANNEL_DORMANT",
                        channel_id=channel.channel_id,
                        session_id=None,
                        carrier_hz=channel.carrier_hz,
                        text="",
                        score=None,
                        reason="channel_inactive",
                    )
                )
            channel.dormant = True
            # A dormant channel must be confirmed again before becoming public.
            # Otherwise a stale, previously strong channel would restart on a
            # single noisy hit just because its historical hit count was high.
            channel.hits = 0


    def _session_passes_publish_filters(self, decoded: NextgenSession) -> bool:
        candidate = decoded.best
        text = decoded.text or ""
        compact = "".join(char for char in text if not char.isspace())
        known_chars = sum(1 for char in compact if char != "?")
        if len(compact) < self.config.min_keying_chars:
            return False
        if known_chars < self.config.min_keying_known_chars:
            return False
        if candidate is None:
            return True
        tone_runs = [run for run in candidate.runs if run.kind == "tone"]
        if len(tone_runs) < self.config.min_keying_tone_runs:
            return False
        active_duration_s = sum(run.duration_s for run in tone_runs)
        if active_duration_s < self.config.min_keying_active_duration_s:
            return False
        if self.config.min_keying_duty_cycle is not None and candidate.duty_cycle < self.config.min_keying_duty_cycle:
            return False
        if self.config.max_keying_duty_cycle is not None and candidate.duty_cycle > self.config.max_keying_duty_cycle:
            return False
        if self.config.min_keying_unit_s and candidate.unit_s is not None and candidate.unit_s < self.config.min_keying_unit_s:
            return False
        if self.config.max_keying_unit_s is not None and candidate.unit_s is not None and candidate.unit_s > self.config.max_keying_unit_s:
            return False
        if self.config.max_keying_score is not None and candidate.quality_score is not None and candidate.quality_score > self.config.max_keying_score:
            return False
        if self.config.reject_et_only_sessions and len(compact) >= self.config.et_only_min_chars:
            if set(compact) <= {"E", "T"}:
                return False
        return True

    def _channel_for_carrier(self, carrier_hz: float) -> _LiveChannelState:
        channel = self._best_channel_for_carrier(carrier_hz)
        if channel is not None:
            channel.hits += 1
            channel.carrier_hz = _smooth_carrier(channel.carrier_hz, carrier_hz, self.config.carrier_smoothing)
            channel.last_seen_s = self.processed_duration_s
            if channel.hits >= self.config.min_track_hits:
                self._ensure_channel_started(channel)
            return channel
        channel = _LiveChannelState(
            channel_id=self._next_channel_id,
            carrier_hz=round(float(carrier_hz), 3),
            first_seen_s=self.processed_duration_s,
            last_seen_s=self.processed_duration_s,
        )
        self._next_channel_id += 1
        self._channels.append(channel)
        if channel.hits >= self.config.min_track_hits:
            self._ensure_channel_started(channel)
        return channel

    def _best_channel_for_carrier(self, carrier_hz: float) -> _LiveChannelState | None:
        strict_match_hz = channel_match_hz(self.config)
        reacquire_hz = _channel_reacquire_hz(self.config)
        reacquire_s = self.config.channel_reacquire_s
        matches: list[tuple[int, float, _LiveChannelState]] = []
        for channel in self._channels:
            delta_hz = abs(channel.carrier_hz - carrier_hz)
            if delta_hz <= strict_match_hz:
                matches.append((0, delta_hz, channel))
                continue
            # A real CW station may be reported a little differently as the
            # rolling window moves, especially around fades or hand-keyed
            # long transmissions.  Reuse a recent channel within a wider
            # tolerance instead of opening a second public channel.  This is
            # still bounded by time and frequency so nearby simultaneous
            # stations do not get merged indefinitely.
            recently_seen = self.processed_duration_s - channel.last_seen_s <= reacquire_s
            if recently_seen and delta_hz <= reacquire_hz:
                matches.append((1, delta_hz, channel))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1], -item[2].hits))
        return matches[0][2]

    def _ensure_channel_started(self, channel: _LiveChannelState) -> None:
        if channel.channel_started and not channel.dormant:
            return
        self._events.append(
            StreamEvent(
                time_s=self.processed_duration_s,
                kind="CHANNEL_STARTED",
                channel_id=channel.channel_id,
                session_id=None,
                carrier_hz=channel.carrier_hz,
            )
        )
        channel.channel_started = True
        channel.dormant = False

    def _session_for_decoded(self, channel: _LiveChannelState, decoded: NextgenSession) -> _LiveSessionState | None:
        for session in channel.sessions:
            if session.finalized:
                continue
            overlap = min(session.end_s, decoded.end_s) - max(session.start_s, decoded.start_s)
            close_start = abs(session.start_s - decoded.start_s) <= max(0.35, self.config.emit_interval_s)
            if overlap > -0.25 or close_start:
                return session
        finalized_end = max((session.end_s for session in channel.sessions if session.finalized), default=None)
        if finalized_end is not None:
            margin = max(0.25, self.config.history_margin_s)
            if decoded.start_s <= finalized_end + margin or decoded.end_s <= finalized_end + margin:
                return None
        session = _LiveSessionState(
            session_id=channel.next_session_id,
            start_s=decoded.start_s,
            end_s=decoded.end_s,
        )
        channel.next_session_id += 1
        channel.sessions.append(session)
        self._ensure_channel_started(channel)
        self._events.append(
            StreamEvent(
                time_s=self.processed_duration_s,
                kind="SESSION_STARTED",
                channel_id=channel.channel_id,
                session_id=session.session_id,
                carrier_hz=channel.carrier_hz,
            )
        )
        return session

    def _remember_final_candidate(
        self,
        session: _LiveSessionState,
        decoded: NextgenSession,
        score: float | None,
    ) -> None:
        text = decoded.text.strip()
        if not text:
            return
        evidence = decoded.best.evidence_score if decoded.best is not None else _text_evidence(text, score)
        if session.final_text:
            current_evidence = session.final_evidence if session.final_evidence is not None else _text_evidence(session.final_text, session.final_score)
            if not _candidate_is_better_final(
                current_text=session.final_text,
                current_score=session.final_score,
                current_evidence=current_evidence,
                new_text=text,
                new_score=score,
                new_evidence=evidence,
            ):
                return
        session.final_text = text
        session.final_score = score
        session.final_evidence = evidence

    def _maybe_emit_pending_final(self, channel: _LiveChannelState, session: _LiveSessionState) -> None:
        if session.finalized:
            return
        if session.pending_final_since_s is None:
            session.pending_final_since_s = self.processed_duration_s
            return
        settle_s = max(0.25, self.config.emit_interval_s)
        if self.processed_duration_s - session.pending_final_since_s < settle_s:
            return
        self._emit_session_final(channel, session, reason="silence_gap")

    def _emit_session_final(self, channel: _LiveChannelState, session: _LiveSessionState, *, reason: str) -> None:
        if session.finalized:
            return
        session.finalized = True
        final_text, final_score = self._final_text_for(session)
        self._events.append(
            StreamEvent(
                time_s=self.processed_duration_s,
                kind="SESSION_FINAL",
                channel_id=channel.channel_id,
                session_id=session.session_id,
                carrier_hz=channel.carrier_hz,
                text=final_text,
                score=final_score,
                reason=reason,
            )
        )

    def _text_to_commit(self, session: _LiveSessionState, current_text: str, score: float | None) -> str | None:
        if not current_text:
            return None
        if not self.config.stable_updates:
            if current_text == session.committed_text:
                session.last_candidate_text = current_text
                return None
            session.last_candidate_text = current_text
            session.committed_text = current_text
            session.committed_score = score
            session.last_commit_s = self.processed_duration_s
            return current_text
        if score is not None and score > self.config.min_update_score:
            return None

        previous_text = session.last_candidate_text
        session.last_candidate_text = current_text
        if not previous_text:
            return self._progress_text_to_commit(session, current_text, score)

        stable_prefix = _common_text_prefix(previous_text, current_text).rstrip()
        if stable_prefix and _compact_len(stable_prefix) >= self.config.min_live_commit_chars:
            if stable_prefix.startswith(session.committed_text) and len(stable_prefix) > len(session.committed_text):
                session.committed_text = stable_prefix
                session.committed_score = score
                session.last_commit_s = self.processed_duration_s
                return stable_prefix

        return self._progress_text_to_commit(session, current_text, score)

    def _progress_text_to_commit(self, session: _LiveSessionState, current_text: str, score: float | None) -> str | None:
        # Stable-prefix commits deliberately avoid flickering early hypotheses.
        # For a long continuous transmission, however, the rolling decode window
        # eventually no longer contains the beginning of the session.  In that
        # case prefix compatibility is impossible, so stitch the current rolling
        # text onto the already committed suffix when there is a compact-text
        # overlap.  This keeps active QSOs/long prose moving even without a
        # SESSION_FINAL boundary.
        committed_text = session.committed_text.strip()
        if not committed_text:
            return None

        if self.processed_duration_s - session.last_commit_s < self.config.live_progress_interval_s:
            return None

        stitched = _stitch_rolling_text(
            committed_text,
            current_text.strip(),
            min_overlap_chars=self.config.live_progress_min_overlap_chars,
        )
        if stitched is None:
            return None
        if _compact_len(stitched) <= _compact_len(committed_text):
            return None
        session.committed_text = stitched
        session.committed_score = score
        session.last_commit_s = self.processed_duration_s
        return stitched

    def _final_text_for(self, session: _LiveSessionState) -> tuple[str, float | None]:
        final_text = session.final_text.strip()
        committed_text = session.committed_text.strip()
        if not committed_text:
            return final_text, session.final_score
        if not final_text:
            return committed_text, session.committed_score
        if _texts_are_compatible(committed_text, final_text):
            return final_text, session.final_score
        if session.committed_score is not None and session.final_score is not None:
            # The final rolling-window decode may choose a different threshold/unit
            # hypothesis after the signal has ended.  Avoid replacing a stable
            # live prefix with an incompatible final text unless the final
            # hypothesis is materially better.  Lower quality scores are better.
            if session.final_score >= session.committed_score - self.config.final_text_regression_margin:
                return committed_text, session.committed_score
        return final_text, session.final_score

    def _prune_window_if_needed(self) -> None:
        max_history_s = self.config.max_history_s
        if max_history_s is None or max_history_s <= 0:
            return
        max_samples = int(round(max_history_s * self.sample_rate))
        if len(self._window) <= max_samples:
            return
        drop = len(self._window) - max_samples
        self._window = self._window[drop:].copy()
        self._window_start_s += drop / self.sample_rate
        self.pruned_frames += int(round(drop / max(1, int(round(self.sample_rate * self.config.hop_ms / 1000)))))

    def _update_counters(self) -> None:
        hop_s = max(1e-6, self.config.hop_ms / 1000)
        tracker_hop_s = max(1e-6, effective_tracker_hop_ms(self.config) / 1000)
        self.frames_processed = int(round(self.processed_duration_s / hop_s))
        self.tracker_frames_processed = int(round(self.processed_duration_s / tracker_hop_s))
        self.retained_frames = int(round((len(self._window) / self.sample_rate) / hop_s))


def _stitch_rolling_text(previous: str, current: str, *, min_overlap_chars: int) -> str | None:
    if not previous or not current:
        return None
    if current.startswith(previous):
        return current
    if previous.startswith(current):
        return None

    previous_compact, _previous_map = _compact_with_index_map(previous)
    current_compact, current_map = _compact_with_index_map(current)
    max_overlap = min(len(previous_compact), len(current_compact))
    for overlap in range(max_overlap, min_overlap_chars - 1, -1):
        if previous_compact[-overlap:] != current_compact[:overlap]:
            continue
        append_start = current_map[overlap - 1] + 1
        suffix = current[append_start:].lstrip()
        if not suffix:
            return None
        separator = "" if previous.endswith((" ", "/")) or suffix.startswith((" ", "/")) else " "
        return f"{previous}{separator}{suffix}".strip()
    return None


def _compact_with_index_map(text: str) -> tuple[str, list[int]]:
    compact_chars: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        indexes.append(index)
    return "".join(compact_chars), indexes


def _channel_inactive_after_s(config: StreamingConfig) -> float:
    return max(
        6.0,
        config.max_track_gap_s * 3.0,
        config.min_session_gap_s + config.finalization_delay_s + config.emit_interval_s * 2.0,
    )


def _channel_reacquire_hz(config: StreamingConfig) -> float:
    if config.channel_reacquire_hz is not None:
        return config.channel_reacquire_hz
    # Wider than the strict channel match, but no wider than the configured
    # carrier separation by default.  With the default 80 Hz peak separation this
    # lets a drifting/retuned single station stay on one public channel while
    # still keeping clearly separate carriers apart.
    return max(channel_match_hz(config), config.min_separation_hz)


def _smooth_carrier(previous: float, current: float, smoothing: float) -> float:
    alpha = min(1.0, max(0.0, float(smoothing)))
    return round(previous * (1.0 - alpha) + current * alpha, 3)


def _common_text_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def _compact_len(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _texts_are_compatible(left: str, right: str) -> bool:
    return bool(left and right and (left.startswith(right) or right.startswith(left)))


def _compact_common_prefix_len(left: str, right: str) -> int:
    left_compact = "".join(char for char in left if not char.isspace())
    right_compact = "".join(char for char in right if not char.isspace())
    limit = min(len(left_compact), len(right_compact))
    index = 0
    while index < limit and left_compact[index] == right_compact[index]:
        index += 1
    return index


def _text_evidence(text: str, score: float | None) -> float:
    compact = "".join(char for char in text if not char.isspace())
    known_chars = sum(1 for char in compact if char != "?")
    unknowns = compact.count("?")
    score_penalty = 0.0 if score is None else max(0.0, score) * 0.20
    return known_chars * 1.8 - unknowns * 2.0 - score_penalty


def _candidate_is_better_final(
    *,
    current_text: str,
    current_score: float | None,
    current_evidence: float,
    new_text: str,
    new_score: float | None,
    new_evidence: float,
) -> bool:
    if not current_text:
        return True
    if new_text == current_text:
        return new_score is not None and (current_score is None or new_score < current_score)
    compatible = _texts_are_compatible(current_text, new_text)
    if compatible:
        current_compact_len = _compact_len(current_text)
        new_compact_len = _compact_len(new_text)
        if new_compact_len > current_compact_len:
            if new_evidence >= current_evidence - 4.0:
                return True
            if new_score is not None and current_score is not None and new_score <= current_score + 5.0:
                return True
        if new_evidence >= current_evidence - 1.0 and len(new_text.strip()) >= len(current_text.strip()):
            return True
        if new_score is not None and current_score is not None and new_score + 3.0 < current_score:
            return True
        return False
    current_compact_len = _compact_len(current_text)
    new_compact_len = _compact_len(new_text)
    shared_prefix_len = _compact_common_prefix_len(current_text, new_text)
    strong_shared_prefix = shared_prefix_len >= 6 and shared_prefix_len / max(1, min(current_compact_len, new_compact_len)) >= 0.70
    if strong_shared_prefix and new_compact_len >= current_compact_len:
        if new_evidence >= current_evidence + 2.0:
            return True
        if new_score is not None and current_score is not None and new_score <= current_score + 3.0:
            return True
    # Incompatible late rolling-window candidates are common in live decoding.
    # Keep the earlier/better final candidate unless the new hypothesis brings
    # materially stronger signal evidence or a much better timing score.
    if new_evidence >= current_evidence + 8.0:
        return True
    if new_score is not None and current_score is not None and new_score + 12.0 < current_score:
        return True
    return False
