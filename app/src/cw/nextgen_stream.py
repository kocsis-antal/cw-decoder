from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from cw.nextgen import NextgenCarrierResult, NextgenSession, _detect_carriers_nextgen, decode_signal_carrier_nextgen
from cw.stream_models import StreamChunkResult, StreamEvent, StreamingConfig, channel_match_hz, effective_tracker_hop_ms
from cw.live_layers import (
    LiveCarrierDecoder as _LiveCarrierDecoder,
    LiveCarrierDetector as _LiveCarrierDetector,
    LiveSessionHypothesisArbiter as _LiveSessionHypothesisArbiter,
)


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
    last_preview_text: str = ""
    last_preview_score: float | None = None
    last_preview_evidence: float | None = None
    last_preview_s: float = -1.0e9
    best_preview_text: str = ""
    best_preview_score: float | None = None
    best_preview_evidence: float | None = None
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
    last_symbol_hmm_s: float = -1.0e9
    last_activity_event_s: float = -1.0e9


class NextgenStreamProcessor:
    """Incremental JSON-event streamer backed by the carrier-centric decoder.

    This is intentionally a live layer over ``cw.nextgen`` rather than a
    second decoder.  Carrier demodulation and text candidates use the same
    signal-domain primitives as ``decode-raw``, but the live path is not a
    long batch re-decode loop: it decodes a short recent window and keeps
    committed transcript state across windows.  That lets simultaneous/late
    carriers show up quickly instead of being masked by the strongest signal
    in a long rolling history.
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
        self._carrier_detector = _LiveCarrierDetector(self.sample_rate, self.config)
        self._carrier_decoder = _LiveCarrierDecoder(self.sample_rate, self.config)
        self._hypothesis_arbiter = _LiveSessionHypothesisArbiter()

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
        # Use two separate time scales.  Carrier detection must be short and
        # responsive so a second station can appear immediately.  Text decoding
        # needs a longer per-carrier window, otherwise common live fragments like
        # "CQ CQ DE ..." are seen only as unstable scraps and never settle.
        detect_signal, _detect_start_s = self._recent_window(self.config.live_carrier_window_s)
        if len(detect_signal) < max(1, int(self.sample_rate * 0.20)):
            return
        detected_carrier_hz = self._carrier_detector.detect(detect_signal)
        if not detected_carrier_hz:
            self._finalize_inactive_channels()
            return

        seen_channels: set[int] = set()
        for carrier_hz in detected_carrier_hz:
            channel = self._channel_for_carrier(carrier_hz)
            seen_channels.add(channel.channel_id)
            if channel.hits < self.config.min_track_hits:
                continue
            decode_signal, decode_start_s = self._channel_decode_window(channel)
            if len(decode_signal) < max(1, int(self.sample_rate * 0.20)):
                continue
            decode_config = self._config_for_channel_decode(channel, final=final)
            result = self._carrier_decoder.decode(
                decode_signal,
                start_s=decode_start_s,
                carrier_hz=channel.carrier_hz,
                decode_config=decode_config,
            )
            self._publish_carrier_result(channel, result, final=final)

        self._finalize_inactive_channels(seen_channel_ids=seen_channels)

    def _config_for_channel_decode(self, channel: _LiveChannelState, *, final: bool) -> StreamingConfig:
        # Live viewing needs a low-latency receiver, not an expensive offline
        # re-interpreter.  The symbol HMM is useful for batch/rescue decoding,
        # but repeatedly running it on every rolling live window causes latency,
        # flicker, and long stalls.  Keep it opt-in for the live processor.
        if not self.config.symbol_hmm_decoding or not self.config.live_symbol_hmm_decoding:
            return replace(self.config, symbol_hmm_decoding=False)
        interval_s = self.config.symbol_hmm_live_interval_s
        run_hmm = final or interval_s <= 0 or self.processed_duration_s - channel.last_symbol_hmm_s >= interval_s
        if run_hmm:
            channel.last_symbol_hmm_s = self.processed_duration_s
            return self.config
        return replace(self.config, symbol_hmm_decoding=False)


    def _decode_window(self) -> tuple[np.ndarray, float]:
        # Backwards-compatible helper used by tests and callers that want the
        # text-decode window.  Carrier detection now has its own shorter window.
        return self._recent_window(self.config.live_decode_window_s)

    def _recent_window(self, target_s: float) -> tuple[np.ndarray, float]:
        target_s = float(target_s)
        if self.config.max_history_s is not None:
            target_s = min(target_s, float(self.config.max_history_s))
        max_samples = max(1, int(round(target_s * self.sample_rate)))
        if len(self._window) <= max_samples:
            return self._window, self._window_start_s
        return self._window[-max_samples:], self._window_start_s + (len(self._window) - max_samples) / self.sample_rate

    def _channel_decode_window(self, channel: _LiveChannelState) -> tuple[np.ndarray, float]:
        # Start near the carrier's first public appearance when possible, but
        # cap the amount of audio decoded per tick.  This gives the decoder
        # enough context for calls/CQ phrases while the carrier detector remains
        # short-windowed.
        earliest_start_s = max(self._window_start_s, channel.first_seen_s - self.config.history_margin_s)
        latest_start_s = self.processed_duration_s - self.config.live_decode_window_s
        start_s = max(earliest_start_s, latest_start_s)
        if start_s <= self._window_start_s:
            return self._window, self._window_start_s
        start_index = int(round((start_s - self._window_start_s) * self.sample_rate))
        start_index = min(max(0, start_index), len(self._window))
        return self._window[start_index:].copy(), self._window_start_s + start_index / self.sample_rate

    def _publish_carrier_result(self, channel: _LiveChannelState, result: NextgenCarrierResult, *, final: bool) -> None:
        channel.carrier_hz = _smooth_carrier(channel.carrier_hz, result.carrier_hz, self.config.carrier_smoothing)
        channel.last_seen_s = self.processed_duration_s
        events_before = len(self._events)
        decoded_any_session = False
        for decoded_session in result.sessions:
            if not decoded_session.text:
                continue
            decoded_any_session = True
            prelim_stable_publishable = self._session_passes_publish_filters(decoded_session)
            prelim_preview_publishable = self._session_passes_preview_filters(decoded_session)
            if not prelim_stable_publishable and not prelim_preview_publishable:
                continue
            session = self._session_for_decoded(channel, decoded_session)
            if session is None or session.finalized:
                continue
            decoded_session = self._hypothesis_arbiter.choose(decoded_session, session)
            stable_publishable = self._session_passes_publish_filters(decoded_session)
            preview_publishable = self._session_passes_preview_filters(decoded_session)
            if not stable_publishable and not preview_publishable:
                continue
            previous_end_s = session.end_s
            session.end_s = max(session.end_s, decoded_session.end_s)
            session.last_observed_s = self.processed_duration_s
            score = decoded_session.best.quality_score if decoded_session.best is not None else None

            if stable_publishable:
                self._remember_best_preview(session, decoded_session.text, score)
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
                elif self.config.preview_updates:
                    self._maybe_emit_text_preview(channel, session, decoded_session.text, score, reason="awaiting_stable_prefix")
                if session.end_s > previous_end_s + max(0.05, self.config.hop_ms / 1000 * 4):
                    session.pending_final_since_s = None
                if final:
                    self._emit_session_final(channel, session, reason=self.config.final_event_reason)
                elif decoded_session.end_s <= self.processed_duration_s - self.config.finalization_delay_s:
                    self._maybe_emit_pending_final(channel, session)
            elif self.config.preview_updates:
                self._maybe_emit_text_preview(channel, session, decoded_session.text, score, reason="below_commit_threshold")

        if len(self._events) == events_before:
            self._maybe_emit_signal_active(
                channel,
                result,
                reason="awaiting_decodable_text" if decoded_any_session else "carrier_detected",
            )


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


    def _maybe_emit_signal_active(self, channel: _LiveChannelState, result: NextgenCarrierResult, *, reason: str) -> None:
        if not channel.channel_started or channel.dormant:
            return
        interval_s = self.config.signal_activity_interval_s
        if interval_s <= 0:
            return
        if self.processed_duration_s - channel.last_activity_event_s < interval_s:
            return
        channel.last_activity_event_s = self.processed_duration_s
        self._events.append(
            StreamEvent(
                time_s=self.processed_duration_s,
                kind="SIGNAL_ACTIVE",
                channel_id=channel.channel_id,
                session_id=None,
                carrier_hz=channel.carrier_hz,
                text="",
                score=result.best.quality_score if result.best is not None else None,
                reason=reason,
            )
        )

    def _maybe_emit_text_preview(
        self,
        channel: _LiveChannelState,
        session: _LiveSessionState,
        text: str,
        score: float | None,
        *,
        reason: str,
    ) -> None:
        preview_text = text.strip()
        if not preview_text:
            return
        if preview_text == session.last_preview_text and self.processed_duration_s - session.last_preview_s < self.config.preview_interval_s:
            return
        if session.committed_text and preview_text == session.committed_text:
            return
        if self.processed_duration_s - session.last_preview_s < self.config.preview_interval_s:
            # Allow immediate correction from an empty/noisy state only if the
            # text grew materially; otherwise previews would become the old
            # flickering live updates under a different event name.
            if _compact_len(preview_text) <= _compact_len(session.last_preview_text) + 2:
                return
        session.last_preview_text = preview_text
        session.last_preview_score = score
        session.last_preview_evidence = _text_evidence(preview_text, score)
        session.last_preview_s = self.processed_duration_s
        self._remember_best_preview(session, preview_text, score)
        self._events.append(
            StreamEvent(
                time_s=self.processed_duration_s,
                kind="TEXT_PREVIEW",
                channel_id=channel.channel_id,
                session_id=session.session_id,
                carrier_hz=channel.carrier_hz,
                text=preview_text,
                score=score,
                reason=reason,
            )
        )

    def _session_passes_preview_filters(self, decoded: NextgenSession) -> bool:
        if not self.config.preview_updates:
            return False
        text = decoded.text or ""
        compact = "".join(char for char in text if not char.isspace())
        known_chars = sum(1 for char in compact if char != "?")
        if len(compact) < self.config.preview_min_chars:
            return False
        if known_chars < max(1, self.config.preview_min_chars):
            return False
        candidate = decoded.best
        if candidate is None:
            return True
        if self.config.preview_max_score is not None and candidate.quality_score is not None:
            if candidate.quality_score > self.config.preview_max_score:
                return False
        if self.config.reject_et_only_sessions and len(compact) >= self.config.et_only_min_chars:
            if set(compact) <= {"E", "T"}:
                return False
        if self.config.min_keying_unit_s and candidate.unit_s is not None and candidate.unit_s < self.config.min_keying_unit_s:
            return False
        if self.config.max_keying_unit_s is not None and candidate.unit_s is not None and candidate.unit_s > self.config.max_keying_unit_s:
            return False
        tone_runs = [run for run in candidate.runs if run.kind == "tone"]
        if len(tone_runs) < max(1, min(self.config.min_keying_tone_runs, 2)):
            return False
        active_duration_s = sum(run.duration_s for run in tone_runs)
        if active_duration_s < min(0.05, self.config.min_keying_active_duration_s or 0.05):
            return False
        if self.config.max_keying_duty_cycle is not None and candidate.duty_cycle > self.config.max_keying_duty_cycle:
            return False
        return True

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

    def _remember_best_preview(self, session: _LiveSessionState, text: str, score: float | None) -> None:
        text = text.strip()
        if not text:
            return
        evidence = _text_evidence(text, score)
        if not session.best_preview_text:
            session.best_preview_text = text
            session.best_preview_score = score
            session.best_preview_evidence = evidence
            return
        current_evidence = session.best_preview_evidence if session.best_preview_evidence is not None else _text_evidence(session.best_preview_text, session.best_preview_score)
        if _candidate_is_better_live_memory(
            current_text=session.best_preview_text,
            current_score=session.best_preview_score,
            current_evidence=current_evidence,
            new_text=text,
            new_score=score,
            new_evidence=evidence,
        ):
            session.best_preview_text = text
            session.best_preview_score = score
            session.best_preview_evidence = evidence

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
            stitched = _stitch_rolling_text_with_leading_skip(
                committed_text,
                current_text.strip(),
                min_overlap_chars=max(3, self.config.live_progress_min_overlap_chars),
                max_skip_chars=2,
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
        preview_text = (session.last_preview_text or session.best_preview_text).strip()
        best_preview_text = session.best_preview_text.strip()

        # Live reception has a different failure mode than offline decoding:
        # the operator may briefly see the correct interpretation as a preview,
        # then the final rolling-window hypothesis over-explains the trailing
        # silence/fade and replaces it with a shorter or incompatible text.  Do
        # not treat SESSION_FINAL as an oracle; it is just another hypothesis.
        final_text, final_score = _protect_live_candidate_on_final(
            committed_text=committed_text,
            final_text=final_text,
            final_score=session.final_score,
            preview_text=preview_text,
            preview_score=session.last_preview_score if preview_text == session.last_preview_text.strip() else session.best_preview_score,
            best_preview_text=best_preview_text,
            best_preview_score=session.best_preview_score,
        )

        if committed_text and preview_text:
            stitched_preview = _stitch_rolling_text_with_leading_skip(
                committed_text,
                preview_text,
                min_overlap_chars=max(3, self.config.live_progress_min_overlap_chars),
                max_skip_chars=2,
            )
            if stitched_preview is not None and _compact_len(stitched_preview) > max(_compact_len(committed_text), _compact_len(final_text)):
                preview_score = session.last_preview_score if preview_text == session.last_preview_text.strip() else session.best_preview_score
                if _score_not_much_worse(preview_score, final_score, margin=14.0):
                    return stitched_preview, preview_score
        if committed_text and best_preview_text and best_preview_text != preview_text:
            stitched_preview = _stitch_rolling_text_with_leading_skip(
                committed_text,
                best_preview_text,
                min_overlap_chars=max(3, self.config.live_progress_min_overlap_chars),
                max_skip_chars=2,
            )
            if stitched_preview is not None and _compact_len(stitched_preview) > max(_compact_len(committed_text), _compact_len(final_text)):
                if _score_not_much_worse(session.best_preview_score, final_score, margin=14.0):
                    return stitched_preview, session.best_preview_score

        if not committed_text:
            return final_text, final_score
        if not final_text:
            return committed_text, session.committed_score
        if _texts_are_compatible(committed_text, final_text):
            return final_text, final_score
        stitched = _stitch_rolling_text(
            committed_text,
            final_text,
            min_overlap_chars=max(2, self.config.live_progress_min_overlap_chars),
        )
        if stitched is not None and _compact_len(stitched) > _compact_len(committed_text):
            return stitched, final_score
        if session.committed_score is not None and final_score is not None:
            final_evidence = session.final_evidence if session.final_evidence is not None else _text_evidence(final_text, final_score)
            committed_evidence = _text_evidence(committed_text, session.committed_score)
            final_len = _compact_len(final_text)
            committed_len = _compact_len(committed_text)
            final_not_much_worse = final_score <= session.committed_score + self.config.final_text_regression_margin
            if final_len >= committed_len and final_evidence >= committed_evidence + 5.0 and final_not_much_worse:
                return final_text, final_score
            if final_len >= committed_len + 4 and final_score <= session.committed_score + 3.0:
                return final_text, final_score
            # The final rolling-window decode may choose a different threshold/unit
            # hypothesis after the signal has ended.  Avoid replacing a stable
            # live prefix with an incompatible final text unless the final
            # hypothesis is materially better.  Lower quality scores are better.
            if final_score >= session.committed_score - self.config.final_text_regression_margin:
                return committed_text, session.committed_score
        return final_text, final_score

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


def _stitch_rolling_text_with_leading_skip(
    previous: str,
    current: str,
    *,
    min_overlap_chars: int,
    max_skip_chars: int,
) -> str | None:
    """Stitch a rolling decode that starts a few characters before/after sync.

    A short live window often catches the tail of the already committed word
    with one bad/missing leading character, e.g. committed ``HA7VY`` and preview
    ``T7VY 5NN``.  Exact suffix-prefix overlap cannot join those, but dropping
    the stray leading ``T`` reveals the real overlap ``7VY``.  Limit this to a
    tiny leading skip so unrelated new traffic is not glued together.
    """

    if not previous or not current:
        return None
    previous_compact, _previous_map = _compact_with_index_map(previous)
    current_compact, current_map = _compact_with_index_map(current)
    if not previous_compact or not current_compact:
        return None
    max_skip = min(max(0, max_skip_chars), max(0, len(current_compact) - min_overlap_chars))
    for skip in range(1, max_skip + 1):
        remaining = current_compact[skip:]
        max_overlap = min(len(previous_compact), len(remaining))
        for overlap in range(max_overlap, min_overlap_chars - 1, -1):
            if previous_compact[-overlap:] != remaining[:overlap]:
                continue
            append_compact_index = skip + overlap
            if append_compact_index >= len(current_map):
                return None
            append_start = current_map[append_compact_index - 1] + 1
            suffix = current[append_start:].lstrip()
            if not suffix:
                return None
            separator = "" if previous.endswith(" ") or suffix.startswith(" ") else " "
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


def _protect_live_candidate_on_final(
    *,
    committed_text: str,
    final_text: str,
    final_score: float | None,
    preview_text: str,
    preview_score: float | None,
    best_preview_text: str,
    best_preview_score: float | None,
) -> tuple[str, float | None]:
    """Choose a final text without throwing away a better live hypothesis.

    The final decode is still derived from the same rolling batch window as the
    preview.  It is not more authoritative; it merely happened later.  Prefer a
    recent/best live candidate when the final text is shorter, incompatible with
    already committed text, or loses a plausible suffix that was visible live.
    """

    protected_text = preview_text or best_preview_text
    protected_score = preview_score if preview_text else best_preview_score
    if best_preview_text:
        best_evidence = _text_evidence(best_preview_text, best_preview_score)
        protected_evidence = _text_evidence(protected_text, protected_score) if protected_text else -1e9
        if best_evidence > protected_evidence + 2.0:
            protected_text = best_preview_text
            protected_score = best_preview_score
    protected_text = protected_text.strip()
    final_text = final_text.strip()
    if not protected_text:
        return final_text, final_score
    if not final_text:
        return protected_text, protected_score
    if final_text == protected_text:
        return final_text, final_score

    final_len = _compact_len(final_text)
    protected_len = _compact_len(protected_text)
    compatible = _texts_are_compatible(final_text, protected_text)
    common = _compact_common_prefix_len(final_text, protected_text)
    min_len = max(1, min(final_len, protected_len))
    strong_shared_prefix = common >= 4 and common / min_len >= 0.70

    if compatible:
        if protected_len > final_len:
            # Example: preview shows "EQS 5NN", final settles on "R1BQS 5".
            # If the longer live candidate is not materially worse, do not lose
            # the suffix the operator already saw.
            if _score_not_much_worse(protected_score, final_score, margin=12.0):
                return protected_text, protected_score
        return final_text, final_score

    if committed_text:
        # Never let an incompatible final hypothesis replace a committed prefix
        # unless it is clearly better.  This is the core live hysteresis.
        if _texts_are_compatible(committed_text, protected_text) or _stitch_rolling_text(committed_text, protected_text, min_overlap_chars=2):
            if protected_len >= final_len - 1 and _score_not_much_worse(protected_score, final_score, margin=14.0):
                return protected_text, protected_score
        if not _texts_are_compatible(committed_text, final_text) and final_len <= protected_len + 2:
            if _score_not_much_worse(protected_score, final_score, margin=18.0):
                return protected_text, protected_score

    if protected_len >= final_len + 2 and _score_not_much_worse(protected_score, final_score, margin=10.0):
        return protected_text, protected_score
    if strong_shared_prefix and protected_len > final_len and _score_not_much_worse(protected_score, final_score, margin=12.0):
        return protected_text, protected_score
    return final_text, final_score


def _score_not_much_worse(candidate_score: float | None, reference_score: float | None, *, margin: float) -> bool:
    if candidate_score is None or reference_score is None:
        return True
    return candidate_score <= reference_score + margin


def _candidate_is_better_live_memory(
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
    if _texts_are_compatible(current_text, new_text):
        if _compact_len(new_text) >= _compact_len(current_text):
            return _score_not_much_worse(new_score, current_score, margin=8.0)
        return False
    common = _compact_common_prefix_len(current_text, new_text)
    min_len = max(1, min(_compact_len(current_text), _compact_len(new_text)))
    if common >= 4 and common / min_len >= 0.70 and _compact_len(new_text) >= _compact_len(current_text):
        return _score_not_much_worse(new_score, current_score, margin=8.0)
    # Otherwise keep a candidate only if it is materially stronger, not merely
    # longer.  This prevents late chaotic rolling decodes from becoming memory.
    if new_evidence >= current_evidence + 8.0 and _score_not_much_worse(new_score, current_score, margin=3.0):
        return True
    if new_score is not None and current_score is not None and new_score + 12.0 < current_score:
        return True
    return False


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
    # materially stronger signal evidence, a much better timing score, or a
    # substantially longer explanation that is not worse by score.  The latter
    # protects live startup: early rolling windows can decode a short wrong
    # fragment before the full call/text is visible.
    if new_compact_len >= current_compact_len + 4:
        if new_score is not None and current_score is not None and new_score <= current_score + 3.0:
            return True
        if new_evidence >= current_evidence + 2.0:
            return True
    if new_score is not None and current_score is not None:
        if new_evidence >= current_evidence + 3.0 and new_score <= current_score + 3.0:
            return True
    if new_evidence >= current_evidence + 8.0:
        return True
    if new_score is not None and current_score is not None and new_score + 12.0 < current_score:
        return True
    return False
