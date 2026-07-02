from __future__ import annotations

from dataclasses import dataclass, field, replace

from cw.decoder import DecodeResult
from cw.morse_table import decode_tokens
from cw.quality import QualityScore
from cw.stream_models import StreamEvent, StreamingConfig, StreamSessionResult, channel_match_hz


@dataclass
class SessionState:
    """Mutable live state for one transmission within a carrier channel.

    The decoded timing/tempo belongs to the StreamSessionResult produced by the
    decoder.  This state keeps live bookkeeping: session start/final events,
    stable committed text, and the safe time up to which already committed audio
    may be pruned from the rolling frame history.
    """

    session_id: int
    first_seen_s: float | None = None
    required_frame_start_s: float | None = None
    start_event_emitted: bool = False
    last_candidate_text: str = ""
    committed_text: str = ""
    committed_quality: QualityScore | None = None
    committed_until_s: float | None = None
    pruned_text: str = ""
    last_tail_text: str = ""
    last_tail_overlap_chars: int = 0
    last_tail_cutoff_s: float | None = None

    def mark_observed(self, session: StreamSessionResult, config: StreamingConfig) -> None:
        if self.first_seen_s is None:
            self.first_seen_s = session.first_seen_s
        if config.prune_committed_active_sessions and self.committed_until_s is not None:
            self.required_frame_start_s = self.committed_until_s
        else:
            self.required_frame_start_s = session.first_seen_s

    def reset_live_text(self) -> None:
        self.last_candidate_text = ""
        self.committed_text = ""
        self.committed_quality = None
        self.committed_until_s = None
        self.pruned_text = ""
        self.last_tail_text = ""
        self.last_tail_overlap_chars = 0
        self.last_tail_cutoff_s = None

    def with_committed_prefix(self, session: StreamSessionResult) -> StreamSessionResult:
        tail_cutoff_s = self.committed_until_s
        tail_text = _decoded_text_after_s(session.decoded, tail_cutoff_s) if tail_cutoff_s is not None else session.decoded.text
        if tail_cutoff_s is None:
            merged_text = session.decoded.text
            overlap_chars = 0
        else:
            merged_text, overlap_chars = _merge_committed_prefix_and_tail(self.committed_text, tail_text)
        self.last_tail_text = tail_text
        self.last_tail_overlap_chars = overlap_chars
        self.last_tail_cutoff_s = tail_cutoff_s
        first_seen_s = self.first_seen_s if self.first_seen_s is not None else session.first_seen_s
        if merged_text == session.decoded.text and first_seen_s == session.first_seen_s:
            return session
        return replace(
            session,
            first_seen_s=round(first_seen_s, 3),
            decoded=replace(session.decoded, text=merged_text),
        )

    def commit_from_session(self, session: StreamSessionResult, config: StreamingConfig) -> str | None:
        previous_committed = self.committed_text
        text_to_emit = self.text_to_commit(session.decoded.text, session.quality.score, config)
        if text_to_emit is None:
            return None

        self.committed_quality = session.quality
        safe_prune_text = _safe_active_prune_prefix(self.committed_text)
        if safe_prune_text.startswith(self.pruned_text) and len(safe_prune_text) > len(self.pruned_text):
            tail_prefix_chars = _tail_prefix_chars_for_merged_prefix(
                prefix_len=len(safe_prune_text),
                base_len=len(previous_committed),
                overlap_len=self.last_tail_overlap_chars,
            )
            if tail_prefix_chars is not None:
                committed_until_s = _decoded_tail_prefix_end_s(
                    session.decoded,
                    tail_prefix_chars,
                    after_s=self.last_tail_cutoff_s,
                )
                if committed_until_s is not None:
                    self.pruned_text = safe_prune_text
                    self.committed_until_s = max(self.committed_until_s or 0.0, committed_until_s)
                    if config.prune_committed_active_sessions:
                        self.required_frame_start_s = self.committed_until_s

        return text_to_emit

    def final_session_candidate(self, session: StreamSessionResult, config: StreamingConfig) -> StreamSessionResult:
        """Avoid replacing already-stable live text with a worse final re-decode.

        Live processing repeatedly decodes a rolling history window.  When a
        session closes, the final decode can include a bit more trailing noise
        and occasionally scores much worse than the stable text that was already
        emitted while the signal was active.  In that case keep the last stable
        text instead of letting the final event regress.
        """

        committed_text = self.committed_text.strip()
        final_text = session.decoded.text.strip()
        if not committed_text or not final_text:
            return session
        if _texts_are_compatible(committed_text, final_text):
            return session
        if self.committed_quality is None:
            return session
        if session.quality.score <= self.committed_quality.score + config.final_text_regression_margin:
            return session
        return replace(
            session,
            decoded=replace(session.decoded, text=committed_text),
            quality=self.committed_quality,
        )

    def text_to_commit(self, current_text: str, score: float, config: StreamingConfig) -> str | None:
        if not config.stable_updates:
            if current_text == self.committed_text:
                self.last_candidate_text = current_text
                return None
            self.last_candidate_text = current_text
            self.committed_text = current_text
            return current_text

        if score > config.min_update_score:
            return None

        previous_text = self.last_candidate_text
        self.last_candidate_text = current_text
        if not previous_text:
            return None

        stable_prefix = _common_text_prefix(previous_text, current_text).rstrip()
        if not stable_prefix:
            return None
        if not stable_prefix.startswith(self.committed_text):
            return None
        if len(stable_prefix) <= len(self.committed_text):
            return None

        self.committed_text = stable_prefix
        return stable_prefix


@dataclass
class ChannelState:
    """Long lived carrier/channel state used as GUI/logging anchor.

    A channel may contain multiple sessions. Session timing is intentionally not
    stored as a channel-wide decoding parameter: each TransmissionSession gets
    its own unit/tempo estimate from the decoder output.
    """

    track_id: int
    carrier_hz: float
    current_session: SessionState = field(default_factory=lambda: SessionState(1))
    finalized_session_ids: set[int] = field(default_factory=set)
    finalized_sessions: list[StreamSessionResult] = field(default_factory=list)
    suppressed_until_s: float | None = None

    @property
    def session_id(self) -> int:
        return self.current_session.session_id

    @property
    def active_session_first_seen_s(self) -> float | None:
        # Backwards-compatible name used by the pruning code.  It now means the
        # earliest frame that the active session still needs, not necessarily the
        # original first tone of the session.
        return self.current_session.required_frame_start_s

    @property
    def finalized_until_s(self) -> float | None:
        times = [session.final_time_s for session in self.finalized_sessions]
        if self.suppressed_until_s is not None:
            times.append(self.suppressed_until_s)
        if not times:
            return None
        return max(times)

    def start_next_session(self) -> None:
        self.current_session = SessionState(self.current_session.session_id + 1)

    def commit_text_candidate(self, current_text: str, score: float, config: StreamingConfig) -> str | None:
        return self.current_session.text_to_commit(current_text, score, config)

    def commit_session_candidate(self, session: StreamSessionResult, config: StreamingConfig) -> str | None:
        return self.current_session.commit_from_session(session, config)


class ChannelRegistry:
    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self._next_track_id = 1
        self._channels: list[ChannelState] = []
        self._pending_events: list[StreamEvent] = []

    @property
    def channels(self) -> list[ChannelState]:
        return self._channels

    @property
    def tracks(self) -> list[ChannelState]:
        # Backwards-compatible internal alias while the public result still uses track_id.
        return self.channels

    def channel_for(self, carrier_hz: float, time_s: float = 0.0) -> ChannelState:
        max_match_hz = channel_match_hz(self.config)
        candidates = [channel for channel in self._channels if abs(channel.carrier_hz - carrier_hz) <= max_match_hz]
        if candidates:
            channel = min(candidates, key=lambda existing: abs(existing.carrier_hz - carrier_hz))
            smoothing = self.config.carrier_smoothing
            channel.carrier_hz = (1 - smoothing) * channel.carrier_hz + smoothing * carrier_hz
            return channel

        channel = ChannelState(track_id=self._next_track_id, carrier_hz=carrier_hz)
        channel.current_session.start_event_emitted = True
        self._channels.append(channel)
        self._next_track_id += 1
        self._pending_events.extend(
            [
                StreamEvent(
                    time_s=round(time_s, 3),
                    kind="CHANNEL_STARTED",
                    channel_id=channel.track_id,
                    session_id=None,
                    carrier_hz=round(carrier_hz, 3),
                ),
                StreamEvent(
                    time_s=round(time_s, 3),
                    kind="SESSION_STARTED",
                    channel_id=channel.track_id,
                    session_id=channel.session_id,
                    carrier_hz=round(carrier_hz, 3),
                ),
            ]
        )
        return channel

    def sync_sessions(self, channel: ChannelState, sessions: list[StreamSessionResult]) -> StreamSessionResult | None:
        active_session: StreamSessionResult | None = None
        channel.current_session.required_frame_start_s = None

        for decoded_session in sessions:
            if self._is_stale_decoded_session(channel, decoded_session):
                continue

            existing = self._matching_finalized_session(channel, decoded_session)
            if existing is not None:
                if channel.session_id <= existing.session_id:
                    channel.current_session = SessionState(existing.session_id + 1)
                continue

            session = self._with_current_session_id(channel, decoded_session)
            self._ensure_session_started(channel, session)
            channel.current_session.mark_observed(session, self.config)
            session = channel.current_session.with_committed_prefix(session)

            if session.final_reason == "silence_gap":
                self._finalize_session(channel, session)
            elif session.final_reason == self.config.final_event_reason:
                active_session = session

        return active_session

    def prune_cutoff_s(self) -> tuple[float, str] | None:
        active_starts = [
            channel.active_session_first_seen_s
            for channel in self._channels
            if channel.active_session_first_seen_s is not None
        ]
        if active_starts:
            return min(active_starts), "active"

        finalized_times = [
            session.final_time_s
            for channel in self._channels
            for session in channel.finalized_sessions
        ]
        if finalized_times:
            return max(finalized_times), "finalized"
        return None

    def prune_before_s(self) -> float | None:
        cutoff = self.prune_cutoff_s()
        return None if cutoff is None else cutoff[0]

    def channel_by_id(self, track_id: int) -> ChannelState | None:
        for channel in self._channels:
            if channel.track_id == track_id:
                return channel
        return None

    def pop_pending_events(self) -> list[StreamEvent]:
        events = self._pending_events
        self._pending_events = []
        return events

    def _with_current_session_id(
        self, channel: ChannelState, session: StreamSessionResult
    ) -> StreamSessionResult:
        return replace(session, session_id=channel.session_id)

    def _ensure_session_started(self, channel: ChannelState, session: StreamSessionResult) -> None:
        state = channel.current_session
        if state.start_event_emitted:
            return
        self._pending_events.append(
            StreamEvent(
                time_s=round(session.first_seen_s, 3),
                kind="SESSION_STARTED",
                channel_id=channel.track_id,
                session_id=session.session_id,
                carrier_hz=round(channel.carrier_hz, 3),
            )
        )
        state.start_event_emitted = True
        state.reset_live_text()

    def _is_stale_decoded_session(self, channel: ChannelState, session: StreamSessionResult) -> bool:
        """Ignore already-finalized audio that is still present in the rolling frame buffer.

        Live decoding repeatedly re-evaluates a retained time window.  Real audio
        can make an old closed segment decode with slightly different text after
        smoothing/tracker drift, so exact text matching is not enough to recognize
        it as already handled.  Once a channel has finalized up to time T, any
        newly decoded segment that ends before T is stale history and must not
        create a fresh session id/event.
        """

        finalized_until_s = channel.finalized_until_s
        if finalized_until_s is None:
            return False

        tolerance_s = max(self.config.hop_ms / 1000 * 6, 0.05)
        return session.last_seen_s <= finalized_until_s + tolerance_s

    def _matching_finalized_session(
        self, channel: ChannelState, session: StreamSessionResult
    ) -> StreamSessionResult | None:
        tolerance_s = max(self.config.hop_ms / 1000 * 3, 0.03)
        for existing in channel.finalized_sessions:
            if (
                abs(existing.first_seen_s - session.first_seen_s) <= tolerance_s
                and abs(existing.last_seen_s - session.last_seen_s) <= tolerance_s
                and existing.decoded.text == session.decoded.text
            ):
                return existing
        return None

    def _finalize_session(self, channel: ChannelState, session: StreamSessionResult) -> None:
        if session.session_id in channel.finalized_session_ids:
            return
        session = channel.current_session.final_session_candidate(session, self.config)
        channel.finalized_session_ids.add(session.session_id)
        if self._should_publish_final_session(session):
            self._pending_events.append(
                StreamEvent(
                    time_s=round(session.final_time_s, 3),
                    kind="SESSION_FINAL",
                    channel_id=channel.track_id,
                    session_id=session.session_id,
                    carrier_hz=round(channel.carrier_hz, 3),
                    text=session.decoded.text,
                    score=session.quality.score,
                    reason=session.final_reason,
                )
            )
            channel.finalized_sessions.append(session)
            channel.finalized_sessions.sort(key=lambda item: item.session_id)
        else:
            self._pending_events.append(
                StreamEvent(
                    time_s=round(session.final_time_s, 3),
                    kind="SESSION_FINAL",
                    channel_id=channel.track_id,
                    session_id=session.session_id,
                    carrier_hz=round(channel.carrier_hz, 3),
                    text="",
                    score=session.quality.score,
                    reason="quality_suppressed",
                )
            )
            channel.suppressed_until_s = max(channel.suppressed_until_s or 0.0, session.final_time_s)
        if channel.session_id <= session.session_id:
            channel.start_next_session()

    def _should_publish_final_session(self, session: StreamSessionResult) -> bool:
        if self.config.max_final_score is None:
            return True
        return session.quality.score <= self.config.max_final_score


def _common_text_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]



def _texts_are_compatible(left: str, right: str) -> bool:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return True
    return left.startswith(right) or right.startswith(left)

def _merge_committed_prefix_and_tail(prefix: str, tail: str) -> tuple[str, int]:
    if not prefix:
        return tail, 0
    if not tail:
        return prefix, 0

    max_overlap = min(len(prefix), len(tail))
    for overlap in range(max_overlap, 0, -1):
        if prefix.endswith(tail[:overlap]):
            return prefix + tail[overlap:], overlap
    return f"{prefix} {tail}".strip(), 0


def _safe_active_prune_prefix(text: str) -> str:
    last_space = text.rstrip().rfind(" ")
    if last_space <= 0:
        return ""
    return text[:last_space].rstrip()


def _tail_prefix_chars_for_merged_prefix(prefix_len: int, base_len: int, overlap_len: int) -> int | None:
    if prefix_len <= 0:
        return None
    if prefix_len <= base_len:
        if overlap_len < prefix_len:
            return None
        return prefix_len
    return overlap_len + prefix_len - base_len


def _decoded_text_after_s(decoded: DecodeResult, after_s: float | None) -> str:
    tokens = [token for token, end_s in _decoded_tokens_with_end_times(decoded) if after_s is None or end_s > after_s]
    return decode_tokens(tokens)


def _decoded_tail_prefix_end_s(decoded: DecodeResult, char_count: int, after_s: float | None = None) -> float | None:
    if char_count <= 0:
        return None

    tokens: list[str] = []
    for token, end_s in _decoded_tokens_with_end_times(decoded):
        if after_s is not None and end_s <= after_s:
            continue
        tokens.append(token)
        if len(decode_tokens(tokens)) >= char_count:
            return end_s
    return None


def _decoded_tokens_with_end_times(decoded: DecodeResult) -> list[tuple[str, float]]:
    tokens: list[tuple[str, float]] = []
    current = ""
    current_end_s: float | None = None

    def append_current() -> None:
        nonlocal current, current_end_s
        if not current:
            return
        tokens.append((current, current_end_s or 0.0))
        current = ""
        current_end_s = None

    for run in decoded.classified_runs:
        if run.kind == "tone":
            current += run.symbol
            current_end_s = run.start_s + run.duration_s
            continue

        if run.symbol not in {"letter_gap", "word_gap"}:
            continue

        append_current()
        if run.symbol == "word_gap":
            tokens.append(("/", run.start_s + run.duration_s))

    append_current()
    return tokens
