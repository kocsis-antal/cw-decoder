from __future__ import annotations

from dataclasses import dataclass, field, replace

from cw.stream_models import StreamEvent, StreamingConfig, StreamSessionResult


@dataclass
class SessionState:
    """Mutable live state for one transmission within a carrier channel.

    The decoded timing/tempo belongs to the StreamSessionResult produced by the
    decoder. This state only keeps live bookkeeping: whether the session-start
    event has already been emitted, which prefix was stable enough to commit,
    and whether the session has already been finalized.
    """

    session_id: int
    first_seen_s: float | None = None
    start_event_emitted: bool = False
    last_candidate_text: str = ""
    committed_text: str = ""

    def mark_observed(self, session: StreamSessionResult) -> None:
        if self.first_seen_s is None:
            self.first_seen_s = session.first_seen_s

    def reset_live_text(self) -> None:
        self.last_candidate_text = ""
        self.committed_text = ""

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

    @property
    def session_id(self) -> int:
        return self.current_session.session_id

    @property
    def active_session_first_seen_s(self) -> float | None:
        return self.current_session.first_seen_s

    def start_next_session(self) -> None:
        self.current_session = SessionState(self.current_session.session_id + 1)

    def commit_text_candidate(self, current_text: str, score: float, config: StreamingConfig) -> str | None:
        return self.current_session.text_to_commit(current_text, score, config)


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
        max_match_hz = max(self.config.min_separation_hz / 2, self.config.bandwidth_hz)
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
        channel.current_session.first_seen_s = None

        for decoded_session in sessions:
            existing = self._matching_finalized_session(channel, decoded_session)
            if existing is not None:
                if channel.session_id <= existing.session_id:
                    channel.current_session = SessionState(existing.session_id + 1)
                continue

            session = self._with_current_session_id(channel, decoded_session)
            self._ensure_session_started(channel, session)
            channel.current_session.mark_observed(session)

            if session.final_reason == "silence_gap":
                self._finalize_session(channel, session)
            elif session.final_reason == self.config.final_event_reason:
                active_session = session

        return active_session

    def prune_before_s(self) -> float | None:
        active_starts = [
            channel.active_session_first_seen_s
            for channel in self._channels
            if channel.active_session_first_seen_s is not None
        ]
        if active_starts:
            return min(active_starts)

        finalized_times = [
            session.final_time_s
            for channel in self._channels
            for session in channel.finalized_sessions
        ]
        if finalized_times:
            return max(finalized_times)
        return None

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
        channel.finalized_session_ids.add(session.session_id)
        channel.finalized_sessions.append(session)
        channel.finalized_sessions.sort(key=lambda item: item.session_id)
        if channel.session_id <= session.session_id:
            channel.start_next_session()


def _common_text_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]
