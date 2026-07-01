from cw.decoder import DecodeResult
from cw.quality import QualityScore
from cw.stream_models import StreamingConfig, StreamSessionResult
from cw.stream_state import ChannelRegistry, SessionState


def _quality(score: float = 1.0) -> QualityScore:
    return QualityScore(
        score=score,
        unknown_count=0,
        token_count=1,
        dot_count=1,
        dash_count=0,
        tone_ratio_error=0.0,
        gap_min_error=0.0,
        unit_cv=0.0,
    )


def _session(text: str, *, first: float, last: float, reason: str) -> StreamSessionResult:
    return StreamSessionResult(
        session_id=1,
        first_seen_s=first,
        last_seen_s=last,
        hits=10,
        final_time_s=last + 1.0,
        final_reason=reason,
        quality=_quality(),
        decoded=DecodeResult(
            text=text,
            tokens=[],
            runs=[],
            classified_runs=[],
            carrier_hz=700.0,
            unit_s=0.060,
            threshold=1.0,
        ),
    )


def test_session_state_commits_only_stable_prefix() -> None:
    config = StreamingConfig(min_update_score=25.0)
    state = SessionState(session_id=1)

    assert state.text_to_commit("CQ", score=1.0, config=config) is None
    assert state.text_to_commit("CQ D", score=1.0, config=config) == "CQ"
    assert state.committed_text == "CQ"
    assert state.text_to_commit("CQ DE", score=100.0, config=config) is None


def test_channel_registry_finalizes_session_and_starts_clean_next_session() -> None:
    config = StreamingConfig()
    registry = ChannelRegistry(config)
    channel = registry.channel_for(700.0, time_s=0.0)

    startup_events = registry.pop_pending_events()
    assert [event.kind for event in startup_events] == ["CHANNEL_STARTED", "SESSION_STARTED"]
    assert channel.session_id == 1

    active = registry.sync_sessions(channel, [_session("CQ", first=0.0, last=1.0, reason="end_of_stream")])
    assert active is not None
    assert active.session_id == 1
    assert channel.active_session_first_seen_s == 0.0

    registry.sync_sessions(channel, [_session("CQ", first=0.0, last=1.0, reason="silence_gap")])
    final_events = registry.pop_pending_events()
    assert [(event.kind, event.session_id, event.text) for event in final_events] == [("SESSION_FINAL", 1, "CQ")]
    assert channel.session_id == 2
    assert channel.current_session.committed_text == ""
    assert channel.current_session.last_candidate_text == ""

    second_active = registry.sync_sessions(channel, [_session("DE", first=2.5, last=3.0, reason="end_of_stream")])
    session_events = registry.pop_pending_events()
    assert second_active is not None
    assert second_active.session_id == 2
    assert [(event.kind, event.session_id) for event in session_events] == [("SESSION_STARTED", 2)]


def test_channel_registry_uses_channel_merge_width_for_reacquisition() -> None:
    from cw.stream_models import StreamingConfig
    from cw.stream_state import ChannelRegistry

    registry = ChannelRegistry(StreamingConfig(channel_merge_hz=100.0, bandwidth_hz=20.0))
    first = registry.channel_for(700.0, 0.0)
    same = registry.channel_for(745.0, 1.0)
    other = registry.channel_for(760.0, 2.0)

    assert same.track_id == first.track_id
    assert other.track_id != first.track_id
