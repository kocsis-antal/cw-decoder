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


def _session(text: str, *, first: float, last: float, reason: str, score: float = 1.0) -> StreamSessionResult:
    return StreamSessionResult(
        session_id=1,
        first_seen_s=first,
        last_seen_s=last,
        hits=10,
        final_time_s=last + 1.0,
        final_reason=reason,
        quality=_quality(score),
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


def test_channel_registry_ignores_stale_redecoded_history_with_different_text() -> None:
    config = StreamingConfig()
    registry = ChannelRegistry(config)
    channel = registry.channel_for(700.0, time_s=0.0)
    registry.pop_pending_events()

    registry.sync_sessions(channel, [_session("CQ", first=1.0, last=2.0, reason="silence_gap")])
    first_final_events = registry.pop_pending_events()
    assert [(event.kind, event.session_id, event.text) for event in first_final_events] == [("SESSION_FINAL", 1, "CQ")]
    assert channel.session_id == 2

    # A long retained live window can decode the same old audio slightly
    # differently after carrier smoothing/deglitching.  It must not become a new
    # live session just because the text no longer matches exactly.
    stale_variant = _session("C=", first=1.0, last=2.01, reason="silence_gap")
    assert registry.sync_sessions(channel, [stale_variant]) is None
    assert registry.pop_pending_events() == []
    assert channel.session_id == 2


def test_session_state_keeps_stable_live_text_when_final_decode_regresses() -> None:
    config = StreamingConfig(stable_updates=False, final_text_regression_margin=10.0)
    state = SessionState(session_id=1)

    stable = _session("OK1GC", first=1.0, last=2.0, reason="end_of_stream", score=5.0)
    assert state.commit_from_session(stable, config) == "OK1GC"

    regressed = _session("E EH E J EE", first=1.0, last=4.0, reason="silence_gap", score=50.0)
    final = state.final_session_candidate(regressed, config)

    assert final.decoded.text == "OK1GC"
    assert final.quality.score == 5.0


def test_session_state_keeps_better_final_decode_even_when_text_differs() -> None:
    config = StreamingConfig(stable_updates=False, final_text_regression_margin=10.0)
    state = SessionState(session_id=1)

    stable = _session("73TIE", first=1.0, last=2.0, reason="end_of_stream", score=25.0)
    assert state.commit_from_session(stable, config) == "73TIE"

    corrected = _session("73TU", first=1.0, last=4.0, reason="silence_gap", score=5.0)
    final = state.final_session_candidate(corrected, config)

    assert final.decoded.text == "73TU"
    assert final.quality.score == 5.0


def test_channel_registry_suppresses_bad_final_session_but_uses_it_for_stale_cutoff() -> None:
    config = StreamingConfig(max_final_score=30.0)
    registry = ChannelRegistry(config)
    channel = registry.channel_for(700.0, time_s=0.0)
    registry.pop_pending_events()

    bad = _session("E E E E", first=1.0, last=2.0, reason="silence_gap", score=80.0)
    registry.sync_sessions(channel, [bad])

    final_events = registry.pop_pending_events()
    assert [(event.kind, event.reason, event.text) for event in final_events] == [("SESSION_FINAL", "quality_suppressed", "")]
    assert channel.session_id == 2
    assert channel.finalized_sessions == []
    assert channel.finalized_until_s == bad.final_time_s

    stale_variant = _session("T T T", first=1.0, last=2.01, reason="silence_gap", score=10.0)
    assert registry.sync_sessions(channel, [stale_variant]) is None
    assert registry.pop_pending_events() == []
    assert channel.session_id == 2
