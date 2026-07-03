import numpy as np

from cw.generator import GeneratorConfig, build_events, render_wave
from cw.nextgen_stream import NextgenStreamProcessor
from cw.stream_models import StreamingConfig


def _push_signal(processor: NextgenStreamProcessor, signal: np.ndarray, block_samples: int = 80):
    events = []
    for start in range(0, len(signal), block_samples):
        events.extend(processor.push(signal[start : start + block_samples]).events)
    events.extend(processor.finish().events[len(events) :])
    return events


def test_nextgen_stream_emits_lifecycle_events_for_raw_cq() -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.7)
    signal = render_wave(build_events("CQ CQ", config), config)
    signal = np.concatenate([signal, np.zeros(int(config.sample_rate * 0.8), dtype=np.float32)])
    processor = NextgenStreamProcessor(
        config.sample_rate,
        StreamingConfig(
            max_tone_hz=3000,
            threshold_ratios=(0.20, 0.25, 0.30, 0.35),
            merge_short_gaps_ms=25.0,
            drop_short_tones_ms=12.0,
            unit_candidate_spread=0.12,
            unit_candidate_steps=5,
            min_keying_chars=2,
            min_keying_known_chars=2,
            emit_interval_s=0.5,
            finalization_delay_s=0.4,
        ),
    )

    events = _push_signal(processor, signal)
    kinds = [event.kind for event in events]

    assert "CHANNEL_STARTED" in kinds
    assert "SESSION_STARTED" in kinds
    assert "TEXT_COMMITTED" in kinds
    assert any(event.kind == "SESSION_FINAL" and "CQ" in event.text for event in events)


def test_nextgen_stream_suppresses_silence() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(max_tone_hz=3000, threshold_ratios=(0.25,), emit_interval_s=0.2),
    )
    signal = np.zeros(8000, dtype=np.float32)

    events = _push_signal(processor, signal)

    assert events == []


def test_nextgen_live_stable_updates_wait_for_repeated_prefix() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(stable_updates=True, min_update_score=25.0, min_live_commit_chars=2),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type(
            "DecodedSessionStub",
            (),
            {"start_s": 0.0, "end_s": 1.0},
        )(),
    )
    assert session is not None

    assert processor._text_to_commit(session, "C", 1.0) is None
    assert processor._text_to_commit(session, "CM", 1.0) is None
    assert processor._text_to_commit(session, "CQ", 1.0) is None
    assert processor._text_to_commit(session, "CQ D", 1.0) == "CQ"
    assert processor._text_to_commit(session, "CQT", 1.0) is None
    assert processor._text_to_commit(session, "CQ DE", 1.0) is None
    assert processor._text_to_commit(session, "CQ DE H", 1.0) == "CQ DE"


def test_nextgen_live_raw_updates_still_emit_immediate_candidates() -> None:
    processor = NextgenStreamProcessor(8000, StreamingConfig(stable_updates=False))
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type(
            "DecodedSessionStub",
            (),
            {"start_s": 0.0, "end_s": 1.0},
        )(),
    )
    assert session is not None

    assert processor._text_to_commit(session, "C", 1.0) == "C"
    assert processor._text_to_commit(session, "CQ", 1.0) == "CQ"


def test_nextgen_stream_runner_emits_json_events_as_chunks_are_processed() -> None:
    from types import SimpleNamespace

    from cw.cli_stream import _run_nextgen_stream
    from cw.stream_sources import ArrayAudioSource

    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.7)
    signal = render_wave(build_events("CQ CQ", config), config)
    signal = np.concatenate([signal, np.zeros(int(config.sample_rate * 0.8), dtype=np.float32)])
    source = ArrayAudioSource(signal, config.sample_rate, block_ms=10)
    stream_config = StreamingConfig(
        max_tone_hz=3000,
        threshold_ratios=(0.20, 0.25, 0.30, 0.35),
        merge_short_gaps_ms=25.0,
        drop_short_tones_ms=12.0,
        unit_candidate_spread=0.12,
        unit_candidate_steps=5,
        min_keying_chars=2,
        min_keying_known_chars=2,
        emit_interval_s=0.5,
        finalization_delay_s=0.4,
    )
    emitted = []

    result = _run_nextgen_stream(
        source,
        stream_config,
        SimpleNamespace(events=False, live_stats_interval_s=0.0, no_finalize_on_interrupt=False),
        json_events=True,
        event_sink=emitted.append,
    )

    assert emitted == result.events
    assert any(event.kind == "TEXT_COMMITTED" for event in emitted)


def test_nextgen_live_keeps_stronger_final_candidate_over_late_weaker_incompatible_text() -> None:
    from types import SimpleNamespace

    processor = NextgenStreamProcessor(8000, StreamingConfig())
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        SimpleNamespace(start_s=0.0, end_s=1.0),
    )
    assert session is not None

    processor._remember_final_candidate(
        session,
        SimpleNamespace(text="CQ OG50YL", best=SimpleNamespace(evidence_score=28.0)),
        4.0,
    )
    processor._remember_final_candidate(
        session,
        SimpleNamespace(text="TT EEE DA", best=SimpleNamespace(evidence_score=15.0)),
        2.0,
    )

    assert session.final_text == "CQ OG50YL"
    assert session.final_score == 4.0


def test_nextgen_live_finalization_waits_for_settle_interval() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(emit_interval_s=0.5, finalization_delay_s=0.0),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.final_text = "CQ"
    session.final_score = 1.0

    processor.processed_duration_s = 2.0
    processor._maybe_emit_pending_final(channel, session)
    assert not session.finalized
    assert not any(event.kind == "SESSION_FINAL" for event in processor._events)

    processor.processed_duration_s = 2.6
    processor._maybe_emit_pending_final(channel, session)
    assert session.finalized
    assert any(event.kind == "SESSION_FINAL" and event.text == "CQ" for event in processor._events)


def test_nextgen_live_accepts_longer_final_with_strong_shared_prefix() -> None:
    from types import SimpleNamespace

    processor = NextgenStreamProcessor(8000, StreamingConfig())
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        SimpleNamespace(start_s=0.0, end_s=1.0),
    )
    assert session is not None

    processor._remember_final_candidate(
        session,
        SimpleNamespace(text="CQ DE YU7ND", best=SimpleNamespace(evidence_score=45.5)),
        6.6,
    )
    processor._remember_final_candidate(
        session,
        SimpleNamespace(text="CQ DE YU7NKA", best=SimpleNamespace(evidence_score=48.5)),
        6.4,
    )

    assert session.final_text == "CQ DE YU7NKA"
    assert session.final_score == 6.4


def test_nextgen_live_stitches_rolling_window_progress_for_long_active_session() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(
            stable_updates=True,
            min_update_score=25.0,
            min_live_commit_chars=2,
            live_progress_interval_s=3.0,
            live_progress_min_overlap_chars=3,
        ),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.committed_text = "CQ CQ DE M0PKD"
    session.last_candidate_text = "CQ CQ DE M0PKD"
    session.last_commit_s = 10.0

    processor.processed_duration_s = 12.0
    assert processor._text_to_commit(session, "DE M0PKD OP NAME", 2.0) is None

    processor.processed_duration_s = 13.5
    assert processor._text_to_commit(session, "DE M0PKD OP NAME", 2.0) == "CQ CQ DE M0PKD OP NAME"


def test_nextgen_live_does_not_stitch_unrelated_rolling_text() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(
            stable_updates=True,
            live_progress_interval_s=1.0,
            live_progress_min_overlap_chars=3,
        ),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.committed_text = "CQ CQ DE M0PKD"
    session.last_candidate_text = "CQ CQ DE M0PKD"
    session.last_commit_s = 0.0
    processor.processed_duration_s = 2.0

    assert processor._text_to_commit(session, "TU 5NN BK", 2.0) is None
    assert session.committed_text == "CQ CQ DE M0PKD"


def test_nextgen_live_confirms_channel_after_min_track_hits() -> None:
    processor = NextgenStreamProcessor(8000, StreamingConfig(min_track_hits=2))
    processor.processed_duration_s = 1.0

    channel = processor._channel_for_carrier(700.0)
    assert channel.hits == 1
    assert not channel.channel_started
    assert not any(event.kind == "CHANNEL_STARTED" for event in processor._events)

    processor.processed_duration_s = 1.5
    same_channel = processor._channel_for_carrier(702.0)
    assert same_channel is channel
    assert channel.hits == 2
    assert channel.channel_started
    assert [event.kind for event in processor._events].count("CHANNEL_STARTED") == 1


def test_nextgen_live_inactive_channel_finalizes_and_goes_dormant() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(min_track_hits=1, max_track_gap_s=1.0, emit_interval_s=0.5),
    )
    processor.processed_duration_s = 0.5
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.final_text = "CQ TEST"
    session.final_score = 1.0
    channel.last_seen_s = 0.5

    processor.processed_duration_s = 7.0
    processor._finalize_inactive_channels()

    assert session.finalized
    assert channel.dormant
    assert any(event.kind == "SESSION_FINAL" and event.reason == "channel_inactive" for event in processor._events)
    assert any(event.kind == "CHANNEL_DORMANT" and event.reason == "channel_inactive" for event in processor._events)


def test_nextgen_live_reuses_recent_channel_with_wider_reacquire_tolerance() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(
            min_track_hits=1,
            channel_merge_hz=60.0,
            channel_reacquire_hz=90.0,
            channel_reacquire_s=10.0,
        ),
    )
    processor.processed_duration_s = 1.0
    channel = processor._channel_for_carrier(700.0)

    processor.processed_duration_s = 5.0
    same_channel = processor._channel_for_carrier(760.0)

    assert same_channel is channel
    assert len(processor._channels) == 1


def test_nextgen_live_reacquired_dormant_channel_requires_confirmation_again() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(
            min_track_hits=2,
            max_track_gap_s=1.0,
            emit_interval_s=0.5,
            channel_reacquire_hz=90.0,
            channel_reacquire_s=15.0,
        ),
    )
    processor.processed_duration_s = 0.5
    channel = processor._channel_for_carrier(700.0)
    processor.processed_duration_s = 1.0
    assert processor._channel_for_carrier(702.0) is channel
    assert channel.channel_started

    channel.last_seen_s = 1.0
    processor.processed_duration_s = 7.5
    processor._finalize_inactive_channels()
    assert channel.dormant
    assert channel.hits == 0

    processor.processed_duration_s = 8.0
    assert processor._channel_for_carrier(760.0) is channel
    assert channel.hits == 1
    assert channel.dormant
    assert [event.kind for event in processor._events].count("CHANNEL_STARTED") == 1

    processor.processed_duration_s = 8.5
    assert processor._channel_for_carrier(758.0) is channel
    assert channel.hits == 2
    assert not channel.dormant
    assert [event.kind for event in processor._events].count("CHANNEL_STARTED") == 2


def test_nextgen_live_keeps_stable_committed_text_over_much_worse_long_final() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(final_text_regression_margin=10.0),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.committed_text = "CQ CQ CQ"
    session.committed_score = 3.5
    session.final_text = "ST EK C Q C Q C Q C NE E H E"
    session.final_score = 28.0
    session.final_evidence = 50.0

    text, score = processor._final_text_for(session)

    assert text == "CQ CQ CQ"
    assert score == 3.5


def test_nextgen_live_symbol_hmm_is_budgeted_per_carrier() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(symbol_hmm_decoding=True, live_symbol_hmm_decoding=True, symbol_hmm_live_interval_s=2.0),
    )
    channel = processor._channel_for_carrier(700.0)

    processor.processed_duration_s = 1.0
    assert processor._config_for_channel_decode(channel, final=False).symbol_hmm_decoding
    processor.processed_duration_s = 1.5
    assert not processor._config_for_channel_decode(channel, final=False).symbol_hmm_decoding
    processor.processed_duration_s = 3.1
    assert processor._config_for_channel_decode(channel, final=False).symbol_hmm_decoding
    processor.processed_duration_s = 3.2
    assert processor._config_for_channel_decode(channel, final=True).symbol_hmm_decoding


def test_nextgen_live_emits_text_preview_before_stable_commit() -> None:
    from types import SimpleNamespace

    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(
            stable_updates=True,
            preview_updates=True,
            preview_interval_s=0.0,
            preview_min_chars=1,
            min_update_score=25.0,
            min_live_commit_chars=2,
        ),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(channel, SimpleNamespace(start_s=0.0, end_s=1.0))
    assert session is not None

    assert processor._text_to_commit(session, "C", 1.0) is None
    processor._maybe_emit_text_preview(channel, session, "C", 1.0, reason="awaiting_stable_prefix")

    assert any(event.kind == "TEXT_PREVIEW" and event.text == "C" for event in processor._events)


def test_nextgen_live_emits_signal_active_heartbeat_without_text() -> None:
    from types import SimpleNamespace

    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(min_track_hits=1, signal_activity_interval_s=1.0),
    )
    processor.processed_duration_s = 2.0
    channel = processor._channel_for_carrier(700.0)

    processor._maybe_emit_signal_active(
        channel,
        SimpleNamespace(best=SimpleNamespace(quality_score=12.0)),
        reason="carrier_detected",
    )

    assert any(
        event.kind == "SIGNAL_ACTIVE"
        and event.channel_id == channel.channel_id
        and event.reason == "carrier_detected"
        for event in processor._events
    )


def test_nextgen_live_defaults_to_short_decode_window() -> None:
    processor = NextgenStreamProcessor(8000, StreamingConfig(live_decode_window_s=2.5))
    processor._window = __import__('numpy').ones(8000 * 8, dtype=__import__('numpy').float32)
    signal, start_s = processor._decode_window()
    assert len(signal) == 8000 * 2 + 4000
    assert start_s == 5.5


def test_nextgen_live_symbol_hmm_is_opt_in_for_low_latency() -> None:
    processor = NextgenStreamProcessor(8000, StreamingConfig(symbol_hmm_decoding=True))
    channel = processor._channel_for_carrier(700.0)
    processor.processed_duration_s = 1.0
    assert not processor._config_for_channel_decode(channel, final=False).symbol_hmm_decoding


def test_nextgen_live_final_uses_preview_suffix_with_small_leading_error() -> None:
    processor = NextgenStreamProcessor(8000, StreamingConfig(live_progress_min_overlap_chars=3))
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.committed_text = "HA7VY"
    session.committed_score = 7.9
    session.final_text = "HA7VY 5E"
    session.final_score = 7.9
    session.last_preview_text = "T7VY 5NN"
    session.last_preview_score = 3.5
    session.best_preview_text = "T7VY 5NN"
    session.best_preview_score = 3.5

    text, score = processor._final_text_for(session)

    assert text == "HA7VY 5NN"
    assert score == 3.5


def test_nextgen_live_progress_stitches_preview_with_small_leading_error() -> None:
    processor = NextgenStreamProcessor(
        8000,
        StreamingConfig(
            stable_updates=True,
            live_progress_interval_s=1.0,
            live_progress_min_overlap_chars=3,
            min_update_score=25.0,
        ),
    )
    channel = processor._channel_for_carrier(700.0)
    session = processor._session_for_decoded(
        channel,
        type("DecodedSessionStub", (), {"start_s": 0.0, "end_s": 1.0})(),
    )
    assert session is not None
    session.committed_text = "HA7VY"
    session.last_candidate_text = "HA7VY"
    session.last_commit_s = 0.0
    processor.processed_duration_s = 2.0

    assert processor._text_to_commit(session, "T7VY 5NN", 3.5) == "HA7VY 5NN"


def test_nextgen_live_arbiter_prefers_cleaner_candidate_when_evidence_is_tied() -> None:
    from cw.nextgen import NextgenCandidate, NextgenSession
    from cw.nextgen_stream import _LiveSessionHypothesisArbiter

    def cand(text: str, quality: float, evidence: float) -> NextgenCandidate:
        return NextgenCandidate(
            carrier_hz=700.0,
            detector="threshold",
            threshold_ratio=0.20,
            threshold=0.0,
            noise_floor=0.0,
            signal_floor=1.0,
            duty_cycle=0.5,
            unit_s=0.06,
            wpm=20.0,
            text=text,
            tokens=(),
            quality_score=quality,
            confidence=0.9,
            evidence_score=evidence,
            start_s=0.0,
            end_s=3.0,
            runs=(),
        )

    noisy = cand("CQ KQ TE T", quality=2.29, evidence=38.94)
    cleaner = cand("CQ CQ TE T", quality=1.98, evidence=38.88)
    session = NextgenSession(
        carrier_hz=700.0,
        session_id=1,
        start_s=0.0,
        end_s=3.0,
        text=noisy.text,
        confidence=noisy.confidence,
        best=noisy,
        candidates=(noisy, cleaner),
    )

    selected = _LiveSessionHypothesisArbiter().choose(session)

    assert selected.text == "CQ CQ TE T"
    assert selected.best is cleaner
