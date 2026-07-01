from pathlib import Path

from cw.qso_generator import ContestQsoConfig, build_contest_qso_sources, write_contest_qso_sample
from cw.streaming import StreamingConfig, simulate_stream_from_wav


def test_build_contest_qso_sources_creates_sequential_turns() -> None:
    sources = build_contest_qso_sources(
        ContestQsoConfig(caller_call="YU7NKA", responder_call="YT7MK", seed=123)
    )

    assert [source.text for source in sources] == [
        "CQ TEST YU7NKA",
        "YU7NKA YT7MK",
        "YT7MK 599 001",
        "TU 599 002",
        "TU",
    ]
    assert [source.start_s for source in sources] == sorted(source.start_s for source in sources)
    assert sources[0].config.tone_hz == 700.0
    assert sources[1].config.tone_hz == 706.0
    assert sources[0].config.wpm != sources[1].config.wpm


def test_stream_sim_splits_contest_qso_into_sessions(tmp_path: Path) -> None:
    wav_path = tmp_path / "contest_qso.wav"
    write_contest_qso_sample(
        wav_path,
        ContestQsoConfig(caller_call="YU7NKA", responder_call="YT7MK", turn_gap_s=1.6, seed=123),
    )

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(
            max_tracks=2,
            min_separation_hz=80,
            emit_interval_s=0.5,
            min_session_gap_s=1.0,
            session_gap_units=12,
        ),
    )

    assert len(result.tracks) == 1
    sessions = result.tracks[0].sessions
    assert [session.decoded.text for session in sessions] == [
        "CQ TEST YU7NKA",
        "YU7NKA YT7MK",
        "YT7MK 599 001",
        "TU 599 002",
        "TU",
    ]
    assert [session.final_reason for session in sessions[:-1]] == ["silence_gap"] * 4
    assert sessions[-1].final_reason == "end_of_stream"

    final_events = [event for event in result.events if event.kind == "SESSION_FINAL"]
    assert [event.session_id for event in final_events] == [1, 2, 3, 4, 5]
    assert [event.text for event in final_events] == [session.decoded.text for session in sessions]

    started_session_ids = [event.session_id for event in result.events if event.kind == "SESSION_STARTED"]
    assert started_session_ids == [1, 2, 3, 4, 5]
    # The very short closing TU may only appear as the SESSION_FINAL flush,
    # but longer overs should have live committed updates.
    assert {1, 2, 3, 4}.issubset({update.session_id for update in result.updates})


def test_stream_sim_handles_two_parallel_contest_qsos(tmp_path: Path) -> None:
    wav_path = tmp_path / "two_parallel_qsos.wav"
    first = build_contest_qso_sources(
        ContestQsoConfig(
            caller_call="YU7NKA",
            responder_call="YT7MK",
            base_frequency_hz=700.0,
            responder_offset_hz=6.0,
            seed=123,
            source_prefix="qso1",
        )
    )
    second = build_contest_qso_sources(
        ContestQsoConfig(
            caller_call="HA5ABC",
            responder_call="HA7XYZ",
            base_frequency_hz=1000.0,
            responder_offset_hz=6.0,
            start_s=0.4,
            seed=999,
            source_prefix="qso2",
            caller_amplitude=0.50,
            responder_amplitude=0.45,
        )
    )

    from cw.multi_generator import write_multi_sample

    write_multi_sample(first + second, wav_path, sample_rate=8000, normalize_peak=0.95, noise_snr_db=22.0, seed=123)

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(
            max_tracks=4,
            min_separation_hz=120,
            emit_interval_s=0.5,
            min_session_gap_s=1.0,
            session_gap_units=12,
        ),
    )

    assert len(result.tracks) == 2
    session_texts_by_track = [[session.decoded.text for session in track.sessions] for track in result.tracks]
    flattened = {text for texts in session_texts_by_track for text in texts}
    assert "CQ TEST YU7NKA" in flattened
    assert "YU7NKA YT7MK" in flattened
    assert "CQ TEST HA5ABC" in flattened
    assert "HA5ABC HA7XYZ" in flattened


def test_stream_sim_discards_finalized_session_frame_history(tmp_path: Path) -> None:
    wav_path = tmp_path / "contest_qso.wav"
    write_contest_qso_sample(
        wav_path,
        ContestQsoConfig(caller_call="YU7NKA", responder_call="YT7MK", turn_gap_s=1.6, seed=123),
    )

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(
            max_tracks=2,
            min_separation_hz=80,
            emit_interval_s=0.5,
            min_session_gap_s=1.0,
            session_gap_units=12,
            prune_finalized_sessions=True,
        ),
    )

    assert result.pruned_frames > 0
    assert result.retained_frames < result.frames_processed
    assert [session.decoded.text for session in result.tracks[0].sessions] == [
        "CQ TEST YU7NKA",
        "YU7NKA YT7MK",
        "YT7MK 599 001",
        "TU 599 002",
        "TU",
    ]


def test_stream_sim_can_keep_full_frame_history_for_debugging(tmp_path: Path) -> None:
    wav_path = tmp_path / "contest_qso.wav"
    write_contest_qso_sample(
        wav_path,
        ContestQsoConfig(caller_call="YU7NKA", responder_call="YT7MK", turn_gap_s=1.6, seed=123),
    )

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(
            max_tracks=2,
            min_separation_hz=80,
            emit_interval_s=0.5,
            min_session_gap_s=1.0,
            session_gap_units=12,
            prune_finalized_sessions=False,
        ),
    )

    assert result.pruned_frames == 0
    assert result.retained_frames == result.frames_processed
    assert [session.decoded.text for session in result.tracks[0].sessions] == [
        "CQ TEST YU7NKA",
        "YU7NKA YT7MK",
        "YT7MK 599 001",
        "TU 599 002",
        "TU",
    ]
