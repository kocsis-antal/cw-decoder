from pathlib import Path

import numpy as np

from cw.multi_generator import parse_source_spec, write_multi_sample
from cw.streaming import StreamingConfig, StreamingSTFT, simulate_stream_from_wav


def test_streaming_stft_links_split_input_with_overlapping_frames() -> None:
    sample_rate = 1000
    stft = StreamingSTFT(sample_rate=sample_rate, frame_ms=20, hop_ms=10)
    signal = np.ones(50, dtype=np.float32)

    first = stft.push(signal[:15])
    second = stft.push(signal[15:])

    assert first == []
    assert len(second) >= 3
    assert second[0].start_s == 0.0
    assert round(second[1].start_s, 3) == 0.01


def test_stream_sim_decodes_two_generated_sources(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;preset=straight;text=CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(wav_path, StreamingConfig(max_tracks=3, emit_interval_s=0.5))

    texts = {track.decoded.text for track in result.tracks}
    assert "CQ DE YU7NKA" in texts
    assert "CQ DE YT7MK" in texts
    assert result.updates


def test_stream_sim_stable_updates_are_prefixes_of_final_text(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;preset=straight;text=CQ CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(wav_path, StreamingConfig(max_tracks=3, emit_interval_s=0.5))

    final_text_by_track = {track.track_id: track.decoded.text for track in result.tracks}
    assert result.updates
    for update in result.updates:
        assert final_text_by_track[update.track_id].startswith(update.text)


def test_stream_sim_raw_updates_can_show_unstable_candidates(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;preset=straight;text=CQ CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(max_tracks=3, emit_interval_s=0.5, stable_updates=False),
    )

    final_text_by_track = {track.track_id: track.decoded.text for track in result.tracks}
    assert any(
        not final_text_by_track.get(update.track_id, "").startswith(update.text)
        for update in result.updates
    )


def test_stream_sim_emits_channel_and_session_lifecycle_events(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;preset=straight;text=CQ CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(wav_path, StreamingConfig(max_tracks=3, emit_interval_s=0.5))

    kinds = [event.kind for event in result.events]
    assert "CHANNEL_STARTED" in kinds
    assert "SESSION_STARTED" in kinds
    assert "TEXT_COMMITTED" in kinds
    assert "SESSION_FINAL" in kinds
    assert "CHANNEL_DORMANT" in kinds

    final_events = [event for event in result.events if event.kind == "SESSION_FINAL"]
    final_texts = {track.decoded.text for track in result.tracks}
    assert {event.text for event in final_events} == final_texts
    assert all(event.session_id == 1 for event in final_events)


def test_stream_sim_final_event_flushes_complete_session_text(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;preset=straight;text=CQ CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(wav_path, StreamingConfig(max_tracks=3, emit_interval_s=0.5))

    committed_by_channel: dict[int, str] = {}
    for event in result.events:
        if event.kind == "TEXT_COMMITTED":
            committed_by_channel[event.channel_id] = event.text

    for event in result.events:
        if event.kind != "SESSION_FINAL":
            continue
        committed = committed_by_channel.get(event.channel_id, "")
        assert event.text.startswith(committed)
        assert event.reason == "end_of_stream"


def test_stream_sim_splits_same_channel_into_sessions_after_long_silence(tmp_path: Path) -> None:
    import soundfile as sf

    from cw.generator import GeneratorConfig, build_events, render_wave

    sample_rate = 8000
    config = GeneratorConfig(sample_rate=sample_rate, tone_hz=700.0)
    first = render_wave(build_events("CQ", config), config)
    second = render_wave(build_events("DE", config), config)
    gap = np.zeros(int(sample_rate * 2.0), dtype=np.float32)
    wav_path = tmp_path / "two_sessions.wav"
    sf.write(wav_path, np.concatenate([first, gap, second]), sample_rate)

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(max_tracks=1, emit_interval_s=0.5, min_session_gap_s=1.0, session_gap_units=12),
    )

    assert len(result.tracks) == 1
    sessions = result.tracks[0].sessions
    assert [session.decoded.text for session in sessions] == ["CQ", "DE"]
    assert sessions[0].final_reason == "silence_gap"
    assert sessions[1].final_reason == "end_of_stream"

    final_events = [event for event in result.events if event.kind == "SESSION_FINAL"]
    assert [event.session_id for event in final_events] == [1, 2]
    assert [event.text for event in final_events] == ["CQ", "DE"]


def test_stream_sim_can_use_longer_tracker_fft_than_decode_fft(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=straight;text=CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=820;preset=straight;text=CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(
        wav_path,
        StreamingConfig(
            frame_ms=20,
            hop_ms=5,
            tracker_frame_ms=80,
            tracker_hop_ms=10,
            max_tracks=3,
            emit_interval_s=0.5,
        ),
    )

    texts = {track.decoded.text for track in result.tracks}
    assert "CQ DE YU7NKA" in texts
    assert "CQ DE YT7MK" in texts
    assert result.frames_processed > result.tracker_frames_processed > 0


def test_stream_squelch_suppresses_white_noise() -> None:
    import numpy as np

    from cw.streaming import StreamingConfig, simulate_stream

    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 0.02, 8000 * 4).astype(np.float32)

    result = simulate_stream(
        noise,
        8000,
        StreamingConfig(input_block_ms=10.0, min_peak_snr_db=14.0),
    )

    assert result.tracks == []
    assert result.events == []
