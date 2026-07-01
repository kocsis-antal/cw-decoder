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
