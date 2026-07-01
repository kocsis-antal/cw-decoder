from pathlib import Path

import pytest

from cw.decoder import read_wav_mono
from cw.multi_generator import parse_source_spec, write_multi_sample
from cw.streaming import StreamProcessor, StreamingConfig, simulate_stream_from_wav


def _write_two_source_sample(path: Path) -> None:
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec(
            "id=two;freq=1000;preset=straight;text=CQ CQ DE YT7MK;start=0.2;amplitude=0.45",
            index=1,
            sample_rate=8000,
        ),
    ]
    write_multi_sample(sources, path, sample_rate=8000)


def test_stream_processor_matches_simulate_stream_for_chunked_replay(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    _write_two_source_sample(wav_path)
    config = StreamingConfig(max_tracks=3, emit_interval_s=0.5)

    reference = simulate_stream_from_wav(wav_path, config)
    signal, sample_rate = read_wav_mono(wav_path)
    processor = StreamProcessor(sample_rate, config)
    for start in range(0, len(signal), 137):
        processor.push(signal[start : start + 137])
    chunked = processor.finish(final_time_s=len(signal) / sample_rate)

    assert [track.decoded.text for track in chunked.tracks] == [track.decoded.text for track in reference.tracks]
    assert [event.kind for event in chunked.events] == [event.kind for event in reference.events]
    assert chunked.frames_processed == reference.frames_processed
    assert chunked.tracker_frames_processed == reference.tracker_frames_processed


def test_stream_processor_push_returns_only_new_live_output(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    _write_two_source_sample(wav_path)
    signal, sample_rate = read_wav_mono(wav_path)
    processor = StreamProcessor(sample_rate, StreamingConfig(max_tracks=3, emit_interval_s=0.5))

    chunk_events = []
    chunk_updates = []
    block_length = sample_rate // 10
    for start in range(0, len(signal), block_length):
        chunk = processor.push(signal[start : start + block_length])
        assert chunk.time_s >= 0
        assert chunk.retained_frames == processor.retained_frames
        chunk_events.extend(chunk.events)
        chunk_updates.extend(chunk.updates)

    result = processor.finish(final_time_s=len(signal) / sample_rate)

    assert chunk_updates == result.updates
    assert [event.kind for event in chunk_events if event.kind != "SESSION_FINAL"]
    assert "CHANNEL_STARTED" in {event.kind for event in chunk_events}
    assert "TEXT_COMMITTED" in {event.kind for event in chunk_events}


def test_stream_processor_finish_is_idempotent_and_closes_input(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    _write_two_source_sample(wav_path)
    signal, sample_rate = read_wav_mono(wav_path)
    processor = StreamProcessor(sample_rate, StreamingConfig(max_tracks=3, emit_interval_s=0.5))
    processor.push(signal)

    first = processor.finish(final_time_s=len(signal) / sample_rate)
    second = processor.finish()

    assert first is second
    with pytest.raises(RuntimeError, match="cannot push samples after finish"):
        processor.push(signal[:100])
