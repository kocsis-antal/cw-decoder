from pathlib import Path

import numpy as np
import pytest

from cw.decoder import read_wav_mono
from cw.multi_generator import parse_source_spec, write_multi_sample
from cw.streaming import ArrayAudioSource, StreamingConfig, WavFileSource, process_audio_source, simulate_stream_from_wav


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


def test_wav_file_source_streams_blocks_with_timing_metadata(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    _write_two_source_sample(wav_path)
    signal, sample_rate = read_wav_mono(wav_path)

    source = WavFileSource(wav_path, block_ms=17)
    blocks = list(source)

    assert source.sample_rate == sample_rate
    assert source.duration_s == pytest.approx(len(signal) / sample_rate)
    assert sum(len(block.samples) for block in blocks) == len(signal)
    assert blocks[0].index == 0
    assert blocks[0].start_s == pytest.approx(0.0)
    assert blocks[-1].end_s == pytest.approx(source.duration_s)
    assert all(block.sample_rate == sample_rate for block in blocks)
    assert all(block.start_s < block.end_s for block in blocks)
    assert all(blocks[index].end_s == pytest.approx(blocks[index + 1].start_s) for index in range(len(blocks) - 1))


def test_array_audio_source_uses_same_block_shape_as_wav_source() -> None:
    samples = np.arange(25, dtype=np.float32)
    source = ArrayAudioSource(samples, sample_rate=1000, block_ms=10)
    blocks = list(source)

    assert [len(block.samples) for block in blocks] == [10, 10, 5]
    assert [block.index for block in blocks] == [0, 1, 2]
    assert [block.start_s for block in blocks] == pytest.approx([0.0, 0.01, 0.02])
    assert blocks[-1].end_s == pytest.approx(0.025)


def test_process_audio_source_matches_wav_simulation(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    _write_two_source_sample(wav_path)
    config = StreamingConfig(max_tracks=3, emit_interval_s=0.5)

    reference = simulate_stream_from_wav(wav_path, config)
    source = WavFileSource(wav_path, config.input_block_ms)
    chunk_events = []
    chunk_updates = []
    replay = process_audio_source(
        source,
        config,
        on_chunk=lambda chunk: (chunk_events.extend(chunk.events), chunk_updates.extend(chunk.updates)),
    )

    assert [track.decoded.text for track in replay.tracks] == [track.decoded.text for track in reference.tracks]
    assert chunk_updates == replay.updates
    assert replay.frames_processed == reference.frames_processed
    assert replay.tracker_frames_processed == reference.tracker_frames_processed
    assert chunk_events == replay.events[: len(chunk_events)]
    assert any(event.kind == "TEXT_COMMITTED" for event in chunk_events)

import io

from cw.streaming import RawPcmStreamSource, decode_raw_pcm, supported_pcm_formats


def _float_to_s16le(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def test_decode_raw_pcm_supports_s16le_and_stereo_downmix() -> None:
    stereo = np.array(
        [
            [32767, 32767],
            [32767, -32768],
            [0, 0],
        ],
        dtype="<i2",
    )

    mono = decode_raw_pcm(stereo.tobytes(), "s16le", channels=2)

    assert "s16le" in supported_pcm_formats()
    assert mono.dtype == np.float32
    assert mono == pytest.approx([32767 / 32768, -1 / 65536, 0.0], abs=1e-5)


def test_raw_pcm_stream_source_reads_stdin_like_binary_blocks() -> None:
    samples = np.linspace(-0.5, 0.5, 35, dtype=np.float32)
    source = RawPcmStreamSource(
        io.BytesIO(_float_to_s16le(samples)),
        sample_rate=1000,
        sample_format="s16le",
        channels=1,
        block_ms=10,
        duration_s=0.025,
    )

    blocks = list(source)

    assert [len(block.samples) for block in blocks] == [10, 10, 5]
    assert [block.index for block in blocks] == [0, 1, 2]
    assert [block.start_s for block in blocks] == pytest.approx([0.0, 0.01, 0.02])
    assert blocks[-1].end_s == pytest.approx(0.025)


def test_decode_raw_pcm_rejects_partial_frames() -> None:
    with pytest.raises(ValueError, match="partial sample frame"):
        decode_raw_pcm(b"\x00", "s16le", channels=1)


def test_raw_pcm_stream_source_can_capture_exact_bytes(tmp_path: Path) -> None:
    samples = np.linspace(-0.25, 0.25, 30, dtype=np.float32)
    raw = _float_to_s16le(samples)
    capture_path = tmp_path / "capture" / "sample.s16le"
    source = RawPcmStreamSource(
        io.BytesIO(raw),
        sample_rate=1000,
        sample_format="s16le",
        channels=1,
        block_ms=10,
        capture_raw_path=capture_path,
    )

    blocks = list(source)

    assert sum(len(block.samples) for block in blocks) == len(samples)
    assert capture_path.read_bytes() == raw
