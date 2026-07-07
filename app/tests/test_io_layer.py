from __future__ import annotations

import io

import numpy as np
import pytest

from cw.io.array_source import ArrayAudioSource
from cw.io.models import AudioBlock, AudioSource
from cw.io.pcm import decode_raw_pcm
from cw.io.raw_stream_source import RawPcmStreamSource


def test_array_audio_source_emits_common_audio_blocks() -> None:
    source: AudioSource = ArrayAudioSource(np.arange(5, dtype=np.float32), sample_rate=10, block_ms=200)
    blocks = list(source)

    assert [type(block) for block in blocks] == [AudioBlock, AudioBlock, AudioBlock]
    assert [block.index for block in blocks] == [0, 1, 2]
    assert [block.start_s for block in blocks] == [0.0, 0.2, 0.4]
    assert [block.duration_s for block in blocks] == [0.2, 0.2, 0.1]
    assert [block.samples.tolist() for block in blocks] == [[0.0, 1.0], [2.0, 3.0], [4.0]]


def test_decode_raw_pcm_stereo_averages_to_mono_float32() -> None:
    raw = np.array([0, 32767, -32768, 0], dtype="<i2").tobytes()
    samples = decode_raw_pcm(raw, sample_format="s16le", channels=2)

    assert samples.dtype == np.float32
    assert samples.shape == (2,)
    assert 0.49 < samples[0] < 0.51
    assert -0.51 < samples[1] < -0.49


def test_decode_raw_pcm_rejects_partial_frame() -> None:
    with pytest.raises(ValueError, match="partial sample frame"):
        decode_raw_pcm(b"\x00", sample_format="s16le", channels=1)


class FragmentedBytesIO(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(1 if size < 0 else min(size, 1))


def test_raw_pcm_stream_source_reassembles_fragmented_reads() -> None:
    raw = np.array([0, 32767, -32768], dtype="<i2").tobytes()
    source = RawPcmStreamSource(FragmentedBytesIO(raw), sample_rate=3, sample_format="s16le", block_ms=1000)

    blocks = list(source)

    assert len(blocks) == 3
    assert [block.index for block in blocks] == [0, 1, 2]
    assert [len(block.samples) for block in blocks] == [1, 1, 1]
    assert blocks[0].start_s == 0.0
    assert blocks[1].start_s == pytest.approx(1 / 3)
    assert blocks[2].start_s == pytest.approx(2 / 3)


def test_raw_pcm_stream_source_duration_limits_output() -> None:
    raw = np.array([0, 1000, 2000, 3000], dtype="<i2").tobytes()
    source = RawPcmStreamSource(io.BytesIO(raw), sample_rate=4, sample_format="s16le", block_ms=1000, duration_s=0.5)

    blocks = list(source)

    assert len(blocks) == 1
    assert len(blocks[0].samples) == 2
    assert blocks[0].duration_s == 0.5


def test_capture_raw_writes_read_ahead_before_decoder_consumes(tmp_path) -> None:
    import time

    raw = np.arange(200, dtype="<i2").tobytes()
    capture_path = tmp_path / "captured.s16le"
    source = RawPcmStreamSource(
        io.BytesIO(raw),
        sample_rate=100,
        sample_format="s16le",
        block_ms=10,
        capture_raw_path=capture_path,
    )
    iterator = iter(source)

    # Creating the iterator starts the read-ahead capture thread.  It should be
    # able to save the whole finite input even if the decoder has consumed only
    # the first yielded block so far.
    first = next(iterator)
    deadline = time.monotonic() + 1.0
    while capture_path.stat().st_size < len(raw) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert len(first.samples) == 1
    assert capture_path.read_bytes() == raw
