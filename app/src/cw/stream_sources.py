from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol

import numpy as np
import soundfile as sf

from cw.decoder import _to_mono_float


@dataclass(frozen=True)
class AudioBlock:
    """One continuous block from an audio source.

    The processor currently needs only ``samples``.  The timing metadata is kept
    with the block so future real-time sources, replay logs, and UI/debug code
    can agree on where each push came from without re-deriving it elsewhere.
    """

    samples: np.ndarray
    sample_rate: int
    start_s: float
    duration_s: float
    index: int

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class AudioSource(Protocol):
    """Iterable source of mono float audio blocks."""

    sample_rate: int
    duration_s: float | None

    def __iter__(self) -> Iterator[AudioBlock]: ...


class ArrayAudioSource:
    """Replay an in-memory signal as timed blocks.

    This is useful for tests and compatibility with older code paths that still
    start from a NumPy array, while exercising the same block-source shape as a
    WAV file or future microphone input.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int, block_ms: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if block_ms <= 0:
            raise ValueError("block_ms must be positive")

        self.samples = _to_mono_float(np.asarray(samples))
        self.sample_rate = int(sample_rate)
        self.block_ms = float(block_ms)
        self.block_size = max(1, round(self.sample_rate * self.block_ms / 1000))
        self.duration_s = len(self.samples) / self.sample_rate

    def __iter__(self) -> Iterator[AudioBlock]:
        index = 0
        for start in range(0, len(self.samples), self.block_size):
            block = self.samples[start : start + self.block_size]
            yield AudioBlock(
                samples=block,
                sample_rate=self.sample_rate,
                start_s=start / self.sample_rate,
                duration_s=len(block) / self.sample_rate,
                index=index,
            )
            index += 1


class WavFileSource:
    """Stream a WAV/audio file from disk in small mono float blocks.

    Unlike ``read_wav_mono`` this does not need to load the complete file before
    the stream processor starts receiving samples.  The file is opened fresh for
    every iteration, so the source can be replayed more than once in tests.
    """

    def __init__(self, path: Path, block_ms: float) -> None:
        if block_ms <= 0:
            raise ValueError("block_ms must be positive")

        self.path = Path(path)
        self.block_ms = float(block_ms)
        with sf.SoundFile(self.path) as sound_file:
            self.sample_rate = int(sound_file.samplerate)
            self.frames = int(len(sound_file))
        if self.sample_rate <= 0:
            raise ValueError("WAV sample rate must be positive")
        self.block_size = max(1, round(self.sample_rate * self.block_ms / 1000))
        self.duration_s = self.frames / self.sample_rate

    def __iter__(self) -> Iterator[AudioBlock]:
        with sf.SoundFile(self.path) as sound_file:
            index = 0
            samples_read = 0
            while True:
                raw_block = sound_file.read(self.block_size, dtype="float32", always_2d=False)
                if len(raw_block) == 0:
                    break
                block = _to_mono_float(np.asarray(raw_block))
                yield AudioBlock(
                    samples=block,
                    sample_rate=self.sample_rate,
                    start_s=samples_read / self.sample_rate,
                    duration_s=len(block) / self.sample_rate,
                    index=index,
                )
                samples_read += len(block)
                index += 1


@dataclass(frozen=True)
class RawPcmFormat:
    name: str
    dtype: np.dtype
    sample_width_bytes: int
    scale: float | None = None
    unsigned_zero: float | None = None


_PCM_FORMATS: dict[str, RawPcmFormat] = {
    "s16le": RawPcmFormat("s16le", np.dtype("<i2"), 2, scale=32768.0),
    "s16be": RawPcmFormat("s16be", np.dtype(">i2"), 2, scale=32768.0),
    "s32le": RawPcmFormat("s32le", np.dtype("<i4"), 4, scale=2147483648.0),
    "s32be": RawPcmFormat("s32be", np.dtype(">i4"), 4, scale=2147483648.0),
    "f32le": RawPcmFormat("f32le", np.dtype("<f4"), 4),
    "f32be": RawPcmFormat("f32be", np.dtype(">f4"), 4),
    "u8": RawPcmFormat("u8", np.dtype("u1"), 1, scale=128.0, unsigned_zero=128.0),
}


def supported_pcm_formats() -> tuple[str, ...]:
    return tuple(_PCM_FORMATS)


def pcm_sample_width_bytes(sample_format: str) -> int:
    return _pcm_format(sample_format).sample_width_bytes


def decode_raw_pcm(raw: bytes, sample_format: str = "s16le", channels: int = 1) -> np.ndarray:
    """Convert interleaved raw PCM bytes to mono float32 samples.

    Supported formats intentionally match common ``ffmpeg``/``parec`` names.
    Integer samples are normalized to roughly ``[-1.0, 1.0]``; stereo or wider
    input is averaged to mono because the CW decoder consumes one channel.
    """

    if channels <= 0:
        raise ValueError("channels must be positive")
    fmt = _pcm_format(sample_format)
    frame_width_bytes = fmt.sample_width_bytes * channels
    if len(raw) % frame_width_bytes != 0:
        raise ValueError("raw PCM block contains a partial sample frame")
    if not raw:
        return np.array([], dtype=np.float32)

    values = np.frombuffer(raw, dtype=fmt.dtype)
    if fmt.unsigned_zero is not None:
        samples = (values.astype(np.float32) - fmt.unsigned_zero) / float(fmt.scale)
    elif fmt.scale is not None:
        samples = values.astype(np.float32) / float(fmt.scale)
    else:
        samples = values.astype(np.float32)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)


class RawPcmStreamSource:
    """Stream raw PCM bytes from a binary file-like object as audio blocks.

    This is the bridge to live input without depending on a particular audio
    backend.  A browser/WebSDR capture, virtual microphone, ``parec`` or
    ``ffmpeg`` can all write raw PCM to stdin, and the decoder sees the same
    ``AudioBlock`` shape as with WAV replay.
    """

    def __init__(
        self,
        stream: BinaryIO,
        sample_rate: int,
        sample_format: str = "s16le",
        channels: int = 1,
        block_ms: float = 10.0,
        duration_s: float | None = None,
        capture_raw_path: Path | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if block_ms <= 0:
            raise ValueError("block_ms must be positive")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive when set")

        self.stream = stream
        self.sample_rate = int(sample_rate)
        self.sample_format = _pcm_format(sample_format).name
        self.channels = int(channels)
        self.block_ms = float(block_ms)
        self.duration_s = duration_s
        self.capture_raw_path = Path(capture_raw_path) if capture_raw_path is not None else None
        self.block_size_frames = max(1, round(self.sample_rate * self.block_ms / 1000))
        self.sample_width_bytes = pcm_sample_width_bytes(self.sample_format)
        self.frame_width_bytes = self.sample_width_bytes * self.channels
        self.block_size_bytes = self.block_size_frames * self.frame_width_bytes
        self.max_frames = None if duration_s is None else int(round(duration_s * self.sample_rate))

    def __iter__(self) -> Iterator[AudioBlock]:
        pending = b""
        index = 0
        frames_read = 0
        capture_file = None
        try:
            if self.capture_raw_path is not None:
                self.capture_raw_path.parent.mkdir(parents=True, exist_ok=True)
                capture_file = self.capture_raw_path.open("wb")

            while self.max_frames is None or frames_read < self.max_frames:
                wanted = self.block_size_bytes
                if self.max_frames is not None:
                    remaining_frames = self.max_frames - frames_read
                    wanted = min(wanted, remaining_frames * self.frame_width_bytes)
                raw = self.stream.read(wanted)
                if not raw:
                    break
                if capture_file is not None:
                    capture_file.write(raw)

                pending += raw
                usable_bytes = (len(pending) // self.frame_width_bytes) * self.frame_width_bytes
                if usable_bytes <= 0:
                    continue

                block_raw = pending[:usable_bytes]
                pending = pending[usable_bytes:]
                samples = decode_raw_pcm(block_raw, self.sample_format, self.channels)
                if self.max_frames is not None:
                    samples = samples[: self.max_frames - frames_read]
                if len(samples) == 0:
                    break

                yield AudioBlock(
                    samples=samples,
                    sample_rate=self.sample_rate,
                    start_s=frames_read / self.sample_rate,
                    duration_s=len(samples) / self.sample_rate,
                    index=index,
                )
                frames_read += len(samples)
                index += 1
        finally:
            if capture_file is not None:
                capture_file.close()


def _pcm_format(sample_format: str) -> RawPcmFormat:
    try:
        return _PCM_FORMATS[sample_format]
    except KeyError as exc:
        supported = ", ".join(supported_pcm_formats())
        raise ValueError(f"unsupported raw PCM sample format: {sample_format!r}; supported: {supported}") from exc
