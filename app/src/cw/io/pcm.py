from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    return pcm_format(sample_format).sample_width_bytes


def decode_raw_pcm(raw: bytes, sample_format: str = "s16le", channels: int = 1) -> np.ndarray:
    """Convert interleaved raw PCM bytes to mono float32 samples.

    Supported formats intentionally match common ``ffmpeg``/``parec`` names.
    Integer samples are normalized to roughly ``[-1.0, 1.0]``; stereo or wider
    input is averaged to mono because upper layers consume one audio stream.
    """

    if channels <= 0:
        raise ValueError("channels must be positive")
    fmt = pcm_format(sample_format)
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


def pcm_format(sample_format: str) -> RawPcmFormat:
    try:
        return _PCM_FORMATS[sample_format]
    except KeyError as exc:
        supported = ", ".join(supported_pcm_formats())
        raise ValueError(f"unsupported raw PCM sample format: {sample_format!r}; supported: {supported}") from exc
