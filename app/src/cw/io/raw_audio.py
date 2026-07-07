from __future__ import annotations

from pathlib import Path

import numpy as np

from cw.io.pcm import decode_raw_pcm, pcm_sample_width_bytes

def read_raw_audio_slice(
    path: Path,
    *,
    sample_rate: int,
    sample_format: str = "s16le",
    channels: int = 1,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if start_s < 0:
        raise ValueError("start_s must not be negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive when set")

    frame_width = pcm_sample_width_bytes(sample_format) * channels
    start_frame = int(round(start_s * sample_rate))
    frames_to_read = None if duration_s is None else int(round(duration_s * sample_rate))

    with Path(path).open("rb") as raw_file:
        raw_file.seek(start_frame * frame_width)
        if frames_to_read is None:
            raw = raw_file.read()
        else:
            raw = raw_file.read(frames_to_read * frame_width)

    usable_bytes = (len(raw) // frame_width) * frame_width
    return decode_raw_pcm(raw[:usable_bytes], sample_format=sample_format, channels=channels)
