from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from cw.io.audio import to_mono_float
from cw.io.models import AudioBlock


class WavFileSource:
    """Stream a WAV/audio file from disk in small mono float blocks.

    The file is opened fresh for every iteration, so the source can be replayed
    more than once in tests or tools.
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
                block = to_mono_float(np.asarray(raw_block))
                yield AudioBlock(
                    samples=block,
                    sample_rate=self.sample_rate,
                    start_s=samples_read / self.sample_rate,
                    duration_s=len(block) / self.sample_rate,
                    index=index,
                )
                samples_read += len(block)
                index += 1
