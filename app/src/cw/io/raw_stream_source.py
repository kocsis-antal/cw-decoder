from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import BinaryIO, Iterator

from cw.io.models import AudioBlock
from cw.io.pcm import decode_raw_pcm, pcm_format, pcm_sample_width_bytes


_READ_AHEAD_EOF = object()


class RawPcmStreamSource:
    """Stream raw PCM bytes from a binary file-like object as audio blocks.

    This is the bridge to streaming input without depending on a particular audio
    backend. A browser/WebSDR capture, virtual microphone, ``parec`` or
    ``ffmpeg`` can all write raw PCM to stdin, and upper layers see the same
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
        self.sample_format = pcm_format(sample_format).name
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
        yield from self._blocks_from_raw_chunks(self._raw_chunks())

    def _raw_chunks(self) -> Iterator[bytes]:
        if self.capture_raw_path is not None and self.max_frames is None:
            yield from self._raw_chunks_with_capture_read_ahead()
            return
        yield from self._raw_chunks_sync()

    def _raw_chunks_sync(self) -> Iterator[bytes]:
        frames_read = 0
        capture_file = None
        try:
            if self.capture_raw_path is not None:
                self.capture_raw_path.parent.mkdir(parents=True, exist_ok=True)
                capture_file = self.capture_raw_path.open("wb", buffering=0)

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
                frames_read += len(raw) // self.frame_width_bytes
                yield raw
        finally:
            if capture_file is not None:
                capture_file.close()

    def _raw_chunks_with_capture_read_ahead(self) -> Iterator[bytes]:
        """Read and capture stdin in a background thread.

        The decoder can be heavier than realtime, especially while it repeatedly
        re-decodes the current channel window.  If raw capture happens only when
        the decoder asks for the next block, the upstream ffmpeg pipe is
        back-pressured and the saved file reflects decoder throughput rather than
        wall-clock audio.  This read-ahead path writes captured bytes as soon as
        they arrive, then lets decoding consume them at its own pace.
        """

        if self.capture_raw_path is None:
            raise RuntimeError("capture path is required for read-ahead capture")

        queue: Queue[bytes | BaseException | object] = Queue()
        stop = Event()
        self.capture_raw_path.parent.mkdir(parents=True, exist_ok=True)

        def _reader() -> None:
            capture_file = None
            try:
                capture_file = self.capture_raw_path.open("wb", buffering=0)
                while not stop.is_set():
                    raw = self.stream.read(self.block_size_bytes)
                    if not raw:
                        break
                    capture_file.write(raw)
                    queue.put(raw)
            except BaseException as exc:  # pragma: no cover - defensive relay from the reader thread
                queue.put(exc)
            finally:
                if capture_file is not None:
                    capture_file.close()
                queue.put(_READ_AHEAD_EOF)

        thread = Thread(target=_reader, name="cw-raw-capture-reader", daemon=False)
        thread.start()
        try:
            while True:
                item = queue.get()
                if item is _READ_AHEAD_EOF:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            # On Ctrl+C the consuming side can stop while the reader is blocked
            # in stream.read().  Closing the input side is the most reliable way
            # to unblock a pipe-backed stdin and avoid daemon-thread fatal errors
            # during interpreter shutdown.
            try:
                self.stream.close()
            except Exception:  # pragma: no cover - best-effort shutdown path
                pass
            thread.join(timeout=1.0)

    def _blocks_from_raw_chunks(self, raw_chunks: Iterator[bytes]) -> Iterator[AudioBlock]:
        pending = b""
        index = 0
        frames_read = 0
        for raw in raw_chunks:
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
