from __future__ import annotations

"""Current live streaming API.

The project now has one live engine: :class:`cw.nextgen_stream.NextgenStreamProcessor`.
This module intentionally keeps only a small public facade around that engine
and the audio-source helpers.  The old STFT/run based live processor was removed
from the public path so tests and tools exercise the same code that is used for
real ``stream-stdin`` decoding.
"""

from pathlib import Path

import numpy as np

from cw.decoder import DecodeResult
from cw.nextgen_stream import NextgenStreamProcessor
from cw.quality import QualityScore
from cw.stream_models import (
    StreamChunkResult,
    StreamEvent,
    StreamSessionResult,
    StreamSimulationResult,
    StreamTrackResult,
    StreamUpdate,
    StreamingConfig,
)
from cw.stream_sources import (
    ArrayAudioSource,
    AudioBlock,
    AudioSource,
    RawPcmStreamSource,
    WavFileSource,
    decode_raw_pcm,
    supported_pcm_formats,
)

# Backwards compatible name, but it is no longer the legacy STFT processor.
StreamProcessor = NextgenStreamProcessor

__all__ = [
    "ArrayAudioSource",
    "AudioBlock",
    "AudioSource",
    "RawPcmStreamSource",
    "NextgenStreamProcessor",
    "StreamProcessor",
    "StreamChunkResult",
    "StreamEvent",
    "StreamingConfig",
    "StreamSessionResult",
    "StreamSimulationResult",
    "StreamTrackResult",
    "StreamUpdate",
    "WavFileSource",
    "decode_raw_pcm",
    "supported_pcm_formats",
    "process_audio_source",
    "simulate_stream",
    "simulate_stream_from_wav",
]


def simulate_stream_from_wav(path: Path, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    return process_audio_source(WavFileSource(path, config.input_block_ms), config)


def simulate_stream(signal: np.ndarray, sample_rate: int, config: StreamingConfig | None = None) -> StreamSimulationResult:
    config = config or StreamingConfig()
    return process_audio_source(ArrayAudioSource(signal, sample_rate, config.input_block_ms), config)


def process_audio_source(source: AudioSource, config: StreamingConfig | None = None, *, on_chunk=None) -> StreamSimulationResult:
    """Decode a complete block-based source with the current nextgen live engine.

    This is mainly a replay/testing helper.  It feeds the same processor used by
    ``stream-stdin`` and reconstructs a compact compatibility summary from the
    emitted lifecycle events.
    """

    config = config or StreamingConfig()
    processor = NextgenStreamProcessor(source.sample_rate, config)
    events: list[StreamEvent] = []
    for block in source:
        if block.sample_rate != source.sample_rate:
            raise ValueError("audio block sample_rate changed during stream")
        chunk = processor.push(block.samples)
        if on_chunk is not None:
            on_chunk(chunk)
        events.extend(chunk.events)
    final_result = processor.finish(final_time_s=source.duration_s)
    final_chunk = StreamChunkResult(
        time_s=final_result.time_s,
        events=final_result.events[len(events):],
        frames_processed=final_result.frames_processed,
        tracker_frames_processed=final_result.tracker_frames_processed,
        retained_frames=final_result.retained_frames,
        pruned_frames=final_result.pruned_frames,
        active_pruned_frames=final_result.active_pruned_frames,
        finalized_pruned_frames=final_result.finalized_pruned_frames,
    )
    if on_chunk is not None:
        on_chunk(final_chunk)
    events.extend(final_chunk.events)
    updates = _updates_from_events(events)
    tracks = _tracks_from_events(events)
    return StreamSimulationResult(
        duration_s=final_result.time_s,
        updates=updates,
        tracks=tracks,
        events=events,
        frames_processed=final_result.frames_processed,
        tracker_frames_processed=final_result.tracker_frames_processed,
        retained_frames=final_result.retained_frames,
        pruned_frames=final_result.pruned_frames,
        active_pruned_frames=final_result.active_pruned_frames,
        finalized_pruned_frames=final_result.finalized_pruned_frames,
    )


def _updates_from_events(events: list[StreamEvent]) -> list[StreamUpdate]:
    updates: list[StreamUpdate] = []
    for event in events:
        if event.kind != "TEXT_COMMITTED" or event.session_id is None:
            continue
        updates.append(
            StreamUpdate(
                time_s=event.time_s,
                track_id=event.channel_id,
                session_id=event.session_id,
                carrier_hz=event.carrier_hz,
                score=0.0 if event.score is None else event.score,
                text=event.text,
            )
        )
    return updates


def _tracks_from_events(events: list[StreamEvent]) -> list[StreamTrackResult]:
    by_channel: dict[int, list[StreamEvent]] = {}
    carrier_by_channel: dict[int, float] = {}
    for event in events:
        carrier_by_channel[event.channel_id] = event.carrier_hz
        if event.kind == "SESSION_FINAL" and event.session_id is not None:
            by_channel.setdefault(event.channel_id, []).append(event)

    tracks: list[StreamTrackResult] = []
    for channel_id in sorted(by_channel):
        final_events = by_channel[channel_id]
        sessions = [_session_result_from_event(event) for event in final_events]
        if not sessions:
            continue
        carrier_hz = carrier_by_channel.get(channel_id, final_events[-1].carrier_hz)
        combined_text = " | ".join(session.decoded.text for session in sessions if session.decoded.text)
        decoded = _decode_result_for_text(combined_text, carrier_hz)
        quality = _quality_for_score(final_events[-1].score)
        tracks.append(
            StreamTrackResult(
                track_id=channel_id,
                carrier_hz=carrier_hz,
                first_seen_s=min(session.first_seen_s for session in sessions),
                last_seen_s=max(session.last_seen_s for session in sessions),
                hits=len(sessions),
                quality=quality,
                decoded=decoded,
                sessions=sessions,
            )
        )
    return tracks


def _session_result_from_event(event: StreamEvent) -> StreamSessionResult:
    decoded = _decode_result_for_text(event.text, event.carrier_hz)
    return StreamSessionResult(
        session_id=int(event.session_id or 0),
        first_seen_s=event.time_s,
        last_seen_s=event.time_s,
        hits=1,
        final_time_s=event.time_s,
        final_reason=event.reason,
        quality=_quality_for_score(event.score),
        decoded=decoded,
    )


def _decode_result_for_text(text: str, carrier_hz: float) -> DecodeResult:
    return DecodeResult(
        text=text,
        tokens=[],
        runs=[],
        classified_runs=[],
        carrier_hz=carrier_hz,
        unit_s=0.0,
        threshold=0.0,
    )


def _quality_for_score(score: float | None) -> QualityScore:
    return QualityScore(
        score=0.0 if score is None else float(score),
        unknown_count=0,
        token_count=0,
        dot_count=0,
        dash_count=0,
        tone_ratio_error=0.0,
        gap_min_error=0.0,
        unit_cv=0.0,
    )
