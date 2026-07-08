from __future__ import annotations

import numpy as np

from cw.app.config import ProcessingConfig
from cw.io.array_source import ArrayAudioSource
from cw.receiving.models import ChannelState
from cw.receiving.processor import Receiver
from cw.signal.segmenters import SignalSegmenterBank


def _tone_source(sample_rate: int = 8000, duration_s: float = 1.0) -> ArrayAudioSource:
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    return ArrayAudioSource(samples, sample_rate, block_ms=1000.0)


def test_receiving_outputs_channel_state_snapshots_not_events() -> None:
    source = _tone_source()
    receiver = Receiver(source.sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1))

    chunk = receiver.push(next(iter(source)))

    assert chunk.channels
    channel = chunk.channels[0]
    assert not hasattr(channel, "event")
    assert not hasattr(chunk, "events")
    assert channel.state == ChannelState.ACTIVE
    assert channel.has_audio
    assert channel.channel_id > 0


def test_candidate_is_a_pending_channel_state_with_stable_id() -> None:
    source = _tone_source(duration_s=2.0)
    receiver = Receiver(source.sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=2, emit_interval_s=0.1))
    iterator = iter(source)

    candidate_chunk = receiver.push(next(iterator))
    active_chunk = receiver.push(next(iterator))

    candidate = candidate_chunk.channels[0]
    active = active_chunk.channels[0]
    assert candidate.state == ChannelState.CANDIDATE
    assert not candidate.has_audio
    assert active.state == ChannelState.ACTIVE
    assert active.channel_id == candidate.channel_id


def test_receiving_finish_closes_channels_as_state_snapshots() -> None:
    source = _tone_source()
    receiver = Receiver(source.sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1))
    receiver.push(next(iter(source)))

    final_chunk = receiver.finish()

    dormant_channels = [channel for channel in final_chunk.channels if channel.state == ChannelState.DORMANT]
    assert dormant_channels
    assert dormant_channels[0].channel_id > 0


def test_candidate_finish_is_dropped_state_snapshot() -> None:
    source = _tone_source()
    receiver = Receiver(source.sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=2, emit_interval_s=0.1))
    receiver.push(next(iter(source)))

    final_chunk = receiver.finish()

    dropped_channels = [channel for channel in final_chunk.channels if channel.state == ChannelState.DROPPED]
    assert dropped_channels


def test_signal_layer_ignores_candidate_and_closed_channels() -> None:
    source = _tone_source()
    config = ProcessingConfig(max_tracks=1, min_track_hits=2, emit_interval_s=0.1)
    receiver = Receiver(source.sample_rate, config)
    candidate_chunk = receiver.push(next(iter(source)))

    tracks = tuple(track for channel in candidate_chunk.channels for track in SignalSegmenterBank.default(config).segment_channel(channel))

    assert candidate_chunk.channels[0].state == ChannelState.CANDIDATE
    assert tracks == ()

    final_chunk = receiver.finish()
    final_tracks = tuple(track for channel in final_chunk.channels for track in SignalSegmenterBank.default(config).segment_channel(channel))
    assert final_tracks == ()


def test_finish_flushes_active_channel_audio_before_closing() -> None:
    source = _tone_source(duration_s=1.0)
    receiver = Receiver(source.sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1))
    receiver.push(next(iter(source)))

    final_chunk = receiver.finish()

    dormant_channels = [channel for channel in final_chunk.channels if channel.state == ChannelState.DORMANT]
    assert dormant_channels
    assert dormant_channels[0].has_audio
    assert len(dormant_channels[0].audio_window) > 0


def test_spectral_carrier_detection_rejects_noise_only_relative_peaks() -> None:
    import numpy as np
    from cw.receiving.spectrum import power_spectrum_frames, detect_carriers_from_spectrum

    rng = np.random.default_rng(1234)
    noise = rng.normal(0.0, 0.01, 8000).astype(np.float32)
    spectrum, freqs = power_spectrum_frames(noise, 8000, frame_ms=30.0, hop_ms=5.0)

    loose = detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=200.0,
        max_tone_hz=3000.0,
        max_carriers=5,
        peak_separation_hz=80.0,
        relative_threshold=0.05,
        min_snr_db=0.0,
    )
    gated = detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=200.0,
        max_tone_hz=3000.0,
        max_carriers=5,
        peak_separation_hz=80.0,
        relative_threshold=0.05,
        min_snr_db=6.0,
    )

    assert loose
    assert gated == ()
