import numpy as np

from cw.stream_models import SpectrumFrame, StreamingConfig
from cw.stream_tracker import CarrierTracker, SpectralPeak, detect_frame_peaks


def test_detect_frame_peaks_selects_separated_strong_tones() -> None:
    freqs = np.arange(0, 1200, 20, dtype=np.float32)
    spectrum = np.zeros_like(freqs)
    spectrum[np.argmin(np.abs(freqs - 700))] = 100.0
    spectrum[np.argmin(np.abs(freqs - 1000))] = 60.0
    spectrum[np.argmin(np.abs(freqs - 720))] = 80.0  # too close to 700, should be suppressed
    frame = SpectrumFrame(start_s=0.0, spectrum=spectrum, freqs=freqs)

    peaks = detect_frame_peaks(
        frame,
        StreamingConfig(min_tone_hz=200, max_tone_hz=1100, min_separation_hz=80, max_tracks=5),
    )

    assert [round(peak.frequency_hz) for peak in peaks] == [700, 1000]


def test_carrier_tracker_smooths_frequency_and_keeps_one_track_for_drift() -> None:
    config = StreamingConfig(carrier_smoothing=0.5, min_track_hits=1, max_track_gap_s=1.0)
    tracker = CarrierTracker(config)

    tracker.update_peaks([SpectralPeak(0.0, 700.0, 100.0, 1.0)], time_s=0.0)
    tracker.update_peaks([SpectralPeak(0.1, 710.0, 100.0, 1.0)], time_s=0.1)

    active = tracker.active_tracks(0.1)
    assert len(active) == 1
    assert active[0].hits == 2
    assert round(active[0].frequency_hz, 1) == 705.0


def test_carrier_tracker_returns_multiple_carriers_above_long_term_threshold() -> None:
    config = StreamingConfig(min_track_hits=1, max_tracks=5, track_relative_threshold=0.10)
    tracker = CarrierTracker(config)

    tracker.update_peaks(
        [
            SpectralPeak(0.0, 700.0, 100.0, 1.0),
            SpectralPeak(0.0, 1000.0, 30.0, 0.3),
            SpectralPeak(0.0, 1300.0, 5.0, 0.05),
        ],
        time_s=0.0,
    )

    carriers = tracker.candidate_carriers(0.0)
    assert [round(carrier[0]) for carrier in carriers] == [700, 1000]


def test_carrier_tracker_marks_missing_track_dormant() -> None:
    config = StreamingConfig(min_track_hits=1, max_track_gap_s=0.5)
    tracker = CarrierTracker(config)
    tracker.update_peaks([SpectralPeak(0.0, 700.0, 100.0, 1.0)], time_s=0.0)

    assert tracker.active_tracks(0.2)
    tracker.update_peaks([], time_s=1.0)

    assert tracker.tracks[0].state == "dormant"
    assert tracker.active_tracks(1.0) == []


def test_detect_frame_peaks_uses_peak_specific_separation() -> None:
    freqs = np.arange(0, 1200, 20, dtype=np.float32)
    spectrum = np.zeros_like(freqs)
    spectrum[np.argmin(np.abs(freqs - 700))] = 100.0
    spectrum[np.argmin(np.abs(freqs - 740))] = 80.0
    frame = SpectrumFrame(start_s=0.0, spectrum=spectrum, freqs=freqs)

    old_style = detect_frame_peaks(
        frame,
        StreamingConfig(min_tone_hz=200, max_tone_hz=1100, min_separation_hz=80, max_tracks=5),
    )
    tuned = detect_frame_peaks(
        frame,
        StreamingConfig(
            min_tone_hz=200,
            max_tone_hz=1100,
            min_separation_hz=80,
            peak_min_separation_hz=30,
            max_tracks=5,
        ),
    )

    assert [round(peak.frequency_hz) for peak in old_style] == [700]
    assert [round(peak.frequency_hz) for peak in tuned] == [700, 740]


def test_carrier_tracker_track_match_can_be_narrower_than_peak_spacing() -> None:
    config = StreamingConfig(
        min_track_hits=1,
        min_separation_hz=80,
        track_match_hz=20,
        carrier_smoothing=0.5,
    )
    tracker = CarrierTracker(config)

    tracker.update_peaks([SpectralPeak(0.0, 700.0, 100.0, 1.0)], time_s=0.0)
    tracker.update_peaks([SpectralPeak(0.1, 740.0, 90.0, 0.9)], time_s=0.1)

    active = tracker.active_tracks(0.1)
    assert len(active) == 2
    assert [round(track.frequency_hz) for track in active] == [700, 740]


def test_carrier_tracker_candidate_merge_is_separate_from_track_match() -> None:
    split_config = StreamingConfig(
        min_track_hits=1,
        min_separation_hz=80,
        track_match_hz=20,
        channel_merge_hz=50,
        max_tracks=5,
    )
    split_tracker = CarrierTracker(split_config)
    split_tracker.update_peaks(
        [
            SpectralPeak(0.0, 700.0, 100.0, 1.0),
            SpectralPeak(0.0, 780.0, 90.0, 0.9),
        ],
        time_s=0.0,
    )
    assert [round(carrier[0]) for carrier in split_tracker.candidate_carriers(0.0)] == [700, 780]

    merge_config = StreamingConfig(
        min_track_hits=1,
        min_separation_hz=80,
        track_match_hz=20,
        channel_merge_hz=120,
        max_tracks=5,
    )
    merge_tracker = CarrierTracker(merge_config)
    merge_tracker.update_peaks(
        [
            SpectralPeak(0.0, 700.0, 100.0, 1.0),
            SpectralPeak(0.0, 780.0, 90.0, 0.9),
        ],
        time_s=0.0,
    )
    assert [round(carrier[0]) for carrier in merge_tracker.candidate_carriers(0.0)] == [700]
