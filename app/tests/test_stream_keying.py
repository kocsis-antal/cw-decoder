import numpy as np

from cw.generator import GeneratorConfig, build_events, render_wave
from cw.streaming import StreamingConfig, simulate_stream


def _live_keying_config() -> StreamingConfig:
    return StreamingConfig(
        min_peak_snr_db=10.0,
        min_keying_tone_runs=3,
        min_keying_chars=2,
        min_keying_known_chars=2,
        min_keying_active_duration_s=0.12,
        min_keying_duty_cycle=0.03,
        max_keying_duty_cycle=0.92,
        min_keying_unit_s=0.03,
        max_keying_score=120.0,
        reject_et_only_sessions=True,
        emit_interval_s=0.25,
    )


def test_keying_gate_keeps_continuous_carrier_pending() -> None:
    sample_rate = 8000
    duration_s = 2.0
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    signal = 0.7 * np.sin(2 * np.pi * 700.0 * t)

    result = simulate_stream(signal.astype(np.float32), sample_rate, _live_keying_config())

    assert result.events == []
    assert result.tracks == []


def test_keying_gate_allows_short_but_real_tu_session() -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.7)
    signal = render_wave(build_events("TU", config), config)
    signal = np.concatenate([signal, np.zeros(int(config.sample_rate * 0.9), dtype=np.float32)])

    result = simulate_stream(signal, config.sample_rate, _live_keying_config())

    texts = [session.decoded.text for track in result.tracks for session in track.sessions]
    assert "TU" in texts


def test_keying_gate_rejects_repeated_et_only_live_spam() -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.7)
    signal = render_wave(build_events("E T E T E", config), config)
    signal = np.concatenate([signal, np.zeros(int(config.sample_rate * 0.9), dtype=np.float32)])

    result = simulate_stream(signal, config.sample_rate, _live_keying_config())

    assert result.events == []
    assert result.tracks == []
