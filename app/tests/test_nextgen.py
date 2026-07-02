from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cw.generator import GeneratorConfig, build_events, render_wave
from cw.nextgen import decode_raw_file_nextgen


def _write_s16le(path: Path, signal: np.ndarray) -> None:
    clipped = np.clip(signal, -1.0, 0.9999695)
    values = (clipped * 32768.0).astype("<i2")
    path.write_bytes(values.tobytes())


def test_nextgen_decodes_raw_carrier_without_qso_bias(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.65)
    signal = render_wave(build_events("HELLO WORLD", config), config)
    raw_path = tmp_path / "plain.s16le"
    _write_s16le(raw_path, signal)

    report = decode_raw_file_nextgen(raw_path, carriers=(700.0,), detect_carriers=0)

    assert report.carriers[0].text == "HELLO WORLD"
    assert report.carriers[0].best is not None
    assert report.carriers[0].best.confidence > 0.5


def test_nextgen_cli_json_detects_carrier_and_decodes(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=900.0, wpm=18.0, amplitude=0.7)
    signal = render_wave(build_events("CQ TEST", config), config)
    raw_path = tmp_path / "cq.s16le"
    _write_s16le(raw_path, signal)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cw.cli",
            "decode-raw",
            str(raw_path),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    assert payload["sample_rate"] == 8000
    assert payload["detected_carriers"]
    texts = [carrier["text"] for carrier in payload["carriers"]]
    assert any("CQ" in text and "TEST" in text for text in texts)


def test_nextgen_reports_multiple_timed_sessions_on_one_carrier(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.65)
    first = render_wave(build_events("CQ TEST", config), config)
    second = render_wave(build_events("HELLO", config), config)
    silence = np.zeros(int(config.sample_rate * 1.8), dtype=np.float32)
    raw_path = tmp_path / "sessions.s16le"
    _write_s16le(raw_path, np.concatenate([first, silence, second]))

    report = decode_raw_file_nextgen(
        raw_path,
        carriers=(700.0,),
        detect_carriers=0,
        session_gap_s=1.0,
    )

    sessions = report.carriers[0].sessions
    assert len(sessions) == 2
    assert sessions[0].text == "CQ TEST"
    assert sessions[1].text == "HELLO"
    assert sessions[0].end_s < sessions[1].start_s


def test_soft_activity_hysteresis_keeps_ambiguous_tone_state() -> None:
    from cw.nextgen import _hysteresis_activity

    probabilities = np.asarray([0.05, 0.62, 0.48, 0.34, 0.29, 0.27, 0.60, 0.20], dtype=np.float32)

    active = _hysteresis_activity(probabilities, on_probability=0.56, off_probability=0.28)

    assert active.tolist() == [False, True, True, True, True, False, True, False]


def test_soft_bridge_repairs_short_non_silent_fade_gap() -> None:
    from cw.decoder import DetectedRun
    from cw.nextgen import _bridge_soft_fade_gaps
    from cw.stream_models import StreamingConfig

    runs = [
        DetectedRun("tone", 0.00, 0.06),
        DetectedRun("gap", 0.06, 0.04),
        DetectedRun("tone", 0.10, 0.08),
    ]
    frame_times = np.arange(0.0, 0.20, 0.01, dtype=np.float32)
    probabilities = np.full(len(frame_times), 0.9, dtype=np.float32)
    probabilities[(frame_times >= 0.06) & (frame_times < 0.10)] = 0.24

    bridged = _bridge_soft_fade_gaps(
        runs,
        probabilities,
        frame_times,
        0.0,
        StreamingConfig(
            soft_bridge_min_probability=0.18,
            soft_bridge_max_gap_ms=90.0,
            soft_bridge_gap_units=0.0,
        ),
    )

    assert len(bridged) == 1
    assert bridged[0].kind == "tone"
    assert bridged[0].duration_s == 0.18


def test_soft_bridge_does_not_merge_real_silent_element_gap() -> None:
    from cw.decoder import DetectedRun
    from cw.nextgen import _bridge_soft_fade_gaps
    from cw.stream_models import StreamingConfig

    runs = [
        DetectedRun("tone", 0.00, 0.06),
        DetectedRun("gap", 0.06, 0.04),
        DetectedRun("tone", 0.10, 0.08),
    ]
    frame_times = np.arange(0.0, 0.20, 0.01, dtype=np.float32)
    probabilities = np.full(len(frame_times), 0.9, dtype=np.float32)
    probabilities[(frame_times >= 0.06) & (frame_times < 0.10)] = 0.03

    bridged = _bridge_soft_fade_gaps(
        runs,
        probabilities,
        frame_times,
        0.0,
        StreamingConfig(
            soft_bridge_min_probability=0.18,
            soft_bridge_max_gap_ms=90.0,
            soft_bridge_gap_units=0.0,
        ),
    )

    assert bridged == runs
