from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cw.generator import GeneratorConfig, build_events, render_wave
from cw.prob_analysis import analyze_raw_file


def _write_s16le(path: Path, signal: np.ndarray) -> None:
    clipped = np.clip(signal, -1.0, 0.9999695)
    values = (clipped * 32768.0).astype("<i2")
    path.write_bytes(values.tobytes())


def test_analyze_raw_file_reports_threshold_candidates(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=18.0, amplitude=0.6)
    signal = render_wave(build_events("CQ CQ DE TEST", config), config)
    raw_path = tmp_path / "cq.s16le"
    _write_s16le(raw_path, signal)

    report = analyze_raw_file(
        raw_path,
        sample_rate=8000,
        carriers=(700.0,),
        detect_carriers=0,
        threshold_ratios=(0.25, 0.35),
    )

    assert report.audio.samples == len(signal)
    assert report.carriers[0].carrier_hz == 700.0
    texts = [analysis.text for analysis in report.carriers[0].analyses]
    assert any("CQ" in text for text in texts)
    assert {analysis.threshold_ratio for analysis in report.carriers[0].analyses} == {0.25, 0.35}
    assert report.carriers[0].analyses[0].tone_durations.count > 0
    assert report.carriers[0].analyses[0].gap_durations.count > 0


def test_analyze_raw_cli_json(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=650.0, wpm=20.0, amplitude=0.6)
    signal = render_wave(build_events("TEST", config), config)
    raw_path = tmp_path / "test.s16le"
    _write_s16le(raw_path, signal)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cw.cli",
            "analyze-raw",
            str(raw_path),
            "--sample-rate",
            "8000",
            "--carrier",
            "650",
            "--threshold-ratios",
            "0.35",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    assert payload["carriers"][0]["carrier_hz"] == 650.0
    assert payload["carriers"][0]["analyses"][0]["threshold_ratio"] == 0.35
    assert "TEST" in payload["carriers"][0]["analyses"][0]["text"]


def test_accumulated_carrier_detection_can_use_low_threshold_for_weaker_peak() -> None:
    from cw.prob_analysis import _detect_carriers_from_spectrum

    freqs = np.arange(0, 3000, 10, dtype=np.float32)
    spectrum = np.ones((100, len(freqs)), dtype=np.float32)
    strong_index = int(np.argmin(np.abs(freqs - 700.0)))
    weak_index = int(np.argmin(np.abs(freqs - 1500.0)))

    # Loud station dominates the full rolling window.  A weaker station is still
    # recoverable when the operator-facing live profile uses a low accumulated
    # peak threshold, without promoting arbitrary one-frame temporal peaks.
    spectrum[:80, strong_index] = 1000.0
    spectrum[70:100, weak_index] = 600.0

    detected = _detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=200.0,
        max_tone_hz=2500.0,
        max_carriers=4,
        min_separation_hz=80.0,
        relative_threshold=0.05,
    )

    carriers = [round(candidate.carrier_hz) for candidate in detected]
    assert 700 in carriers
    assert 1500 in carriers
