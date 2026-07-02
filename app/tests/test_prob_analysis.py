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
