from pathlib import Path

import pytest

from cw.decoder import DetectedRun, classify_runs, decode_wav
from cw.decoder import _estimate_unit_s
from cw.generator import GeneratorConfig, write_sample


def test_decode_generated_clean_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "cq.wav"
    write_sample("CQ CQ DE HA5ABC", wav_path, GeneratorConfig())

    result = decode_wav(wav_path)

    assert result.text == "CQ CQ DE HA5ABC"
    assert result.carrier_hz == pytest.approx(700.0, abs=1.0)
    assert result.unit_s == pytest.approx(0.06, abs=0.02)
    assert result.tokens == [
        "-.-.",
        "--.-",
        "/",
        "-.-.",
        "--.-",
        "/",
        "-..",
        ".",
        "/",
        "....",
        ".-",
        ".....",
        ".-",
        "-...",
        "-.-.",
    ]


def test_classify_runs() -> None:
    runs = [
        DetectedRun("tone", 0.0, 0.18),
        DetectedRun("gap", 0.18, 0.06),
        DetectedRun("tone", 0.24, 0.06),
        DetectedRun("gap", 0.3, 0.18),
        DetectedRun("gap", 0.48, 0.42),
    ]

    classified = classify_runs(runs, unit_s=0.06)

    assert [(run.kind, run.symbol, run.units) for run in classified] == [
        ("tone", "-", 3.0),
        ("gap", "element_gap", 1.0),
        ("tone", ".", 1.0),
        ("gap", "letter_gap", 3.0),
        ("gap", "word_gap", 7.0),
    ]


def test_estimate_unit_uses_dot_dash_fit_not_minimum() -> None:
    runs = [
        DetectedRun("tone", 0.0, 0.04),
        DetectedRun("tone", 0.1, 0.055),
        DetectedRun("tone", 0.2, 0.06),
        DetectedRun("tone", 0.3, 0.17),
        DetectedRun("tone", 0.5, 0.18),
    ]

    assert _estimate_unit_s(runs) == pytest.approx(0.06, abs=0.01)
