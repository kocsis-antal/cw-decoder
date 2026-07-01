from pathlib import Path

import pytest

from cw.evaluation import evaluate_wav
from cw.generator import GeneratorConfig, write_sample


def test_evaluate_generated_clean_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "cq.wav"
    labels_path = write_sample("CQ CQ DE HA5ABC", wav_path, GeneratorConfig())

    result = evaluate_wav(wav_path, labels_path)

    assert result.expected_text == "CQ CQ DE HA5ABC"
    assert result.decoded_text == "CQ CQ DE HA5ABC"
    assert result.text_ok is True
    assert result.token_accuracy == 1.0
    assert result.carrier_error_hz == pytest.approx(0.0, abs=1.0)
    assert result.unit_error_ms == pytest.approx(0.0, abs=20.0)
    assert result.timing.symbol_accuracy == 1.0
    assert result.timing.compared_count > 0


def test_evaluate_normalizes_whitespace_in_expected_text(tmp_path: Path) -> None:
    wav_path = tmp_path / "cq.wav"
    labels_path = write_sample("CQ CQ DE  YU7NKA", wav_path, GeneratorConfig())

    result = evaluate_wav(wav_path, labels_path)

    assert result.expected_text == "CQ CQ DE YU7NKA"
    assert result.decoded_text == "CQ CQ DE YU7NKA"
    assert result.text_ok is True
