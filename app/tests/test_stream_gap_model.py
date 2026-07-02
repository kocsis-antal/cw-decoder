from __future__ import annotations

from cw.decoder import DetectedRun
from cw.morse_table import MORSE_BY_CHAR
from cw.stream_decode import _decode_with_unit
from cw.stream_models import StreamingConfig


def _runs_for_words(words: list[str], *, unit_s: float, letter_gap_units: float, word_gap_units: float) -> list[DetectedRun]:
    runs: list[DetectedRun] = []
    cursor = 0.0

    def add_run(kind: str, units: float) -> None:
        nonlocal cursor
        duration = round(units * unit_s, 10)
        runs.append(DetectedRun(kind, round(cursor, 10), duration))
        cursor += duration

    for word_index, word in enumerate(words):
        if word_index > 0:
            add_run("gap", word_gap_units)
        for char_index, char in enumerate(word):
            if char_index > 0:
                add_run("gap", letter_gap_units)
            symbols = MORSE_BY_CHAR[char]
            for symbol_index, symbol in enumerate(symbols):
                if symbol_index > 0:
                    add_run("gap", 1.0)
                add_run("tone", 1.0 if symbol == "." else 3.0)
    return runs


def test_adaptive_gap_model_keeps_stretched_letter_gaps_as_letters_without_word_cluster() -> None:
    unit_s = 0.05
    runs = _runs_for_words(["CQ"], unit_s=unit_s, letter_gap_units=5.3, word_gap_units=9.5)

    decoded = _decode_with_unit(runs, carrier_hz=700.0, threshold=1.0, unit_s=unit_s, config=StreamingConfig())

    assert decoded.text == "CQ"
    assert [gap.symbol for gap in decoded.classified_runs if gap.kind == "gap" and gap.units > 2] == ["letter_gap"]


def test_adaptive_gap_model_keeps_distinct_real_word_gap() -> None:
    unit_s = 0.05
    runs = _runs_for_words(["CQ", "CQ"], unit_s=unit_s, letter_gap_units=5.3, word_gap_units=9.5)

    decoded = _decode_with_unit(runs, carrier_hz=700.0, threshold=1.0, unit_s=unit_s, config=StreamingConfig())

    assert decoded.text == "CQ CQ"
    assert [gap.symbol for gap in decoded.classified_runs if gap.kind == "gap" and gap.units > 2] == [
        "letter_gap",
        "word_gap",
        "letter_gap",
    ]


def test_fixed_gap_model_can_still_be_requested_for_compatibility() -> None:
    unit_s = 0.05
    runs = _runs_for_words(["CQ"], unit_s=unit_s, letter_gap_units=5.3, word_gap_units=9.5)
    config = StreamingConfig(adaptive_gap_thresholds=False)

    decoded = _decode_with_unit(runs, carrier_hz=700.0, threshold=1.0, unit_s=unit_s, config=config)

    assert decoded.text == "C Q"
