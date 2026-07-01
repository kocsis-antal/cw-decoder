from cw.decoder import ClassifiedRun, DecodeResult
from cw.quality import score_decode_result


def test_score_decode_result_accepts_longer_gaps() -> None:
    result = DecodeResult(
        text="A",
        tokens=[".-"],
        runs=[],
        classified_runs=[
            ClassifiedRun("tone", 0.0, 0.06, ".", 1.0),
            ClassifiedRun("gap", 0.06, 0.12, "element_gap", 2.0),
            ClassifiedRun("tone", 0.18, 0.18, "-", 3.0),
            ClassifiedRun("gap", 0.36, 0.60, "word_gap", 10.0),
        ],
        carrier_hz=700.0,
        unit_s=0.06,
        threshold=1.0,
    )

    quality = score_decode_result(result)

    assert quality.gap_min_error == 0.0
    assert quality.tone_ratio_error == 0.0


def test_score_decode_result_penalizes_too_short_letter_gap() -> None:
    result = DecodeResult(
        text="EE",
        tokens=[".", "."],
        runs=[],
        classified_runs=[
            ClassifiedRun("tone", 0.0, 0.06, ".", 1.0),
            ClassifiedRun("gap", 0.06, 0.06, "letter_gap", 1.0),
            ClassifiedRun("tone", 0.12, 0.06, ".", 1.0),
        ],
        carrier_hz=700.0,
        unit_s=0.06,
        threshold=1.0,
    )

    quality = score_decode_result(result)

    assert quality.gap_min_error > 0.0
