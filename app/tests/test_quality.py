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


def _minimal_decode(text: str, tokens: list[str]) -> DecodeResult:
    return DecodeResult(
        text=text,
        tokens=tokens,
        runs=[],
        classified_runs=[
            ClassifiedRun("tone", 0.0, 0.06, ".", 1.0),
            ClassifiedRun("gap", 0.06, 0.18, "letter_gap", 3.0),
            ClassifiedRun("tone", 0.24, 0.18, "-", 3.0),
        ],
        carrier_hz=700.0,
        unit_s=0.06,
        threshold=1.0,
    )


def test_score_decode_result_penalizes_unknown_density_not_raw_count() -> None:
    long_with_one_unknown = _minimal_decode(
        "CQ DE DJ?ZM DJ6ZM PSE K",
        ["-.-.", "--.-", "/", "-..", ".", "/", "-..", ".---", "-.-..", "--..", "--", "/", "-..", ".---", "-....", "--..", "--", "/", ".--.", "...", ".", "-.-"],
    )
    mostly_unknown = _minimal_decode("? ?", [".......", "/", "......."])

    assert score_decode_result(long_with_one_unknown).score < 25.0
    assert score_decode_result(mostly_unknown).score >= 200.0
