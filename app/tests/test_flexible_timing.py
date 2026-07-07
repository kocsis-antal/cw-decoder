from __future__ import annotations

from cw.decoder.config import DecoderConfig
from cw.decoder.timing import HardRun, HardRunKind, decode_segments_with_unit
from cw.app.text_display import normalize_cw_display_text


def _mark(ms: float, start: float) -> HardRun:
    return HardRun(HardRunKind.TONE, start, ms / 1000.0)


def _gap(ms: float, start: float) -> HardRun:
    return HardRun(HardRunKind.GAP, start, ms / 1000.0)


def test_flexible_timing_keeps_hand_sent_ti_ta_boundaries() -> None:
    # C Q with imperfect hand timing: dahs around 170 ms, dits around 55 ms,
    # and the C->Q letter gap squeezed to ~115 ms.  A rigid decoder used to
    # merge this into one invalid token in similar live captures.
    runs = [
        _mark(170, 0.000), _gap(50, 0.170), _mark(55, 0.220), _gap(55, 0.275), _mark(165, 0.330), _gap(55, 0.495), _mark(55, 0.550),
        _gap(115, 0.605),
        _mark(170, 0.720), _gap(50, 0.890), _mark(170, 0.940), _gap(50, 1.110), _mark(55, 1.160), _gap(55, 1.215), _mark(165, 1.270),
    ]
    config = DecoderConfig(flexible_timing=True)
    texts = {decoded.text for decoded in decode_segments_with_unit(runs, 0.055, config)}
    assert "CQ" in texts


def test_display_does_not_invent_qso_spacing() -> None:
    assert normalize_cw_display_text("CQ CQDEYO2AMU YO2AMUCQ CQYO2AMU YO2AMUPSEK") == "CQ CQDEYO2AMU YO2AMUCQ CQYO2AMU YO2AMUPSEK"


def test_display_preserves_tentative_brackets_without_qso_rewrite() -> None:
    assert normalize_cw_display_text("CQ CQDEYO[2AMU CQ]") == "CQ CQDEYO[2AMU CQ]"


def test_unknown_character_repair_splits_only_on_clear_internal_gap() -> None:
    # Primary classification is one invalid character: --..- .  The second
    # internal gap is clearly longer than the others, so it can be used as a
    # conservative letter boundary and yields M U.
    runs = [
        _mark(235, 0.000), _gap(100, 0.235), _mark(235, 0.335), _gap(180, 0.570),
        _mark(85, 0.750), _gap(100, 0.835), _mark(95, 0.935), _gap(105, 1.030), _mark(245, 1.135),
    ]
    config = DecoderConfig(
        flexible_timing=True,
        adaptive_gap_thresholds=False,
        element_letter_gap_units=2.8,
    )

    texts = {decoded.text for decoded in decode_segments_with_unit(runs, 0.08, config)}

    assert "MU" in texts


def test_unknown_character_repair_keeps_unknown_when_internal_gap_is_not_distinct() -> None:
    # Same invalid --..- shape, but the possible split gaps are too similar.
    # Do not force a repair just because valid Morse characters could be made.
    runs = [
        _mark(235, 0.000), _gap(130, 0.235), _mark(235, 0.365), _gap(140, 0.600),
        _mark(85, 0.740), _gap(130, 0.825), _mark(95, 0.955), _gap(120, 1.050), _mark(245, 1.170),
    ]
    config = DecoderConfig(
        flexible_timing=True,
        adaptive_gap_thresholds=False,
        element_letter_gap_units=2.8,
    )

    decoded = decode_segments_with_unit(runs, 0.08, config)[0]

    assert decoded.unresolved_tokens == 1
    assert decoded.text == "□"
