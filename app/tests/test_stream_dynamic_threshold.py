from __future__ import annotations

from cw.decoder import DecodeResult
from cw.quality import QualityScore
from cw.stream_decode import _select_threshold_session_candidates, _threshold_candidate_ratios
from cw.stream_models import StreamSessionResult, StreamingConfig


def _session(text: str, score: float, first: float = 1.0, last: float = 4.0) -> StreamSessionResult:
    return StreamSessionResult(
        session_id=1,
        first_seen_s=first,
        last_seen_s=last,
        hits=12,
        final_time_s=last + 1.2,
        final_reason="silence_gap",
        quality=QualityScore(
            score=score,
            unknown_count=text.count("?"),
            token_count=max(1, len(text.split())),
            dot_count=0,
            dash_count=0,
            tone_ratio_error=0.0,
            gap_min_error=0.0,
            unit_cv=0.0,
        ),
        decoded=DecodeResult(
            text=text,
            tokens=[],
            runs=[],
            classified_runs=[],
            carrier_hz=700.0,
            unit_s=0.06,
            threshold=1.0,
        ),
    )


def test_threshold_candidate_ratios_include_base_and_deduplicate() -> None:
    config = StreamingConfig(threshold_ratio=0.35, threshold_ratios=(0.25, 0.35, 0.45))

    assert _threshold_candidate_ratios(config) == (0.25, 0.35, 0.45)


def test_dynamic_threshold_selection_prefers_lower_signal_score_without_text_bias() -> None:
    config = StreamingConfig()

    selected = _select_threshold_session_candidates(
        [
            _session("FQ TM8WWA", 18.0),
            _session("CQ TM8WWA", 22.0),
        ],
        config,
    )

    assert [session.decoded.text for session in selected] == ["FQ TM8WWA"]


def test_dynamic_threshold_selection_keeps_distinct_sessions() -> None:
    config = StreamingConfig()

    selected = _select_threshold_session_candidates(
        [
            _session("CQ A", 4.0, first=1.0, last=3.0),
            _session("CQ B", 5.0, first=6.0, last=8.0),
        ],
        config,
    )

    assert [session.session_id for session in selected] == [1, 2]
    assert [session.decoded.text for session in selected] == ["CQ A", "CQ B"]


def test_dynamic_threshold_selection_prefers_complete_active_candidate_over_short_fragment() -> None:
    config = StreamingConfig()

    selected = _select_threshold_session_candidates(
        [
            _session("DJ5CU KZ T T", 4.36, first=0.795, last=5.93),
            _session("I DJ5CU CQ CN II T", 27.72, first=0.35, last=7.93),
        ],
        config,
    )

    assert [session.decoded.text for session in selected] == ["I DJ5CU CQ CN II T"]
