from cw.decoder import DecodeResult
from cw.quality import QualityScore
from cw.stream_filter import filter_final_tracks
from cw.stream_models import StreamTrackResult, StreamingConfig


def _track(track_id: int, carrier_hz: float, score: float, text: str = "CQ") -> StreamTrackResult:
    decoded = DecodeResult(
        text=text,
        tokens=[],
        runs=[],
        classified_runs=[],
        carrier_hz=carrier_hz,
        unit_s=0.055,
        threshold=1.0,
    )
    quality = QualityScore(
        score=score,
        unknown_count=0,
        token_count=max(1, len(text.split())),
        dot_count=1,
        dash_count=1,
        tone_ratio_error=0.0,
        gap_min_error=0.0,
        unit_cv=0.0,
    )
    return StreamTrackResult(
        track_id=track_id,
        carrier_hz=carrier_hz,
        first_seen_s=0.0,
        last_seen_s=1.0,
        hits=20,
        quality=quality,
        decoded=decoded,
        sessions=[],
    )


def test_final_track_filter_drops_low_quality_shadow_track() -> None:
    good = _track(1, 715.0, 7.0, "CQ CQ DE YU7NKA")
    shadow = _track(2, 650.0, 34.5, "CQ CQ DE YUHEE")

    filtered = filter_final_tracks(
        [good, shadow],
        StreamingConfig(max_final_score=30.0, channel_merge_hz=80.0),
    )

    assert filtered == [good]


def test_final_track_filter_can_be_disabled_for_debug() -> None:
    good = _track(1, 715.0, 7.0, "CQ CQ DE YU7NKA")
    shadow = _track(2, 650.0, 34.5, "CQ CQ DE YUHEE")

    filtered = filter_final_tracks(
        [good, shadow],
        StreamingConfig(max_final_score=None, shadow_suppression_hz=0.0),
    )

    assert filtered == [good, shadow]


def test_shadow_suppression_keeps_close_tracks_when_quality_is_similar() -> None:
    left = _track(1, 700.0, 7.0, "CQ CQ DE YU7NKA")
    right = _track(2, 780.0, 9.0, "CQ CQ DE YT7MK")

    filtered = filter_final_tracks(
        [left, right],
        StreamingConfig(max_final_score=30.0, shadow_suppression_hz=100.0, shadow_score_margin=15.0),
    )

    assert filtered == [left, right]
