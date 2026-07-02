from cw.decoder import DetectedRun
from cw.stream_decode import _decode_run_segment, smooth_keying_runs
from cw.stream_models import StreamingConfig, validate_streaming_config


def _cq_with_split_dash_runs(unit_s: float = 0.060) -> list[DetectedRun]:
    runs: list[DetectedRun] = []
    time_s = 0.0

    def tone(duration_s: float) -> None:
        nonlocal time_s
        runs.append(DetectedRun("tone", round(time_s, 10), round(duration_s, 10)))
        time_s += duration_s

    def gap(duration_s: float) -> None:
        nonlocal time_s
        runs.append(DetectedRun("gap", round(time_s, 10), round(duration_s, 10)))
        time_s += duration_s

    # C: -.-.
    tone(3 * unit_s)
    gap(unit_s)
    tone(unit_s)
    gap(unit_s)
    tone(3 * unit_s)
    gap(unit_s)
    tone(unit_s)
    gap(3 * unit_s)

    # Q: --.-, with the second dash briefly dropping below threshold twice.
    tone(3 * unit_s)
    gap(unit_s)
    tone(0.052)
    gap(0.015)
    tone(0.050)
    gap(0.015)
    tone(0.052)
    gap(unit_s)
    tone(unit_s)
    gap(unit_s)
    tone(3 * unit_s)
    return runs


def test_smooth_keying_runs_repairs_short_dropouts_inside_dash() -> None:
    runs = _cq_with_split_dash_runs()
    raw = _decode_run_segment(runs, carrier_hz=700.0, threshold=1.0, config=StreamingConfig())
    assert raw.text != "CQ"

    cleaned = smooth_keying_runs(runs, merge_short_gaps_s=0.025, drop_short_tones_s=0.012)
    decoded = _decode_run_segment(cleaned, carrier_hz=700.0, threshold=1.0, config=StreamingConfig())

    assert decoded.text == "CQ"
    assert decoded.tokens == ["-.-.", "--.-"]


def test_live_deglitch_config_accepts_reasonable_defaults() -> None:
    validate_streaming_config(
        StreamingConfig(
            merge_short_gaps_ms=25.0,
            drop_short_tones_ms=12.0,
            unit_candidate_spread=0.30,
            unit_candidate_steps=13,
            punctuation_penalty=18.0,
        )
    )
