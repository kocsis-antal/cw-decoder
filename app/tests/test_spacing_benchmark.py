from pathlib import Path

from cw.spacing_benchmark import (
    SpacingBenchmarkConfig,
    check_spacing_expectations,
    parse_float_list,
    run_spacing_benchmark,
)


def test_parse_float_list():
    assert parse_float_list("40, 100,150") == (40.0, 100.0, 150.0)


def test_spacing_benchmark_marks_merge_and_split(tmp_path: Path):
    config = SpacingBenchmarkConfig(
        deltas_hz=(40.0, 100.0),
        merge_below_hz=60.0,
        split_from_hz=100.0,
        source_a_preset="straight",
        source_b_preset="straight",
        stream_frame_ms=80.0,
        stream_hop_ms=10.0,
        seed=123,
    )

    results = run_spacing_benchmark(
        "CQ CQ DE YU7NKA",
        "CQ CQ DE YT7MK",
        tmp_path,
        config,
    )

    assert [result.expected for result in results] == ["merge", "split"]
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[1].source_a_ok is True
    assert results[1].source_b_ok is True
    assert check_spacing_expectations(results).passed is True


def test_spacing_expectation_reports_failures(tmp_path: Path):
    config = SpacingBenchmarkConfig(
        deltas_hz=(100.0,),
        split_from_hz=100.0,
        stream_frame_ms=30.0,
        stream_hop_ms=5.0,
        seed=123,
    )

    # This test does not rely on the exact decoder outcome. It only verifies
    # that the expectation checker surfaces failing rows in a useful format.
    results = run_spacing_benchmark("E", "T", tmp_path, config)
    adjusted = [
        type(results[0])(
            delta_hz=results[0].delta_hz,
            expected="split",
            passed=False,
            wav_path=results[0].wav_path,
            detected_channels=results[0].detected_channels,
            carriers_hz=results[0].carriers_hz,
            source_a_ok=results[0].source_a_ok,
            source_b_ok=results[0].source_b_ok,
            decoded_texts=results[0].decoded_texts,
        )
    ]
    expectation = check_spacing_expectations(adjusted)
    assert expectation.passed is False
    assert "delta=100Hz" in expectation.failures[0]


def test_spacing_streaming_config_exposes_split_tracker_thresholds():
    from cw.spacing_benchmark import _streaming_config

    config = SpacingBenchmarkConfig(
        peak_min_separation_hz=40.0,
        track_match_hz=50.0,
        channel_merge_hz=90.0,
    )

    stream_config = _streaming_config(config)

    assert stream_config.peak_min_separation_hz == 40.0
    assert stream_config.track_match_hz == 50.0
    assert stream_config.channel_merge_hz == 90.0
