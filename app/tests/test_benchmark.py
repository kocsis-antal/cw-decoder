from cw.benchmark import (
    check_benchmark_expectations,
    parse_int_list,
    parse_string_list,
    run_benchmark,
)
from cw.contest import ContestGrid


def test_parse_string_list() -> None:
    assert parse_string_list("clean, hard") == ["clean", "hard"]


def test_parse_int_list() -> None:
    assert parse_int_list("123, 999") == [123, 999]


def test_run_benchmark(tmp_path) -> None:
    grid = ContestGrid(
        frame_ms=[10.0],
        hop_ms=[10.0],
        bandwidth_hz=[40.0],
        threshold_ratio=[0.35],
    )

    results = run_benchmark("CQ", tmp_path, ["clean"], [123], grid)

    assert len(results) == 1
    assert results[0].preset == "clean"
    assert results[0].seed == 123
    assert results[0].known_best.evaluation.text_ok is True
    assert results[0].live_evaluation.text_ok is True
    assert results[0].live_rank_in_known == 1


def test_check_benchmark_expectations(tmp_path) -> None:
    grid = ContestGrid(
        frame_ms=[10.0],
        hop_ms=[10.0],
        bandwidth_hz=[40.0],
        threshold_ratio=[0.35],
    )
    results = run_benchmark("CQ", tmp_path, ["clean"], [123], grid)

    expectation = check_benchmark_expectations(
        results,
        expected_pass_presets=["clean"],
        allowed_fail_presets=["brutal"],
    )

    assert expectation.passed is True
    assert expectation.failures == []
