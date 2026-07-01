from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cw.contest import ContestGrid, ContestResult, LiveContestResult, run_contest, run_live_contest
from cw.evaluation import EvaluationResult, evaluate_wav
from cw.generator import generator_config_from_preset, override_generator_config, write_sample


@dataclass(frozen=True)
class BenchmarkExpectationResult:
    passed: bool
    failures: list[str]
    expected_pass_presets: list[str]
    allowed_fail_presets: list[str]


@dataclass(frozen=True)
class BenchmarkCaseResult:
    preset: str
    seed: int
    wav_path: Path
    labels_path: Path
    known_best: ContestResult
    live_best: LiveContestResult
    live_evaluation: EvaluationResult
    live_rank_in_known: int | None


def run_benchmark(
    text: str,
    out_dir: Path,
    presets: list[str],
    seeds: list[int],
    grid: ContestGrid,
    min_tone_hz: float = 200.0,
    max_tone_hz: float = 2000.0,
) -> list[BenchmarkCaseResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkCaseResult] = []

    for preset in presets:
        for seed in seeds:
            wav_path = out_dir / f"{_safe_name(preset)}_seed{seed}.wav"
            labels_path = _generate_sample(text, preset, seed, wav_path)
            known_results = run_contest(
                wav_path,
                labels_path,
                grid,
                min_tone_hz=min_tone_hz,
                max_tone_hz=max_tone_hz,
            )
            live_results = run_live_contest(
                wav_path,
                grid,
                min_tone_hz=min_tone_hz,
                max_tone_hz=max_tone_hz,
            )
            live_best = live_results[0]

            results.append(
                BenchmarkCaseResult(
                    preset=preset,
                    seed=seed,
                    wav_path=wav_path,
                    labels_path=labels_path,
                    known_best=known_results[0],
                    live_best=live_best,
                    live_evaluation=evaluate_wav(wav_path, labels_path, live_best.config),
                    live_rank_in_known=_rank_config(known_results, live_best.config),
                )
            )

    return results


def parse_string_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def check_benchmark_expectations(
    results: list[BenchmarkCaseResult],
    expected_pass_presets: list[str],
    allowed_fail_presets: list[str],
) -> BenchmarkExpectationResult:
    expected_pass = set(expected_pass_presets)
    allowed_fail = set(allowed_fail_presets)
    failures: list[str] = []

    for result in results:
        known_ok = result.known_best.evaluation.text_ok
        live_ok = result.live_evaluation.text_ok

        if result.preset in expected_pass and not (known_ok and live_ok):
            failures.append(
                f"{result.preset}/seed{result.seed}: expected known_ok and live_ok, "
                f"got known_ok={known_ok}, live_ok={live_ok}, live_text={result.live_best.decoded.text!r}"
            )
        elif result.preset not in expected_pass and result.preset not in allowed_fail:
            failures.append(f"{result.preset}/seed{result.seed}: preset has no expectation rule")

    return BenchmarkExpectationResult(
        passed=not failures,
        failures=failures,
        expected_pass_presets=expected_pass_presets,
        allowed_fail_presets=allowed_fail_presets,
    )


def _generate_sample(text: str, preset: str, seed: int, wav_path: Path) -> Path:
    config = override_generator_config(generator_config_from_preset(preset), seed=seed)
    return write_sample(text, wav_path, config)


def _rank_config(results: list[ContestResult], config) -> int | None:
    for result in results:
        if result.config == config:
            return result.rank
    return None


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
