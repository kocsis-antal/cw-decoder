from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cw.generator import generator_config_from_preset, override_generator_config
from cw.morse_table import normalize_text
from cw.multi_generator import MultiSource, write_multi_sample
from cw.streaming import StreamingConfig, simulate_stream_from_wav


@dataclass(frozen=True)
class SpacingBenchmarkConfig:
    base_frequency_hz: float = 700.0
    deltas_hz: tuple[float, ...] = (40.0, 60.0, 80.0, 100.0, 120.0, 150.0)
    merge_below_hz: float = 60.0
    split_from_hz: float = 100.0
    source_a_preset: str = "field"
    source_b_preset: str = "straight"
    source_a_wpm: float = 20.0
    source_b_wpm: float = 18.0
    source_a_amplitude: float = 0.60
    source_b_amplitude: float = 0.45
    source_b_start_s: float = 0.40
    sample_rate: int = 8000
    seed: int | None = 123
    normalize_peak: float | None = 0.95
    mix_noise_snr_db: float | None = None
    stream_input_block_ms: float = 10.0
    stream_frame_ms: float = 30.0
    stream_hop_ms: float = 5.0
    tracker_frame_ms: float | None = 80.0
    tracker_hop_ms: float | None = 10.0
    stream_bandwidth_hz: float = 40.0
    stream_threshold_ratio: float = 0.35
    peak_relative_threshold: float = 0.25
    track_relative_threshold: float = 0.10
    max_final_score: float | None = 30.0
    shadow_suppression_hz: float | None = None
    shadow_score_margin: float = 15.0
    min_separation_hz: float = 80.0
    peak_min_separation_hz: float | None = None
    track_match_hz: float | None = None
    channel_merge_hz: float | None = None
    max_tracks: int = 5
    max_track_gap_s: float = 2.0
    carrier_smoothing: float = 0.20
    min_track_hits: int = 2
    emit_interval_s: float = 0.50
    session_gap_units: float = 20.0
    min_session_gap_s: float = 1.20


@dataclass(frozen=True)
class SpacingBenchmarkResult:
    delta_hz: float
    expected: str
    passed: bool | None
    wav_path: Path
    detected_channels: int
    carriers_hz: tuple[float, ...]
    source_a_ok: bool
    source_b_ok: bool
    decoded_texts: tuple[str, ...]

    @property
    def result_label(self) -> str:
        if self.passed is None:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True)
class SpacingExpectationResult:
    passed: bool
    failures: list[str]


def run_spacing_benchmark(
    text_a: str,
    text_b: str,
    out_dir: Path,
    config: SpacingBenchmarkConfig | None = None,
) -> list[SpacingBenchmarkResult]:
    config = config or SpacingBenchmarkConfig()
    _validate_spacing_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_text_a = normalize_text(text_a)
    expected_text_b = normalize_text(text_b)
    results: list[SpacingBenchmarkResult] = []

    for delta_hz in config.deltas_hz:
        wav_path = out_dir / f"spacing_{_safe_number(delta_hz)}hz.wav"
        sources = _sources_for_delta(text_a, text_b, delta_hz, config)
        write_multi_sample(
            sources,
            wav_path,
            sample_rate=config.sample_rate,
            normalize_peak=config.normalize_peak,
            noise_snr_db=config.mix_noise_snr_db,
            seed=config.seed,
        )

        stream_result = simulate_stream_from_wav(wav_path, _streaming_config(config))
        tracks = [track for track in stream_result.tracks if track.decoded.text]
        decoded_texts = tuple(track.decoded.text for track in tracks)
        carriers_hz = tuple(round(track.carrier_hz, 1) for track in tracks)
        source_a_ok = expected_text_a in decoded_texts
        source_b_ok = expected_text_b in decoded_texts
        expected = _expected_for_delta(delta_hz, config)
        passed: bool | None
        if expected == "merge":
            # Below the merge threshold we only require that the tracker does not
            # confidently split the two close tones into two valid decoded sources.
            passed = len(tracks) <= 1 or not (source_a_ok and source_b_ok)
        elif expected == "split":
            passed = source_a_ok and source_b_ok
        else:
            passed = None

        results.append(
            SpacingBenchmarkResult(
                delta_hz=delta_hz,
                expected=expected,
                passed=passed,
                wav_path=wav_path,
                detected_channels=len(tracks),
                carriers_hz=carriers_hz,
                source_a_ok=source_a_ok,
                source_b_ok=source_b_ok,
                decoded_texts=decoded_texts,
            )
        )

    return results


def check_spacing_expectations(results: list[SpacingBenchmarkResult]) -> SpacingExpectationResult:
    failures: list[str] = []
    for result in results:
        if result.passed is False:
            failures.append(
                f"delta={result.delta_hz:g}Hz expected={result.expected} "
                f"detected_channels={result.detected_channels} "
                f"source_a_ok={result.source_a_ok} source_b_ok={result.source_b_ok} "
                f"texts={list(result.decoded_texts)!r}"
            )
    return SpacingExpectationResult(passed=not failures, failures=failures)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("At least one numeric value is required")
    return values


def _sources_for_delta(
    text_a: str,
    text_b: str,
    delta_hz: float,
    config: SpacingBenchmarkConfig,
) -> list[MultiSource]:
    seed_a = None if config.seed is None else config.seed
    seed_b = None if config.seed is None else config.seed + 1
    source_a_config = override_generator_config(
        generator_config_from_preset(config.source_a_preset),
        sample_rate=config.sample_rate,
        tone_hz=config.base_frequency_hz,
        wpm=config.source_a_wpm,
        amplitude=config.source_a_amplitude,
        seed=seed_a,
    )
    source_b_config = override_generator_config(
        generator_config_from_preset(config.source_b_preset),
        sample_rate=config.sample_rate,
        tone_hz=config.base_frequency_hz + delta_hz,
        wpm=config.source_b_wpm,
        amplitude=config.source_b_amplitude,
        seed=seed_b,
    )
    return [
        MultiSource("spacing-a", text_a, 0.0, source_a_config),
        MultiSource("spacing-b", text_b, config.source_b_start_s, source_b_config),
    ]


def _streaming_config(config: SpacingBenchmarkConfig) -> StreamingConfig:
    return StreamingConfig(
        input_block_ms=config.stream_input_block_ms,
        frame_ms=config.stream_frame_ms,
        hop_ms=config.stream_hop_ms,
        tracker_frame_ms=config.tracker_frame_ms,
        tracker_hop_ms=config.tracker_hop_ms,
        bandwidth_hz=config.stream_bandwidth_hz,
        threshold_ratio=config.stream_threshold_ratio,
        peak_relative_threshold=config.peak_relative_threshold,
        track_relative_threshold=config.track_relative_threshold,
        max_final_score=config.max_final_score,
        shadow_suppression_hz=config.shadow_suppression_hz,
        shadow_score_margin=config.shadow_score_margin,
        min_separation_hz=config.min_separation_hz,
        peak_min_separation_hz=config.peak_min_separation_hz,
        track_match_hz=config.track_match_hz,
        channel_merge_hz=config.channel_merge_hz,
        max_tracks=config.max_tracks,
        max_track_gap_s=config.max_track_gap_s,
        carrier_smoothing=config.carrier_smoothing,
        min_track_hits=config.min_track_hits,
        emit_interval_s=config.emit_interval_s,
        stable_updates=True,
        session_gap_units=config.session_gap_units,
        min_session_gap_s=config.min_session_gap_s,
    )


def _expected_for_delta(delta_hz: float, config: SpacingBenchmarkConfig) -> str:
    if delta_hz < config.merge_below_hz:
        return "merge"
    if delta_hz >= config.split_from_hz:
        return "split"
    return "ambiguous"


def _validate_spacing_config(config: SpacingBenchmarkConfig) -> None:
    if config.base_frequency_hz <= 0:
        raise ValueError("base_frequency_hz must be positive")
    if not config.deltas_hz:
        raise ValueError("At least one delta is required")
    if any(delta <= 0 for delta in config.deltas_hz):
        raise ValueError("All deltas must be positive")
    if config.merge_below_hz <= 0 or config.split_from_hz <= 0:
        raise ValueError("merge_below_hz and split_from_hz must be positive")
    if config.merge_below_hz > config.split_from_hz:
        raise ValueError("merge_below_hz must not be higher than split_from_hz")
    if config.max_final_score is not None and config.max_final_score <= 0:
        raise ValueError("max_final_score must be positive when set")
    if config.shadow_suppression_hz is not None and config.shadow_suppression_hz < 0:
        raise ValueError("shadow_suppression_hz must not be negative when set")
    if config.shadow_score_margin < 0:
        raise ValueError("shadow_score_margin must not be negative")
    if config.peak_min_separation_hz is not None and config.peak_min_separation_hz <= 0:
        raise ValueError("peak_min_separation_hz must be positive when set")
    if config.track_match_hz is not None and config.track_match_hz <= 0:
        raise ValueError("track_match_hz must be positive when set")
    if config.channel_merge_hz is not None and config.channel_merge_hz <= 0:
        raise ValueError("channel_merge_hz must be positive when set")
    if config.tracker_frame_ms is not None and config.tracker_frame_ms <= 0:
        raise ValueError("tracker_frame_ms must be positive when set")
    if config.tracker_hop_ms is not None and config.tracker_hop_ms <= 0:
        raise ValueError("tracker_hop_ms must be positive when set")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.source_a_amplitude <= 0 or config.source_b_amplitude <= 0:
        raise ValueError("source amplitudes must be positive")
    if config.source_a_wpm <= 0 or config.source_b_wpm <= 0:
        raise ValueError("source WPM values must be positive")
    if config.source_b_start_s < 0:
        raise ValueError("source_b_start_s must not be negative")


def _safe_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "minus_").replace(".", "p")
