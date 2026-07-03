from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cw.decoder import (
    DecoderConfig,
    _classified_runs_to_tokens,
    _energy_threshold,
    _power_spectrum_frames,
    _runs_from_activity,
)
from cw.morse_table import decode_tokens
from cw.multi_decoder import _local_peak_indices
from cw.quality import score_decode_result
from cw.stream_decode import (
    _band_energy,
    _classify_runs_with_adaptive_gaps,
    _decode_with_unit,
    _estimate_unit_from_runs,
    smooth_keying_runs,
)
from cw.stream_models import StreamingConfig
from cw.stream_sources import decode_raw_pcm, pcm_sample_width_bytes


@dataclass(frozen=True)
class RawAudioStats:
    samples: int
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float


@dataclass(frozen=True)
class CarrierCandidate:
    carrier_hz: float
    relative_power: float
    power: float


@dataclass(frozen=True)
class DurationSummary:
    count: int
    min_units: float | None
    median_units: float | None
    max_units: float | None
    values_units: tuple[float, ...]


@dataclass(frozen=True)
class ThresholdAnalysis:
    threshold_ratio: float
    threshold: float
    noise_floor: float
    signal_floor: float
    active_duty_cycle: float
    tone_run_count: int
    gap_run_count: int
    unit_s: float | None
    unit_wpm: float | None
    element_gap_boundary_units: float
    word_gap_boundary_units: float
    text: str
    tokens: tuple[str, ...]
    quality_score: float | None
    unknown_count: int | None
    tone_durations: DurationSummary
    gap_durations: DurationSummary
    classified_counts: dict[str, int]
    run_preview: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class CarrierAnalysis:
    carrier_hz: float
    analyses: tuple[ThresholdAnalysis, ...]


@dataclass(frozen=True)
class RawAnalysisReport:
    path: str
    sample_rate: int
    sample_format: str
    channels: int
    start_s: float
    duration_s: float
    audio: RawAudioStats
    detected_carriers: tuple[CarrierCandidate, ...]
    carriers: tuple[CarrierAnalysis, ...]


def parse_float_csv(value: str | None) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def read_raw_audio_slice(
    path: Path,
    *,
    sample_rate: int,
    sample_format: str = "s16le",
    channels: int = 1,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if start_s < 0:
        raise ValueError("start_s must not be negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive when set")

    frame_width = pcm_sample_width_bytes(sample_format) * channels
    start_frame = int(round(start_s * sample_rate))
    frames_to_read = None if duration_s is None else int(round(duration_s * sample_rate))

    with Path(path).open("rb") as raw_file:
        raw_file.seek(start_frame * frame_width)
        if frames_to_read is None:
            raw = raw_file.read()
        else:
            raw = raw_file.read(frames_to_read * frame_width)

    usable_bytes = (len(raw) // frame_width) * frame_width
    return decode_raw_pcm(raw[:usable_bytes], sample_format=sample_format, channels=channels)


def analyze_raw_file(
    path: Path,
    *,
    sample_rate: int,
    sample_format: str = "s16le",
    channels: int = 1,
    start_s: float = 0.0,
    duration_s: float | None = None,
    carriers: tuple[float, ...] = (),
    detect_carriers: int = 5,
    min_tone_hz: float = 200.0,
    max_tone_hz: float = 3000.0,
    min_separation_hz: float = 80.0,
    peak_relative_threshold: float = 0.10,
    frame_ms: float = 30.0,
    hop_ms: float = 5.0,
    bandwidth_hz: float = 40.0,
    threshold_ratios: tuple[float, ...] = (0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    adaptive_gap_thresholds: bool = True,
    element_letter_gap_units: float = 2.0,
    default_word_gap_units: float = 7.0,
    gap_cluster_min_ratio: float = 1.45,
    gap_cluster_min_delta_units: float = 1.0,
    gap_cluster_min_lower_count: int = 2,
    merge_short_gaps_ms: float = 25.0,
    drop_short_tones_ms: float = 12.0,
    unit_candidate_spread: float = 0.0,
    unit_candidate_steps: int = 1,
    punctuation_penalty: float = 0.0,
    preview_runs: int = 24,
) -> RawAnalysisReport:
    signal = read_raw_audio_slice(
        path,
        sample_rate=sample_rate,
        sample_format=sample_format,
        channels=channels,
        start_s=start_s,
        duration_s=duration_s,
    )
    spectrum, freqs = _power_spectrum_frames(signal, sample_rate, frame_ms, hop_ms)
    detected = _detect_carriers_from_spectrum(
        spectrum,
        freqs,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        max_carriers=detect_carriers,
        min_separation_hz=min_separation_hz,
        relative_threshold=peak_relative_threshold,
    )
    selected_carriers = carriers or tuple(candidate.carrier_hz for candidate in detected)
    stream_config = StreamingConfig(
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_tone_hz=min_tone_hz,
        max_tone_hz=max_tone_hz,
        bandwidth_hz=bandwidth_hz,
        threshold_ratio=threshold_ratios[0] if threshold_ratios else 0.35,
        adaptive_gap_thresholds=adaptive_gap_thresholds,
        element_letter_gap_units=element_letter_gap_units,
        default_word_gap_units=default_word_gap_units,
        gap_cluster_min_ratio=gap_cluster_min_ratio,
        gap_cluster_min_delta_units=gap_cluster_min_delta_units,
        gap_cluster_min_lower_count=gap_cluster_min_lower_count,
        merge_short_gaps_ms=merge_short_gaps_ms,
        drop_short_tones_ms=drop_short_tones_ms,
        unit_candidate_spread=unit_candidate_spread,
        unit_candidate_steps=unit_candidate_steps,
        punctuation_penalty=punctuation_penalty,
    )
    ratios = threshold_ratios or (0.35,)
    carrier_reports = tuple(
        _analyze_carrier(
            spectrum,
            freqs,
            carrier_hz=carrier_hz,
            first_frame_start_s=start_s,
            ratios=ratios,
            config=stream_config,
            preview_runs=preview_runs,
        )
        for carrier_hz in selected_carriers
    )
    actual_duration_s = len(signal) / sample_rate if sample_rate else 0.0
    return RawAnalysisReport(
        path=str(path),
        sample_rate=sample_rate,
        sample_format=sample_format,
        channels=channels,
        start_s=round(start_s, 6),
        duration_s=round(actual_duration_s, 6),
        audio=_audio_stats(signal, sample_rate),
        detected_carriers=detected,
        carriers=carrier_reports,
    )


def report_to_json(report: RawAnalysisReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)


def format_human_report(report: RawAnalysisReport) -> str:
    lines: list[str] = []
    lines.append(
        f"raw={report.path} sample_rate={report.sample_rate} format={report.sample_format} "
        f"channels={report.channels} start_s={report.start_s:.3f} duration_s={report.duration_s:.3f}"
    )
    lines.append(
        f"audio samples={report.audio.samples} rms_dbfs={report.audio.rms_dbfs:.1f} "
        f"peak_dbfs={report.audio.peak_dbfs:.1f}"
    )
    if report.detected_carriers:
        lines.append("detected carriers:")
        for candidate in report.detected_carriers:
            lines.append(
                f"  {candidate.carrier_hz:8.1f} Hz  rel={candidate.relative_power:5.3f}  "
                f"power={candidate.power:.3g}"
            )
    for carrier in report.carriers:
        lines.append(f"carrier {carrier.carrier_hz:.1f} Hz:")
        lines.append(
            "  thr   duty  tones gaps unit_ms  wpm   score unk  text"
        )
        for analysis in carrier.analyses:
            unit_ms = "-" if analysis.unit_s is None else f"{analysis.unit_s * 1000:7.1f}"
            wpm = "-" if analysis.unit_wpm is None else f"{analysis.unit_wpm:5.1f}"
            score = "-" if analysis.quality_score is None else f"{analysis.quality_score:5.1f}"
            unk = "-" if analysis.unknown_count is None else str(analysis.unknown_count)
            lines.append(
                f"  {analysis.threshold_ratio:4.2f}  {analysis.active_duty_cycle:5.2%} "
                f"{analysis.tone_run_count:5d} {analysis.gap_run_count:4d} "
                f"{unit_ms:>7} {wpm:>5} {score:>5} {unk:>3}  {analysis.text}"
            )
            lines.append(
                f"        tone_units={_format_duration_summary(analysis.tone_durations)} "
                f"gap_units={_format_duration_summary(analysis.gap_durations)} "
                f"classes={analysis.classified_counts}"
            )
            if analysis.run_preview:
                preview = " ".join(
                    f"{item['kind'][0]}:{item['duration_s']:.3f}s/{item.get('units', '-')}{item.get('symbol', '')}"
                    for item in analysis.run_preview[:12]
                )
                lines.append(f"        runs: {preview}")
    return "\n".join(lines)


def _analyze_carrier(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    carrier_hz: float,
    first_frame_start_s: float,
    ratios: tuple[float, ...],
    config: StreamingConfig,
    preview_runs: int,
) -> CarrierAnalysis:
    energy = np.asarray(
        [_band_energy(frame_spectrum, freqs, carrier_hz, config.bandwidth_hz) for frame_spectrum in spectrum],
        dtype=np.float32,
    )
    analyses = tuple(
        _analyze_threshold(
            energy,
            carrier_hz=carrier_hz,
            first_frame_start_s=first_frame_start_s,
            threshold_ratio=ratio,
            config=config,
            preview_runs=preview_runs,
        )
        for ratio in ratios
    )
    return CarrierAnalysis(carrier_hz=carrier_hz, analyses=analyses)


def _analyze_threshold(
    energy: np.ndarray,
    *,
    carrier_hz: float,
    first_frame_start_s: float,
    threshold_ratio: float,
    config: StreamingConfig,
    preview_runs: int,
) -> ThresholdAnalysis:
    decoder_config = DecoderConfig(
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
        min_tone_hz=config.min_tone_hz,
        max_tone_hz=config.max_tone_hz,
        bandwidth_hz=config.bandwidth_hz,
        threshold_ratio=threshold_ratio,
        target_tone_hz=carrier_hz,
    )
    threshold = _energy_threshold(energy, decoder_config)
    active = energy > threshold
    raw_runs = _runs_from_activity(active, config.hop_ms / 1000)
    runs = smooth_keying_runs(
        raw_runs,
        merge_short_gaps_s=config.merge_short_gaps_ms / 1000,
        drop_short_tones_s=config.drop_short_tones_ms / 1000,
    )
    if first_frame_start_s:
        from cw.stream_decode import _offset_runs

        runs = _offset_runs(runs, first_frame_start_s)
    try:
        unit_s = _estimate_unit_from_runs(runs)
    except ValueError:
        return ThresholdAnalysis(
            threshold_ratio=threshold_ratio,
            threshold=float(threshold),
            noise_floor=float(np.percentile(energy, 10)) if len(energy) else 0.0,
            signal_floor=float(np.percentile(energy, 95)) if len(energy) else 0.0,
            active_duty_cycle=float(np.mean(active)) if len(active) else 0.0,
            tone_run_count=sum(1 for run in runs if run.kind == "tone"),
            gap_run_count=sum(1 for run in runs if run.kind == "gap"),
            unit_s=None,
            unit_wpm=None,
            element_gap_boundary_units=config.element_letter_gap_units,
            word_gap_boundary_units=config.default_word_gap_units,
            text="",
            tokens=(),
            quality_score=None,
            unknown_count=None,
            tone_durations=_duration_summary([]),
            gap_durations=_duration_summary([]),
            classified_counts={},
            run_preview=_run_preview(runs, None, preview_runs),
        )

    classified = _classify_runs_with_adaptive_gaps(runs, unit_s, config)
    tokens = tuple(_classified_runs_to_tokens(classified))
    text = decode_tokens(list(tokens))
    decoded = _decode_with_unit(runs, carrier_hz, float(threshold), unit_s, config)
    quality = score_decode_result(decoded)
    classified_counts: dict[str, int] = {}
    for run in classified:
        if run.kind == "gap":
            classified_counts[run.symbol] = classified_counts.get(run.symbol, 0) + 1
        elif run.kind == "tone":
            classified_counts[run.symbol] = classified_counts.get(run.symbol, 0) + 1

    tone_units = [run.units for run in classified if run.kind == "tone"]
    gap_units = [run.units for run in classified if run.kind == "gap"]
    return ThresholdAnalysis(
        threshold_ratio=threshold_ratio,
        threshold=float(threshold),
        noise_floor=float(np.percentile(energy, 10)) if len(energy) else 0.0,
        signal_floor=float(np.percentile(energy, 95)) if len(energy) else 0.0,
        active_duty_cycle=float(np.mean(active)) if len(active) else 0.0,
        tone_run_count=sum(1 for run in runs if run.kind == "tone"),
        gap_run_count=sum(1 for run in runs if run.kind == "gap"),
        unit_s=unit_s,
        unit_wpm=1.2 / unit_s if unit_s > 0 else None,
        element_gap_boundary_units=config.element_letter_gap_units,
        word_gap_boundary_units=_word_gap_boundary_from_classified(classified, config),
        text=text,
        tokens=tokens,
        quality_score=quality.score,
        unknown_count=quality.unknown_count,
        tone_durations=_duration_summary(tone_units),
        gap_durations=_duration_summary(gap_units),
        classified_counts=classified_counts,
        run_preview=_classified_run_preview(classified, preview_runs),
    )


def _detect_carriers_from_spectrum(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    min_tone_hz: float,
    max_tone_hz: float,
    max_carriers: int,
    min_separation_hz: float,
    relative_threshold: float,
) -> tuple[CarrierCandidate, ...]:
    if max_carriers <= 0 or len(spectrum) == 0:
        return ()
    summed = np.sum(spectrum, axis=0)
    mask = (freqs >= min_tone_hz) & (freqs <= max_tone_hz)
    if not np.any(mask):
        return ()
    powers = summed[mask]
    search_freqs = freqs[mask]
    max_power = float(np.max(powers)) if len(powers) else 0.0
    if max_power <= 0:
        return ()

    # Keep carrier detection deliberately simple and conservative: choose real
    # accumulated spectral peaks.  The previous temporal/local-peak admission was
    # too eager in live WebSDR audio and promoted sidebands/noise shadows to
    # separate public carriers, which made the receiver look busy but less useful.
    candidates = _local_peak_indices(powers)
    candidates.sort(key=lambda index: float(powers[index]), reverse=True)
    selected: list[CarrierCandidate] = []
    for index in candidates:
        power = float(powers[index])
        relative = power / max_power
        if relative < relative_threshold:
            continue
        carrier_hz = float(search_freqs[index])
        if any(abs(carrier_hz - existing.carrier_hz) < min_separation_hz for existing in selected):
            continue
        selected.append(CarrierCandidate(round(carrier_hz, 3), round(relative, 6), power))
        if len(selected) >= max_carriers:
            break
    return tuple(selected)


def _audio_stats(signal: np.ndarray, sample_rate: int) -> RawAudioStats:
    rms = float(np.sqrt(np.mean(np.square(signal)))) if len(signal) else 0.0
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    return RawAudioStats(
        samples=int(len(signal)),
        duration_s=round(len(signal) / sample_rate, 6) if sample_rate else 0.0,
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
    )


def _dbfs(value: float) -> float:
    if value <= 0:
        return -999.0
    return round(20.0 * float(np.log10(value)), 3)


def _duration_summary(values: list[float]) -> DurationSummary:
    rounded = tuple(round(float(value), 3) for value in values)
    if not rounded:
        return DurationSummary(0, None, None, None, ())
    return DurationSummary(
        count=len(rounded),
        min_units=min(rounded),
        median_units=round(float(np.median(rounded)), 3),
        max_units=max(rounded),
        values_units=rounded,
    )


def _format_duration_summary(summary: DurationSummary) -> str:
    if summary.count == 0:
        return "n=0"
    preview = ",".join(f"{value:.2f}" for value in summary.values_units[:8])
    if len(summary.values_units) > 8:
        preview += ",..."
    return (
        f"n={summary.count} min={summary.min_units:.2f} med={summary.median_units:.2f} "
        f"max={summary.max_units:.2f} [{preview}]"
    )


def _classified_run_preview(classified_runs, limit: int) -> tuple[dict[str, float | str], ...]:
    items: list[dict[str, float | str]] = []
    for run in classified_runs[: max(0, limit)]:
        items.append(
            {
                "kind": run.kind,
                "start_s": round(float(run.start_s), 3),
                "duration_s": round(float(run.duration_s), 3),
                "units": round(float(run.units), 3),
                "symbol": run.symbol,
            }
        )
    return tuple(items)


def _run_preview(runs, unit_s: float | None, limit: int) -> tuple[dict[str, float | str], ...]:
    items: list[dict[str, float | str]] = []
    for run in runs[: max(0, limit)]:
        item: dict[str, float | str] = {
            "kind": run.kind,
            "start_s": round(float(run.start_s), 3),
            "duration_s": round(float(run.duration_s), 3),
        }
        if unit_s and unit_s > 0:
            item["units"] = round(float(run.duration_s / unit_s), 3)
        items.append(item)
    return tuple(items)


def _word_gap_boundary_from_classified(classified_runs, config: StreamingConfig) -> float:
    word_gaps = [run.units for run in classified_runs if run.kind == "gap" and run.symbol == "word_gap"]
    letter_gaps = [run.units for run in classified_runs if run.kind == "gap" and run.symbol == "letter_gap"]
    if word_gaps and letter_gaps:
        return round((max(letter_gaps) * min(word_gaps)) ** 0.5, 3)
    return config.default_word_gap_units
