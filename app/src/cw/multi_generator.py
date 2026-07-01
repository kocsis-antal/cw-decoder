from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cw.generator import (
    GeneratorConfig,
    MorseEvent,
    build_events,
    generator_config_from_preset,
    override_generator_config,
    render_wave,
)
from cw.morse_table import normalize_text


@dataclass(frozen=True)
class MultiSource:
    source_id: str
    text: str
    start_s: float
    config: GeneratorConfig


@dataclass(frozen=True)
class MultiGenerateResult:
    wav_path: Path
    label_path: Path
    source_count: int
    sample_rate: int
    duration_s: float
    normalized_gain: float


_SOURCE_CONFIG_FIELDS = {
    "sample_rate": int,
    "tone_hz": float,
    "wpm": float,
    "amplitude": float,
    "ramp_ms": float,
    "timing_jitter": float,
    "dot_jitter": float,
    "dash_jitter": float,
    "element_gap_jitter": float,
    "letter_gap_jitter": float,
    "word_gap_jitter": float,
    "dash_ratio": float,
    "speed_wobble": float,
    "speed_wobble_hz": float,
    "frequency_drift_hz": float,
    "frequency_wobble_hz": float,
    "frequency_wobble_rate_hz": float,
    "amplitude_fade": float,
    "amplitude_fade_hz": float,
    "noise_snr_db": float,
    "seed": int,
}


def parse_source_spec(
    spec: str,
    *,
    index: int = 0,
    sample_rate: int | None = None,
    seed: int | None = None,
) -> MultiSource:
    values = _parse_key_value_spec(spec)

    text = values.pop("text", None)
    if not text:
        raise ValueError("Multi source spec must contain text=...")

    source_id = values.pop("id", None) or values.pop("source_id", None) or f"src{index + 1}"
    preset = values.pop("preset", "clean")
    start_s = _parse_float(values.pop("start", values.pop("start_s", "0")), "start")

    if "freq" in values:
        values["tone_hz"] = values.pop("freq")
    if "frequency" in values:
        values["tone_hz"] = values.pop("frequency")

    config = generator_config_from_preset(preset)
    overrides: dict[str, float | int | None] = {}
    for key, value in list(values.items()):
        normalized_key = key.replace("-", "_")
        parser = _SOURCE_CONFIG_FIELDS.get(normalized_key)
        if parser is None:
            supported = "id,text,preset,start,freq," + ",".join(sorted(_SOURCE_CONFIG_FIELDS))
            raise ValueError(f"Unsupported source option: {key!r}. Supported keys: {supported}")
        overrides[normalized_key] = parser(value)

    if sample_rate is not None:
        overrides["sample_rate"] = sample_rate
    if seed is not None and "seed" not in overrides:
        overrides["seed"] = seed + index

    config = override_generator_config(config, **overrides)
    return MultiSource(source_id=source_id, text=text, start_s=start_s, config=config)


def write_multi_sample(
    sources: list[MultiSource],
    wav_path: Path,
    *,
    sample_rate: int | None = None,
    normalize_peak: float | None = 0.95,
    noise_snr_db: float | None = None,
    seed: int | None = None,
) -> MultiGenerateResult:
    import soundfile as sf

    if not sources:
        raise ValueError("At least one source is required")

    effective_sample_rate = sample_rate or sources[0].config.sample_rate
    normalized_sources = [
        source if source.config.sample_rate == effective_sample_rate else replace(
            source,
            config=override_generator_config(source.config, sample_rate=effective_sample_rate),
        )
        for source in sources
    ]

    rendered_sources: list[tuple[MultiSource, list[MorseEvent], np.ndarray, int]] = []
    total_samples = 0
    for source in normalized_sources:
        events = build_events(source.text, source.config)
        signal = render_wave(events, source.config)
        start_sample = int(round(source.start_s * effective_sample_rate))
        if start_sample < 0:
            raise ValueError(f"Source {source.source_id!r} has negative start time")
        rendered_sources.append((source, events, signal, start_sample))
        total_samples = max(total_samples, start_sample + len(signal))

    mix = np.zeros(total_samples, dtype=np.float32)
    for _source, _events, signal, start_sample in rendered_sources:
        mix[start_sample : start_sample + len(signal)] += signal.astype(np.float32, copy=False)

    if noise_snr_db is not None:
        mix = _add_mix_noise(mix, noise_snr_db, seed)

    normalized_gain = 1.0
    if normalize_peak is not None:
        if not 0 < normalize_peak <= 1:
            raise ValueError("normalize_peak must be in the (0, 1] range")
        peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
        if peak > normalize_peak:
            normalized_gain = normalize_peak / peak
            mix = (mix * normalized_gain).astype(np.float32)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, mix, effective_sample_rate)

    label_path = wav_path.with_suffix(".labels.json")
    label_path.write_text(
        json.dumps(
            {
                "kind": "multi",
                "sample_rate": effective_sample_rate,
                "duration_s": _round_time(total_samples / effective_sample_rate),
                "source_count": len(normalized_sources),
                "normalize_peak": normalize_peak,
                "normalized_gain": _round_time(normalized_gain),
                "noise_snr_db": noise_snr_db,
                "seed": seed,
                "sources": [
                    _source_to_dict(source, events)
                    for source, events, _signal, _start_sample in rendered_sources
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return MultiGenerateResult(
        wav_path=wav_path,
        label_path=label_path,
        source_count=len(normalized_sources),
        sample_rate=effective_sample_rate,
        duration_s=total_samples / effective_sample_rate,
        normalized_gain=normalized_gain,
    )


def _source_to_dict(source: MultiSource, events: list[MorseEvent]) -> dict:
    return {
        "id": source.source_id,
        "raw_text": source.text,
        "text": normalize_text(source.text),
        "start_s": _round_time(source.start_s),
        "preset": source.config.preset,
        "tone_hz": source.config.tone_hz,
        "wpm": source.config.wpm,
        "amplitude": source.config.amplitude,
        "unit_s": _round_time(source.config.unit_s),
        "timing_jitter": source.config.timing_jitter,
        "dot_jitter": source.config.dot_jitter,
        "dash_jitter": source.config.dash_jitter,
        "element_gap_jitter": source.config.element_gap_jitter,
        "letter_gap_jitter": source.config.letter_gap_jitter,
        "word_gap_jitter": source.config.word_gap_jitter,
        "dash_ratio": source.config.dash_ratio,
        "speed_wobble": source.config.speed_wobble,
        "speed_wobble_hz": source.config.speed_wobble_hz,
        "frequency_drift_hz": source.config.frequency_drift_hz,
        "frequency_start_hz": _round_time(source.config.tone_hz - source.config.frequency_drift_hz / 2),
        "frequency_end_hz": _round_time(source.config.tone_hz + source.config.frequency_drift_hz / 2),
        "frequency_wobble_hz": source.config.frequency_wobble_hz,
        "frequency_wobble_rate_hz": source.config.frequency_wobble_rate_hz,
        "amplitude_fade": source.config.amplitude_fade,
        "amplitude_fade_hz": source.config.amplitude_fade_hz,
        "noise_snr_db": source.config.noise_snr_db,
        "seed": source.config.seed,
        "events": [_event_to_absolute_dict(event, source.start_s) for event in events],
    }


def _event_to_absolute_dict(event: MorseEvent, source_start_s: float) -> dict[str, str | float | None]:
    return {
        "kind": event.kind,
        "start_s": _round_time(source_start_s + event.start_s),
        "relative_start_s": _round_time(event.start_s),
        "duration_s": _round_time(event.duration_s),
        "symbol": event.symbol,
        "char": event.char,
    }


def _parse_key_value_spec(spec: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_part in spec.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid source spec part {part!r}; expected key=value")
        key, value = part.split("=", 1)
        values[key.strip().lower().replace("-", "_")] = value.strip()
    return values


def _parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {name}: {value!r}") from exc


def _add_mix_noise(signal: np.ndarray, noise_snr_db: float, seed: int | None) -> np.ndarray:
    if noise_snr_db <= 0:
        raise ValueError("noise_snr_db must be positive")
    signal_rms = float(np.sqrt(np.mean(signal**2))) if len(signal) else 0.0
    if signal_rms == 0:
        return signal
    noise_rms = signal_rms / (10 ** (noise_snr_db / 20))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_rms, len(signal)).astype(np.float32)
    return (signal + noise).astype(np.float32)


def _round_time(value: float) -> float:
    return round(float(value), 10)
