from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cw.morse_table import MORSE_BY_CHAR, normalize_text


@dataclass(frozen=True)
class GeneratorConfig:
    preset: str | None = None
    sample_rate: int = 8000
    tone_hz: float = 700.0
    wpm: float = 20.0
    amplitude: float = 0.6
    ramp_ms: float = 5.0
    timing_jitter: float = 0.0
    dot_jitter: float = 0.0
    dash_jitter: float = 0.0
    element_gap_jitter: float = 0.0
    letter_gap_jitter: float = 0.0
    word_gap_jitter: float = 0.0
    dash_ratio: float = 3.0
    speed_wobble: float = 0.0
    speed_wobble_hz: float = 0.0
    frequency_drift_hz: float = 0.0
    frequency_wobble_hz: float = 0.0
    frequency_wobble_rate_hz: float = 0.0
    amplitude_fade: float = 0.0
    amplitude_fade_hz: float = 0.0
    noise_snr_db: float | None = None
    seed: int | None = None

    @property
    def unit_s(self) -> float:
        return 1.2 / self.wpm


GENERATOR_PRESETS: dict[str, GeneratorConfig] = {
    "clean": GeneratorConfig(preset="clean"),
    "jitter": GeneratorConfig(preset="jitter", timing_jitter=0.15, seed=123),
    "drift": GeneratorConfig(preset="drift", frequency_drift_hz=25.0, seed=123),
    "noise": GeneratorConfig(preset="noise", noise_snr_db=25.0, seed=123),
    "straight": GeneratorConfig(
        preset="straight",
        dot_jitter=0.18,
        dash_jitter=0.22,
        element_gap_jitter=0.20,
        letter_gap_jitter=0.35,
        word_gap_jitter=0.20,
        dash_ratio=2.8,
        speed_wobble=0.06,
        speed_wobble_hz=0.04,
        seed=123,
    ),
    "field": GeneratorConfig(
        preset="field",
        dot_jitter=0.18,
        dash_jitter=0.22,
        element_gap_jitter=0.20,
        letter_gap_jitter=0.35,
        word_gap_jitter=0.20,
        dash_ratio=2.8,
        speed_wobble=0.06,
        speed_wobble_hz=0.04,
        frequency_drift_hz=40.0,
        frequency_wobble_hz=10.0,
        frequency_wobble_rate_hz=0.12,
        amplitude_fade=0.30,
        amplitude_fade_hz=0.18,
        noise_snr_db=18.0,
        seed=123,
    ),
    "hard": GeneratorConfig(
        preset="hard",
        timing_jitter=0.15,
        frequency_drift_hz=25.0,
        noise_snr_db=30.0,
        seed=123,
    ),
    "ugly": GeneratorConfig(
        preset="ugly",
        timing_jitter=0.20,
        speed_wobble=0.08,
        speed_wobble_hz=0.05,
        frequency_drift_hz=50.0,
        frequency_wobble_hz=12.0,
        frequency_wobble_rate_hz=0.12,
        amplitude_fade=0.35,
        amplitude_fade_hz=0.18,
        noise_snr_db=16.0,
        seed=123,
    ),
    "brutal": GeneratorConfig(
        preset="brutal",
        timing_jitter=0.30,
        speed_wobble=0.20,
        speed_wobble_hz=0.12,
        frequency_drift_hz=80.0,
        frequency_wobble_hz=25.0,
        frequency_wobble_rate_hz=0.25,
        amplitude_fade=0.55,
        amplitude_fade_hz=0.35,
        noise_snr_db=12.0,
        seed=123,
    ),
}


def generator_config_from_preset(preset: str | None) -> GeneratorConfig:
    preset_name = preset or "clean"
    if preset_name not in GENERATOR_PRESETS:
        supported = ", ".join(sorted(GENERATOR_PRESETS))
        raise ValueError(f"Unsupported generator preset: {preset_name!r}. Supported presets: {supported}")
    return GENERATOR_PRESETS[preset_name]


def override_generator_config(config: GeneratorConfig, **overrides) -> GeneratorConfig:
    active_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **active_overrides)


@dataclass(frozen=True)
class MorseEvent:
    kind: str
    start_s: float
    duration_s: float
    symbol: str
    char: str | None = None


def build_events(text: str, config: GeneratorConfig) -> list[MorseEvent]:
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    events: list[MorseEvent] = []
    cursor_s = 0.0
    pending_word_gap_units = 0

    for part in _text_parts(text):
        if part.isspace():
            if events:
                pending_word_gap_units += 7 * len(part)
            continue

        if pending_word_gap_units:
            cursor_s = _append_gap(events, cursor_s, config, rng, pending_word_gap_units, "word_gap")
            pending_word_gap_units = 0

        cursor_s = _append_word(events, cursor_s, config, rng, part)

    return events


def _text_parts(text: str) -> list[str]:
    return re.findall(r"\S+|\s+", text.upper().strip())


def _append_word(
    events: list[MorseEvent],
    cursor_s: float,
    config: GeneratorConfig,
    rng: np.random.Generator,
    word: str,
) -> float:
    for char_index, char in enumerate(word):
        code = MORSE_BY_CHAR.get(char)
        if code is None:
            raise ValueError(f"Unsupported Morse character: {char!r}")

        for symbol_index, symbol in enumerate(code):
            duration_s = _tone_duration(symbol, cursor_s, config, rng)
            events.append(MorseEvent("tone", cursor_s, duration_s, symbol, char))
            cursor_s += duration_s

            if symbol_index < len(code) - 1:
                cursor_s = _append_gap(events, cursor_s, config, rng, 1, "element_gap", char)

        if char_index < len(word) - 1:
            cursor_s = _append_gap(events, cursor_s, config, rng, 3, "letter_gap")

    return cursor_s


def render_wave(events: list[MorseEvent], config: GeneratorConfig) -> np.ndarray:
    if not events:
        return np.array([], dtype=np.float32)

    total_s = max(event.start_s + event.duration_s for event in events)
    total_samples = int(round(total_s * config.sample_rate))
    signal = np.zeros(total_samples, dtype=np.float32)

    for event in events:
        if event.kind != "tone":
            continue

        start = int(round(event.start_s * config.sample_rate))
        length = int(round(event.duration_s * config.sample_rate))
        end = min(start + length, total_samples)
        if end <= start:
            continue

        absolute_t = np.arange(start, end, dtype=np.float32) / config.sample_rate
        phase = _phase_at_time(absolute_t, total_s, config)
        tone = config.amplitude * _amplitude_gain(absolute_t, config) * np.sin(phase)
        signal[start:end] = tone * _ramp_envelope(end - start, config)

    return _add_noise(signal, config)


def write_sample(text: str, wav_path: Path, config: GeneratorConfig) -> Path:
    import soundfile as sf

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    events = build_events(text, config)
    signal = render_wave(events, config)
    sf.write(wav_path, signal, config.sample_rate)

    label_path = wav_path.with_suffix(".labels.json")
    label_path.write_text(
        json.dumps(
            {
                "raw_text": text,
                "text": normalize_text(text),
                "preset": config.preset,
                "sample_rate": config.sample_rate,
                "tone_hz": config.tone_hz,
                "wpm": config.wpm,
                "amplitude": config.amplitude,
                "unit_s": _round_time(config.unit_s),
                "timing_jitter": config.timing_jitter,
                "dot_jitter": config.dot_jitter,
                "dash_jitter": config.dash_jitter,
                "element_gap_jitter": config.element_gap_jitter,
                "letter_gap_jitter": config.letter_gap_jitter,
                "word_gap_jitter": config.word_gap_jitter,
                "dash_ratio": config.dash_ratio,
                "speed_wobble": config.speed_wobble,
                "speed_wobble_hz": config.speed_wobble_hz,
                "frequency_drift_hz": config.frequency_drift_hz,
                "frequency_start_hz": _round_time(config.tone_hz - config.frequency_drift_hz / 2),
                "frequency_end_hz": _round_time(config.tone_hz + config.frequency_drift_hz / 2),
                "frequency_wobble_hz": config.frequency_wobble_hz,
                "frequency_wobble_rate_hz": config.frequency_wobble_rate_hz,
                "amplitude_fade": config.amplitude_fade,
                "amplitude_fade_hz": config.amplitude_fade_hz,
                "noise_snr_db": config.noise_snr_db,
                "seed": config.seed,
                "events": [_event_to_dict(event) for event in events],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return label_path


def _append_gap(
    events: list[MorseEvent],
    cursor_s: float,
    config: GeneratorConfig,
    rng: np.random.Generator,
    units: int,
    symbol: str,
    char: str | None = None,
) -> float:
    duration_s = _gap_duration(units, symbol, cursor_s, config, rng)
    events.append(MorseEvent("gap", cursor_s, duration_s, symbol, char))
    return cursor_s + duration_s


def _tone_duration(
    symbol: str,
    cursor_s: float,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> float:
    if symbol == ".":
        return _duration_for_units(1.0, cursor_s, config, rng, config.dot_jitter)
    if symbol == "-":
        return _duration_for_units(config.dash_ratio, cursor_s, config, rng, config.dash_jitter)
    raise ValueError(f"Unsupported tone symbol: {symbol!r}")


def _gap_duration(
    units: int,
    symbol: str,
    cursor_s: float,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> float:
    jitter_by_symbol = {
        "element_gap": config.element_gap_jitter,
        "letter_gap": config.letter_gap_jitter,
        "word_gap": config.word_gap_jitter,
    }
    return _duration_for_units(units, cursor_s, config, rng, jitter_by_symbol.get(symbol, 0.0))


def _duration_for_units(
    units: float,
    cursor_s: float,
    config: GeneratorConfig,
    rng: np.random.Generator,
    specific_jitter: float,
) -> float:
    duration_s = units * config.unit_s
    duration_s *= _speed_factor(cursor_s, config)
    jitter = max(config.timing_jitter, specific_jitter)
    if jitter == 0:
        return duration_s
    factor = float(rng.uniform(1 - jitter, 1 + jitter))
    return float(duration_s * factor)


def _speed_factor(cursor_s: float, config: GeneratorConfig) -> float:
    if config.speed_wobble == 0 or config.speed_wobble_hz == 0:
        return 1.0
    phase = 2 * np.pi * config.speed_wobble_hz * cursor_s
    return 1 + config.speed_wobble * np.sin(phase)


def _phase_at_time(t: np.ndarray, total_s: float, config: GeneratorConfig) -> np.ndarray:
    if total_s <= 0:
        return 2 * np.pi * config.tone_hz * t

    start_hz = config.tone_hz - config.frequency_drift_hz / 2
    slope_hz_per_s = config.frequency_drift_hz / total_s
    cycles = start_hz * t + 0.5 * slope_hz_per_s * t**2

    if config.frequency_wobble_hz and config.frequency_wobble_rate_hz:
        angular_rate = 2 * np.pi * config.frequency_wobble_rate_hz
        cycles += -(config.frequency_wobble_hz / angular_rate) * np.cos(angular_rate * t)

    return 2 * np.pi * cycles


def _add_noise(signal: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    if config.noise_snr_db is None:
        return signal

    signal_rms = float(np.sqrt(np.mean(signal**2)))
    if signal_rms == 0:
        return signal

    noise_rms = signal_rms / (10 ** (config.noise_snr_db / 20))
    rng = np.random.default_rng(config.seed)
    noise = rng.normal(0.0, noise_rms, len(signal)).astype(np.float32)
    return (signal + noise).astype(np.float32)


def _validate_config(config: GeneratorConfig) -> None:
    if not 0 <= config.timing_jitter < 1:
        raise ValueError("timing_jitter must be in the [0, 1) range")
    for name, value in {
        "dot_jitter": config.dot_jitter,
        "dash_jitter": config.dash_jitter,
        "element_gap_jitter": config.element_gap_jitter,
        "letter_gap_jitter": config.letter_gap_jitter,
        "word_gap_jitter": config.word_gap_jitter,
    }.items():
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in the [0, 1) range")
    if config.dash_ratio <= 0:
        raise ValueError("dash_ratio must be positive")
    if not 0 <= config.speed_wobble < 1:
        raise ValueError("speed_wobble must be in the [0, 1) range")
    if not 0 <= config.amplitude_fade < 1:
        raise ValueError("amplitude_fade must be in the [0, 1) range")
    if config.speed_wobble_hz < 0:
        raise ValueError("speed_wobble_hz must not be negative")
    if config.frequency_wobble_rate_hz < 0:
        raise ValueError("frequency_wobble_rate_hz must not be negative")
    if config.amplitude_fade_hz < 0:
        raise ValueError("amplitude_fade_hz must not be negative")
    if config.noise_snr_db is not None and config.noise_snr_db <= 0:
        raise ValueError("noise_snr_db must be positive")


def _event_to_dict(event: MorseEvent) -> dict[str, str | float | None]:
    return {
        "kind": event.kind,
        "start_s": _round_time(event.start_s),
        "duration_s": _round_time(event.duration_s),
        "symbol": event.symbol,
        "char": event.char,
    }


def _round_time(value: float) -> float:
    return round(value, 10)


def _ramp_envelope(length: int, config: GeneratorConfig) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float32)
    ramp_samples = int(round(config.ramp_ms / 1000 * config.sample_rate))
    ramp_samples = max(0, min(ramp_samples, length // 2))

    if ramp_samples == 0:
        return envelope

    ramp = np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
    envelope[:ramp_samples] = ramp
    envelope[-ramp_samples:] = ramp[::-1]
    return envelope


def _amplitude_gain(t: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    if config.amplitude_fade == 0 or config.amplitude_fade_hz == 0:
        return np.ones_like(t, dtype=np.float32)
    phase = 2 * np.pi * config.amplitude_fade_hz * t
    return (1 - config.amplitude_fade * (0.5 + 0.5 * np.sin(phase))).astype(np.float32)
