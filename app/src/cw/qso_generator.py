from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cw.generator import build_events, generator_config_from_preset, override_generator_config
from cw.morse_table import normalize_text
from cw.multi_generator import MultiGenerateResult, MultiSource, write_multi_sample


@dataclass(frozen=True)
class ContestQsoConfig:
    caller_call: str = "YU7NKA"
    responder_call: str = "YT7MK"
    base_frequency_hz: float = 700.0
    responder_offset_hz: float = 6.0
    start_s: float = 0.0
    turn_gap_s: float = 1.6
    caller_preset: str = "straight"
    responder_preset: str = "straight"
    caller_wpm: float = 20.0
    responder_wpm: float = 18.0
    caller_amplitude: float = 0.60
    responder_amplitude: float = 0.50
    sample_rate: int = 8000
    seed: int | None = 123
    source_prefix: str = "qso"


def build_contest_qso_sources(config: ContestQsoConfig | None = None) -> list[MultiSource]:
    """Build a short contest-style simplex QSO as sequential transmissions.

    The two stations are intentionally placed close to each other in audio frequency.
    The streaming tracker should keep them on the same channel, but each over should
    become a separate session after the inter-turn silence.
    """
    config = config or ContestQsoConfig()
    _validate_qso_config(config)

    caller = normalize_text(config.caller_call)
    responder = normalize_text(config.responder_call)
    turns = [
        ("caller", f"CQ TEST {caller}"),
        ("responder", f"{caller} {responder}"),
        ("caller", f"{responder} 599 001"),
        ("responder", "TU 599 002"),
        ("caller", "TU"),
    ]

    sources: list[MultiSource] = []
    cursor_s = config.start_s
    for index, (role, text) in enumerate(turns, start=1):
        source_config = _source_config_for_turn(config, role, index)
        source_id = f"{config.source_prefix}-{role}-{index}"
        sources.append(
            MultiSource(
                source_id=source_id,
                text=text,
                start_s=round(cursor_s, 6),
                config=source_config,
            )
        )
        cursor_s += _events_duration_s(build_events(text, source_config)) + config.turn_gap_s

    return sources


def write_contest_qso_sample(
    wav_path: Path,
    config: ContestQsoConfig | None = None,
    *,
    normalize_peak: float | None = 0.95,
    mix_noise_snr_db: float | None = None,
) -> MultiGenerateResult:
    config = config or ContestQsoConfig()
    sources = build_contest_qso_sources(config)
    result = write_multi_sample(
        sources,
        wav_path,
        sample_rate=config.sample_rate,
        normalize_peak=normalize_peak,
        noise_snr_db=mix_noise_snr_db,
        seed=config.seed,
    )
    _annotate_qso_labels(result.label_path, config)
    return result


def _source_config_for_turn(config: ContestQsoConfig, role: str, turn_index: int):
    if role == "caller":
        preset = config.caller_preset
        tone_hz = config.base_frequency_hz
        wpm = config.caller_wpm
        amplitude = config.caller_amplitude
    elif role == "responder":
        preset = config.responder_preset
        tone_hz = config.base_frequency_hz + config.responder_offset_hz
        wpm = config.responder_wpm
        amplitude = config.responder_amplitude
    else:
        raise ValueError(f"Unsupported QSO role: {role!r}")

    seed = None if config.seed is None else config.seed + turn_index
    return override_generator_config(
        generator_config_from_preset(preset),
        sample_rate=config.sample_rate,
        tone_hz=tone_hz,
        wpm=wpm,
        amplitude=amplitude,
        seed=seed,
    )


def _events_duration_s(events) -> float:
    if not events:
        return 0.0
    return max(event.start_s + event.duration_s for event in events)


def _annotate_qso_labels(label_path: Path, config: ContestQsoConfig) -> None:
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    labels["scenario"] = "contest_qso"
    labels["qso"] = asdict(config)
    label_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def _validate_qso_config(config: ContestQsoConfig) -> None:
    if not normalize_text(config.caller_call):
        raise ValueError("caller_call must not be empty")
    if not normalize_text(config.responder_call):
        raise ValueError("responder_call must not be empty")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.turn_gap_s < 0:
        raise ValueError("turn_gap_s must not be negative")
    if config.caller_wpm <= 0 or config.responder_wpm <= 0:
        raise ValueError("WPM values must be positive")
    if config.caller_amplitude <= 0 or config.responder_amplitude <= 0:
        raise ValueError("amplitudes must be positive")
