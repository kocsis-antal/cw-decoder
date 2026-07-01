import json
from pathlib import Path

import soundfile as sf

from cw.generator import (
    GENERATOR_PRESETS,
    GeneratorConfig,
    build_events,
    generator_config_from_preset,
    override_generator_config,
    write_sample,
)


def test_build_events_uses_standard_morse_timing() -> None:
    config = GeneratorConfig(wpm=20.0)

    events = build_events("EE T", config)

    assert [(event.kind, event.symbol, event.duration_s) for event in events] == [
        ("tone", ".", config.unit_s),
        ("gap", "letter_gap", 3 * config.unit_s),
        ("tone", ".", config.unit_s),
        ("gap", "word_gap", 7 * config.unit_s),
        ("tone", "-", 3 * config.unit_s),
    ]


def test_element_gaps_belong_to_current_character() -> None:
    config = GeneratorConfig(wpm=20.0)

    events = build_events("C", config)

    assert [(event.kind, event.symbol, event.char) for event in events] == [
        ("tone", "-", "C"),
        ("gap", "element_gap", "C"),
        ("tone", ".", "C"),
        ("gap", "element_gap", "C"),
        ("tone", "-", "C"),
        ("gap", "element_gap", "C"),
        ("tone", ".", "C"),
    ]


def test_extra_spaces_stretch_word_gap() -> None:
    config = GeneratorConfig(wpm=20.0)

    single_space = build_events("DE YU7NKA", config)
    double_space = build_events("DE  YU7NKA", config)

    single_gap = next(event for event in single_space if event.symbol == "word_gap")
    double_gap = next(event for event in double_space if event.symbol == "word_gap")

    assert single_gap.duration_s == 7 * config.unit_s
    assert double_gap.duration_s == 14 * config.unit_s


def test_timing_jitter_is_reproducible_with_seed() -> None:
    config = GeneratorConfig(wpm=20.0, timing_jitter=0.15, seed=123)

    first = build_events("CQ", config)
    second = build_events("CQ", config)
    clean = build_events("CQ", GeneratorConfig(wpm=20.0))

    assert first == second
    assert [event.duration_s for event in first] != [event.duration_s for event in clean]


def test_generator_presets() -> None:
    assert set(GENERATOR_PRESETS) == {
        "clean",
        "jitter",
        "drift",
        "noise",
        "straight",
        "field",
        "hard",
        "ugly",
        "brutal",
    }
    assert generator_config_from_preset("straight").dot_jitter == 0.18
    assert generator_config_from_preset("straight").dash_jitter == 0.22
    assert generator_config_from_preset("straight").element_gap_jitter == 0.20
    assert generator_config_from_preset("straight").letter_gap_jitter == 0.35
    assert generator_config_from_preset("straight").word_gap_jitter == 0.20
    assert generator_config_from_preset("straight").dash_ratio == 2.8
    assert generator_config_from_preset("straight").speed_wobble == 0.06
    assert generator_config_from_preset("field").dot_jitter == 0.18
    assert generator_config_from_preset("field").dash_jitter == 0.22
    assert generator_config_from_preset("field").dash_ratio == 2.8
    assert generator_config_from_preset("field").frequency_drift_hz == 40.0
    assert generator_config_from_preset("field").frequency_wobble_hz == 10.0
    assert generator_config_from_preset("field").frequency_wobble_rate_hz == 0.12
    assert generator_config_from_preset("field").amplitude_fade == 0.30
    assert generator_config_from_preset("field").amplitude_fade_hz == 0.18
    assert generator_config_from_preset("field").noise_snr_db == 18.0
    assert generator_config_from_preset("hard").timing_jitter == 0.15
    assert generator_config_from_preset("hard").frequency_drift_hz == 25.0
    assert generator_config_from_preset("hard").noise_snr_db == 30.0
    assert generator_config_from_preset("ugly").timing_jitter == 0.20
    assert generator_config_from_preset("ugly").speed_wobble == 0.08
    assert generator_config_from_preset("ugly").frequency_drift_hz == 50.0
    assert generator_config_from_preset("ugly").frequency_wobble_hz == 12.0
    assert generator_config_from_preset("ugly").amplitude_fade == 0.35
    assert generator_config_from_preset("ugly").noise_snr_db == 16.0
    assert generator_config_from_preset("brutal").timing_jitter == 0.30
    assert generator_config_from_preset("brutal").speed_wobble == 0.20
    assert generator_config_from_preset("brutal").frequency_drift_hz == 80.0
    assert generator_config_from_preset("brutal").frequency_wobble_hz == 25.0
    assert generator_config_from_preset("brutal").amplitude_fade == 0.55
    assert generator_config_from_preset("brutal").noise_snr_db == 12.0


def test_override_generator_config_keeps_unspecified_preset_values() -> None:
    config = override_generator_config(generator_config_from_preset("hard"), seed=999, wpm=18.0)

    assert config.preset == "hard"
    assert config.seed == 999
    assert config.wpm == 18.0
    assert config.timing_jitter == 0.15
    assert config.frequency_drift_hz == 25.0
    assert config.noise_snr_db == 30.0


def test_write_sample_creates_wav_and_labels(tmp_path: Path) -> None:
    wav_path = tmp_path / "cq.wav"

    label_path = write_sample(
        "CQ",
        wav_path,
        GeneratorConfig(
            sample_rate=8000,
            wpm=20.0,
            timing_jitter=0.1,
            dot_jitter=0.11,
            dash_jitter=0.12,
            element_gap_jitter=0.13,
            letter_gap_jitter=0.14,
            word_gap_jitter=0.15,
            dash_ratio=2.9,
            speed_wobble=0.2,
            speed_wobble_hz=0.1,
            frequency_drift_hz=25.0,
            frequency_wobble_hz=5.0,
            frequency_wobble_rate_hz=0.2,
            amplitude_fade=0.3,
            amplitude_fade_hz=0.4,
            noise_snr_db=30.0,
            seed=7,
        ),
    )

    assert wav_path.exists()
    assert label_path.exists()

    signal, sample_rate = sf.read(wav_path)
    labels = json.loads(label_path.read_text(encoding="utf-8"))

    assert sample_rate == 8000
    assert len(signal) > 0
    assert labels["text"] == "CQ"
    assert labels["raw_text"] == "CQ"
    assert labels["preset"] is None
    assert labels["events"][0]["kind"] == "tone"
    assert labels["events"][0]["symbol"] == "-"
    assert labels["events"][1]["symbol"] == "element_gap"
    assert labels["events"][1]["char"] == "C"
    assert labels["timing_jitter"] == 0.1
    assert labels["dot_jitter"] == 0.11
    assert labels["dash_jitter"] == 0.12
    assert labels["element_gap_jitter"] == 0.13
    assert labels["letter_gap_jitter"] == 0.14
    assert labels["word_gap_jitter"] == 0.15
    assert labels["dash_ratio"] == 2.9
    assert labels["speed_wobble"] == 0.2
    assert labels["speed_wobble_hz"] == 0.1
    assert labels["frequency_drift_hz"] == 25.0
    assert labels["frequency_start_hz"] == 687.5
    assert labels["frequency_end_hz"] == 712.5
    assert labels["frequency_wobble_hz"] == 5.0
    assert labels["frequency_wobble_rate_hz"] == 0.2
    assert labels["amplitude_fade"] == 0.3
    assert labels["amplitude_fade_hz"] == 0.4
    assert labels["noise_snr_db"] == 30.0
    assert labels["seed"] == 7
