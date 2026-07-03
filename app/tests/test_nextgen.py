from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cw.generator import GeneratorConfig, build_events, render_wave
from cw.nextgen import decode_raw_file_nextgen


def _write_s16le(path: Path, signal: np.ndarray) -> None:
    clipped = np.clip(signal, -1.0, 0.9999695)
    values = (clipped * 32768.0).astype("<i2")
    path.write_bytes(values.tobytes())


def test_nextgen_decodes_raw_carrier_without_qso_bias(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.65)
    signal = render_wave(build_events("HELLO WORLD", config), config)
    raw_path = tmp_path / "plain.s16le"
    _write_s16le(raw_path, signal)

    report = decode_raw_file_nextgen(raw_path, carriers=(700.0,), detect_carriers=0)

    assert report.carriers[0].text == "HELLO WORLD"
    assert report.carriers[0].best is not None
    assert report.carriers[0].best.confidence > 0.5


def test_nextgen_cli_json_detects_carrier_and_decodes(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=900.0, wpm=18.0, amplitude=0.7)
    signal = render_wave(build_events("CQ TEST", config), config)
    raw_path = tmp_path / "cq.s16le"
    _write_s16le(raw_path, signal)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cw.cli",
            "decode-raw",
            str(raw_path),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    assert payload["sample_rate"] == 8000
    assert payload["detected_carriers"]
    texts = [carrier["text"] for carrier in payload["carriers"]]
    assert any("CQ" in text and "TEST" in text for text in texts)


def test_nextgen_reports_multiple_timed_sessions_on_one_carrier(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.65)
    first = render_wave(build_events("CQ TEST", config), config)
    second = render_wave(build_events("HELLO", config), config)
    silence = np.zeros(int(config.sample_rate * 1.8), dtype=np.float32)
    raw_path = tmp_path / "sessions.s16le"
    _write_s16le(raw_path, np.concatenate([first, silence, second]))

    report = decode_raw_file_nextgen(
        raw_path,
        carriers=(700.0,),
        detect_carriers=0,
        session_gap_s=1.0,
    )

    sessions = report.carriers[0].sessions
    assert len(sessions) == 2
    assert sessions[0].text == "CQ TEST"
    assert sessions[1].text == "HELLO"
    assert sessions[0].end_s < sessions[1].start_s


def test_soft_bridge_repairs_short_non_silent_fade_gap() -> None:
    from cw.decoder import DetectedRun
    from cw.nextgen import _bridge_soft_fade_gaps
    from cw.stream_models import StreamingConfig

    runs = [
        DetectedRun("tone", 0.00, 0.06),
        DetectedRun("gap", 0.06, 0.04),
        DetectedRun("tone", 0.10, 0.08),
    ]
    frame_times = np.arange(0.0, 0.20, 0.01, dtype=np.float32)
    probabilities = np.full(len(frame_times), 0.9, dtype=np.float32)
    probabilities[(frame_times >= 0.06) & (frame_times < 0.10)] = 0.24

    bridged = _bridge_soft_fade_gaps(
        runs,
        probabilities,
        frame_times,
        0.0,
        StreamingConfig(
            soft_bridge_min_probability=0.18,
            soft_bridge_max_gap_ms=90.0,
            soft_bridge_gap_units=0.0,
        ),
    )

    assert len(bridged) == 1
    assert bridged[0].kind == "tone"
    assert bridged[0].duration_s == 0.18


def test_soft_bridge_does_not_merge_real_silent_element_gap() -> None:
    from cw.decoder import DetectedRun
    from cw.nextgen import _bridge_soft_fade_gaps
    from cw.stream_models import StreamingConfig

    runs = [
        DetectedRun("tone", 0.00, 0.06),
        DetectedRun("gap", 0.06, 0.04),
        DetectedRun("tone", 0.10, 0.08),
    ]
    frame_times = np.arange(0.0, 0.20, 0.01, dtype=np.float32)
    probabilities = np.full(len(frame_times), 0.9, dtype=np.float32)
    probabilities[(frame_times >= 0.06) & (frame_times < 0.10)] = 0.03

    bridged = _bridge_soft_fade_gaps(
        runs,
        probabilities,
        frame_times,
        0.0,
        StreamingConfig(
            soft_bridge_min_probability=0.18,
            soft_bridge_max_gap_ms=90.0,
            soft_bridge_gap_units=0.0,
        ),
    )

    assert bridged == runs


def test_lattice_decoder_keeps_alternate_dot_dash_boundary_alive() -> None:
    from cw.decoder import DetectedRun
    from cw.nextgen import _decode_lattice_candidates
    from cw.stream_models import StreamingConfig

    runs = [DetectedRun("tone", 0.0, 0.18)]
    frame_times = np.arange(0.0, 0.20, 0.01, dtype=np.float32)
    probabilities = np.ones(len(frame_times), dtype=np.float32)

    candidates = _decode_lattice_candidates(
        runs,
        probabilities,
        frame_times,
        carrier_hz=700.0,
        start_s=0.0,
        threshold_ratio=0.25,
        detector="test-lattice",
        threshold=0.5,
        noise_floor=0.0,
        signal_floor=1.0,
        duty_cycle=1.0,
        unit_s=0.10,
        config=StreamingConfig(lattice_tone_margin_units=0.45, lattice_max_candidates=4),
    )

    texts = {candidate.text for candidate in candidates}
    assert {"E", "T"}.issubset(texts)


def test_lattice_decoder_keeps_alternate_gap_boundary_alive() -> None:
    from cw.decoder import DetectedRun
    from cw.nextgen import _decode_lattice_candidates
    from cw.stream_models import StreamingConfig

    runs = [
        DetectedRun("tone", 0.00, 0.10),
        DetectedRun("gap", 0.10, 0.195),
        DetectedRun("tone", 0.295, 0.10),
    ]
    frame_times = np.arange(0.0, 0.42, 0.01, dtype=np.float32)
    probabilities = np.ones(len(frame_times), dtype=np.float32)
    probabilities[(frame_times >= 0.10) & (frame_times < 0.295)] = 0.01

    candidates = _decode_lattice_candidates(
        runs,
        probabilities,
        frame_times,
        carrier_hz=700.0,
        start_s=0.0,
        threshold_ratio=0.25,
        detector="test-lattice",
        threshold=0.5,
        noise_floor=0.0,
        signal_floor=1.0,
        duty_cycle=0.5,
        unit_s=0.10,
        config=StreamingConfig(lattice_gap_margin_units=0.20, lattice_max_candidates=4),
    )

    texts = {candidate.text for candidate in candidates}
    assert {"I", "EE"}.issubset(texts)


def test_viterbi_activity_bridges_short_weak_fade() -> None:
    from cw.nextgen import _viterbi_activity

    probabilities = np.asarray([0.92, 0.86, 0.31, 0.28, 0.82, 0.88], dtype=np.float32)

    active = _viterbi_activity(probabilities, transition_penalty=1.15)

    assert active.tolist() == [True, True, True, True, True, True]


def test_viterbi_activity_keeps_real_silent_gap() -> None:
    from cw.nextgen import _viterbi_activity

    probabilities = np.asarray([0.92, 0.86, 0.02, 0.01, 0.02, 0.03, 0.82, 0.88], dtype=np.float32)

    active = _viterbi_activity(probabilities, transition_penalty=1.15)

    assert active.tolist() == [True, True, False, False, False, False, True, True]



def test_symbol_hmm_decodes_probability_frames_without_precut_runs() -> None:
    from cw.nextgen import _decode_symbol_hmm_range
    from cw.stream_models import StreamingConfig

    unit_frames = 5
    probabilities: list[float] = []

    def add(value: float, frames: int) -> None:
        probabilities.extend([value] * frames)

    for symbol in "-.-.":
        add(0.92, 3 * unit_frames if symbol == "-" else unit_frames)
        add(0.02, unit_frames)
    add(0.02, 3 * unit_frames)
    for symbol in "--.-":
        add(0.92, 3 * unit_frames if symbol == "-" else unit_frames)
        add(0.02, unit_frames)

    probs = np.asarray(probabilities, dtype=np.float32)
    frame_times = np.arange(len(probs), dtype=np.float32) * 0.01

    candidates = _decode_symbol_hmm_range(
        probs,
        frame_times,
        0,
        len(probs),
        carrier_hz=700.0,
        start_s=0.0,
        unit_s=0.05,
        threshold=0.5,
        noise_floor=0.0,
        signal_floor=1.0,
        config=StreamingConfig(
            hop_ms=10.0,
            symbol_hmm_beam_width=64,
            symbol_hmm_max_candidates=5,
            symbol_hmm_unit_spread=0.0,
            symbol_hmm_unit_steps=1,
        ),
    )

    assert candidates
    assert candidates[0].detector == "symbol-hmm"
    assert candidates[0].text == "CQ"


def test_symbol_hmm_refines_unit_from_its_own_path() -> None:
    from cw.nextgen import NextgenRun, _estimate_unit_from_symbol_runs

    runs = (
        NextgenRun("tone", 0.00, 0.055, 0.95, symbol="."),
        NextgenRun("gap", 0.055, 0.052, 0.90, symbol="element_gap"),
        NextgenRun("tone", 0.107, 0.170, 0.93, symbol="-"),
        NextgenRun("gap", 0.277, 0.165, 0.88, symbol="letter_gap"),
        NextgenRun("tone", 0.442, 0.058, 0.92, symbol="."),
    )

    unit_s = _estimate_unit_from_symbol_runs(runs)

    assert unit_s is not None
    assert abs(unit_s - 0.055) < 0.006


def test_symbol_hmm_participates_even_when_direct_candidate_is_strong(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.70)
    signal = render_wave(build_events("CQ TEST", config), config)
    raw_path = tmp_path / "strong.s16le"
    _write_s16le(raw_path, signal)

    report = decode_raw_file_nextgen(
        raw_path,
        carriers=(700.0,),
        detect_carriers=0,
        max_candidates_per_carrier=30,
        max_candidates_per_session=30,
    )

    detectors = {candidate.detector for session in report.carriers[0].sessions for candidate in session.candidates}
    assert "symbol-hmm" in detectors


def test_character_hmm_decodes_complete_morse_templates_from_probability_frames() -> None:
    from cw.nextgen import _decode_character_hmm_range
    from cw.stream_models import StreamingConfig

    unit_frames = 5
    probabilities: list[float] = []

    def add(value: float, frames: int) -> None:
        probabilities.extend([value] * frames)

    for token in ("-.-.", "--.-"):
        for index, symbol in enumerate(token):
            add(0.92, 3 * unit_frames if symbol == "-" else unit_frames)
            if index < len(token) - 1:
                add(0.02, unit_frames)
        add(0.02, 3 * unit_frames)

    probs = np.asarray(probabilities, dtype=np.float32)
    frame_times = np.arange(len(probs), dtype=np.float32) * 0.01

    candidates = _decode_character_hmm_range(
        probs,
        frame_times,
        0,
        len(probs),
        carrier_hz=700.0,
        start_s=0.0,
        unit_s=0.05,
        threshold=0.5,
        noise_floor=0.0,
        signal_floor=1.0,
        config=StreamingConfig(hop_ms=10.0, symbol_hmm_beam_width=64, symbol_hmm_max_candidates=5),
    )

    assert candidates
    assert candidates[0].detector == "char-hmm"
    assert candidates[0].text == "CQ"


def test_symbol_hmm_rejects_unrealistic_tiny_unit_candidates() -> None:
    from cw.nextgen import _filter_symbol_hmm_unit_candidates
    from cw.stream_models import StreamingConfig

    units = _filter_symbol_hmm_unit_candidates((0.008, 0.025, 0.055, 0.31), StreamingConfig())

    assert units == (0.025, 0.055)


def test_candidate_evidence_penalizes_fragmented_single_letter_words() -> None:
    from cw.decoder import ClassifiedRun, DecodeResult
    from cw.nextgen import _candidate_evidence_score

    runs = [ClassifiedRun("tone", index * 0.1, 0.05, symbol=".", units=1.0) for index in range(16)]
    fragmented = DecodeResult(
        text="STE Q C Q C Q C Q C MEE H E",
        tokens=("...", "-", ".", "/", "--.-", "/", "-.-.", "/", "--.-", "/", "-.-.", "/", "--", "..", "."),
        runs=[],
        classified_runs=runs,
        carrier_hz=700.0,
        unit_s=0.05,
        threshold=0.2,
    )
    compact = DecodeResult(
        text="CQ CQ CQ CQCQ DES",
        tokens=("-.-.", "--.-", "/", "-.-.", "--.-", "/", "-.-.", "--.-", "/", "-.-.", "--.-", "-.-.", "--.-", "/", "-..", ".", "..."),
        runs=[],
        classified_runs=runs,
        carrier_hz=700.0,
        unit_s=0.05,
        threshold=0.2,
    )

    fragmented_score = _candidate_evidence_score(fragmented, quality_score=5.0, confidence=0.85)
    compact_score = _candidate_evidence_score(compact, quality_score=5.0, confidence=0.85)

    assert compact_score > fragmented_score + 8.0
