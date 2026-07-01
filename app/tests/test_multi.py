from pathlib import Path

from cw.contest import ContestGrid
from cw.multi_decoder import CarrierDetectionConfig, detect_carriers, run_multi_live_contest
from cw.multi_generator import parse_source_spec, write_multi_sample


def test_parse_source_spec_applies_frequency_and_overrides() -> None:
    source = parse_source_spec(
        "id=me;freq=700;preset=straight;text=CQ DE YU7NKA;start=0.4;amplitude=0.45;seed=999",
        index=0,
        sample_rate=8000,
    )

    assert source.source_id == "me"
    assert source.text == "CQ DE YU7NKA"
    assert source.start_s == 0.4
    assert source.config.tone_hz == 700
    assert source.config.amplitude == 0.45
    assert source.config.seed == 999
    assert source.config.preset == "straight"


def test_write_multi_sample_creates_wav_and_multi_labels(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;text=CQ", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;text=DE;start=0.2", index=1, sample_rate=8000),
    ]

    result = write_multi_sample(sources, wav_path, sample_rate=8000)

    assert result.wav_path.exists()
    assert result.label_path.exists()
    label_text = result.label_path.read_text(encoding="utf-8")
    assert '"kind": "multi"' in label_text
    assert '"id": "one"' in label_text
    assert '"id": "two"' in label_text


def test_detect_carriers_finds_two_generated_sources(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;text=CQ CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;text=CQ CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    carriers = detect_carriers(
        wav_path,
        CarrierDetectionConfig(max_carriers=2, relative_threshold=0.10, min_separation_hz=120),
    )

    detected = sorted(round(carrier.frequency_hz) for carrier in carriers)
    assert len(detected) == 2
    assert abs(detected[0] - 700) <= 20
    assert abs(detected[1] - 1000) <= 20


def test_run_multi_live_contest_decodes_two_generated_sources(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;text=CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec("id=two;freq=1000;text=CQ DE YT7MK;start=0.2;amplitude=0.45", index=1, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    results = run_multi_live_contest(
        wav_path,
        ContestGrid(frame_ms=[10], hop_ms=[5], bandwidth_hz=[20, 40], threshold_ratio=[0.25, 0.35]),
        CarrierDetectionConfig(max_carriers=2, relative_threshold=0.10, min_separation_hz=120),
    )

    texts = {result.best_consensus.text for result in results}
    assert "CQ DE YU7NKA" in texts
    assert "CQ DE YT7MK" in texts
