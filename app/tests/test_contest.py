from cw.contest import ContestGrid, parse_float_list, run_contest, run_live_contest, summarize_live_consensus
from cw.generator import GeneratorConfig, write_sample


def test_parse_float_list() -> None:
    assert parse_float_list("5, 10,20") == [5.0, 10.0, 20.0]


def test_parse_float_list_empty() -> None:
    assert parse_float_list("") == []


def test_run_contest_ranks_clean_decode_first(tmp_path) -> None:
    wav_path = tmp_path / "cq.wav"
    labels_path = write_sample("CQ CQ", wav_path, GeneratorConfig())
    grid = ContestGrid(
        frame_ms=[20.0],
        hop_ms=[5.0, 10.0],
        bandwidth_hz=[40.0],
        threshold_ratio=[0.35],
    )

    results = run_contest(wav_path, labels_path, grid)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].evaluation.text_ok is True
    assert results[0].evaluation.token_accuracy == 1.0


def test_run_live_contest_ranks_clean_decode_first(tmp_path) -> None:
    wav_path = tmp_path / "cq.wav"
    write_sample("CQ CQ", wav_path, GeneratorConfig())
    grid = ContestGrid(
        frame_ms=[20.0],
        hop_ms=[5.0, 10.0],
        bandwidth_hz=[40.0],
        threshold_ratio=[0.35],
    )

    results = run_live_contest(wav_path, grid)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].decoded.text == "CQ CQ"


def test_summarize_live_consensus_groups_by_decoded_text(tmp_path) -> None:
    wav_path = tmp_path / "cq.wav"
    write_sample("CQ CQ", wav_path, GeneratorConfig())
    grid = ContestGrid(
        frame_ms=[20.0],
        hop_ms=[5.0, 10.0],
        bandwidth_hz=[40.0],
        threshold_ratio=[0.35],
    )

    results = run_live_contest(wav_path, grid)
    consensus = summarize_live_consensus(results)

    assert len(consensus) == 1
    assert consensus[0].rank == 1
    assert consensus[0].text == "CQ CQ"
    assert consensus[0].count == 2
    assert consensus[0].share == 1.0
    assert consensus[0].best_rank == results[0].rank
