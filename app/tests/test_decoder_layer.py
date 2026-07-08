from __future__ import annotations

from dataclasses import fields

from cw.decoder.api import DecodeResult, DecodedText
from cw.decoder.run_decoder import RunDecoder
from cw.app.config import ProcessingConfig
from cw.morse_table import DECODE_ERROR_MARKER, decode_tokens_detailed
from cw.signal.models import SignalRun, SignalState, SignalTrack


def test_decode_result_contains_only_decoder_answers_without_ranking_fields() -> None:
    result_fields = {field.name for field in fields(DecodeResult)}
    answer_fields = {field.name for field in fields(DecodedText)}

    assert result_fields == {"decoder", "answers"}
    assert answer_fields == {"text", "unresolved_tokens", "tokens"}
    assert "channel_id" not in result_fields
    assert "carrier_hz" not in result_fields
    assert "score" not in answer_fields
    assert "confidence" not in answer_fields
    assert "metrics" not in answer_fields


def test_decoder_consumes_signal_track_only() -> None:
    track = SignalTrack(
        analyzer="unit-test",
        runs=(
            SignalRun(SignalState.MARK, 0.08),
            SignalRun(SignalState.SPACE, 0.08),
            SignalRun(SignalState.UNKNOWN, 0.04),
            SignalRun(SignalState.MARK, 0.20),
        ),
    )

    result = RunDecoder(ProcessingConfig()).decode(track)

    assert result.decoder == "run_decoder"
    assert isinstance(result.answers, tuple)


def test_decoder_expands_unknown_runs_locally_not_as_one_global_switch() -> None:
    from cw.decoder.timing import RunState, expand_unknown_runs, timed_runs_from_signal_track

    track = SignalTrack(
        analyzer="unit-test",
        runs=(
            SignalRun(SignalState.MARK, 0.10),
            SignalRun(SignalState.UNKNOWN, 0.05),
            SignalRun(SignalState.SPACE, 0.05),
            SignalRun(SignalState.UNKNOWN, 0.05),
            SignalRun(SignalState.MARK, 0.10),
        ),
    )

    timed_runs = timed_runs_from_signal_track(track)
    paths = expand_unknown_runs(timed_runs)
    path_durations = [tuple(round(run.duration_s, 3) for run in path) for path in paths]

    assert [run.state for run in timed_runs] == [RunState.MARK, RunState.UNKNOWN, RunState.SPACE, RunState.UNKNOWN, RunState.MARK]
    assert (0.15, 0.05, 0.15) in path_durations
    assert (0.10, 0.15, 0.10) in path_durations
    assert len(paths) == 4


def test_question_mark_is_valid_text_and_invalid_morse_token_gets_marker() -> None:
    decoded = decode_tokens_detailed(["..--..", "......-"])

    assert decoded.text == f"?{DECODE_ERROR_MARKER}"
    assert decoded.unresolved_tokens == 1


def test_decoded_answer_reports_only_unresolved_tokens_as_decoder_quality() -> None:
    answer = DecodedText(text=f"CQ{DECODE_ERROR_MARKER}", unresolved_tokens=1)

    assert answer.text.endswith(DECODE_ERROR_MARKER)
    assert answer.unresolved_tokens == 1
    assert not hasattr(answer, "decoded_chars")
    assert not hasattr(answer, "invalid_chars")
    assert not hasattr(answer, "unknown_ratio")
    assert not hasattr(answer, "consumed_ratio")


def test_decoder_treats_long_gap_as_word_gap_not_as_segment_split() -> None:
    track = SignalTrack(
        analyzer="unit-test",
        runs=(
            SignalRun(SignalState.MARK, 0.05),
            SignalRun(SignalState.SPACE, 1.40),
            SignalRun(SignalState.MARK, 0.15),
        ),
    )

    result = RunDecoder(ProcessingConfig()).decode(track)

    assert result.answers
    assert result.answers[0].text == "E   T"
    assert result.answers[0].unresolved_tokens == 0


def test_decoder_answer_order_does_not_use_text_as_tiebreaker() -> None:
    import inspect
    import cw.decoder.run_decoder as run_decoder

    source = inspect.getsource(run_decoder._answer_sort_key)

    assert "answer.text" not in source


def test_decoder_refuses_tracks_that_would_expand_too_many_unknown_branches() -> None:
    track = SignalTrack(
        analyzer="unit-test",
        runs=(
            SignalRun(SignalState.MARK, 0.08),
            SignalRun(SignalState.UNKNOWN, 0.01),
            SignalRun(SignalState.MARK, 0.08),
            SignalRun(SignalState.UNKNOWN, 0.01),
            SignalRun(SignalState.MARK, 0.08),
        ),
        unknown_ratio=0.05,
    )

    result = RunDecoder(ProcessingConfig(decoder_max_unknown_branches=2)).decode(track)

    assert result.answers == ()


def test_decoder_does_not_apply_fixed_answer_count_limit() -> None:
    import inspect
    import cw.decoder.run_decoder as run_decoder

    source = inspect.getsource(run_decoder._decode_timed_runs)

    assert "decoder_max_answers" not in source
    assert "answers[:" not in source
