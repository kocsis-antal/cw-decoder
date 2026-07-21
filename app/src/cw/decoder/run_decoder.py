from __future__ import annotations

from dataclasses import dataclass
from cw.decoder.api import DecodedText, DecodeResult
from cw.decoder.config import DecoderConfig
from cw.decoder.timing import (
    RunState,
    TimedRun,
    decode_segments_with_unit,
    estimate_unit_s,
    expand_unknown_runs,
    timed_runs_from_signal_track,
    timing_quality_score,
    unit_candidates,
)
from cw.signal.models import SignalTrack


@dataclass(frozen=True)
class _DecodedPath:
    text: str
    unresolved_tokens: int
    timing_quality: float
    unit_s: float
    tokens: tuple = ()


class RunDecoder:
    """Decoder that consumes one signal-layer activity track.

    The input is already MARK/SPACE/UNKNOWN timing.  UNKNOWN is handled inside
    the decoder by local branching: every UNKNOWN run can independently become
    MARK or SPACE for one decoded path.  The only hard UNKNOWN gate here is the
    configured branch budget; signal quality is handled by unknown time ratio.
    """

    name = "run_decoder"

    def __init__(self, config: DecoderConfig) -> None:
        self.config = config

    def decode(self, track: SignalTrack) -> DecodeResult:
        if track.unknown_ratio > self.config.decoder_max_unknown_ratio:
            return DecodeResult(decoder=self.name, answers=())
        timed_runs = timed_runs_from_signal_track(track)
        if _unknown_branch_count(timed_runs) > self.config.decoder_max_unknown_branches:
            return DecodeResult(decoder=self.name, answers=())
        answers = _decode_timed_runs(timed_runs, self.config)
        return DecodeResult(decoder=self.name, answers=tuple(_to_public_answers(answers)))


def _unknown_branch_count(timed_runs: tuple[TimedRun, ...]) -> int:
    unknown_runs = sum(1 for run in timed_runs if run.state is RunState.UNKNOWN)
    return 2 ** unknown_runs


def _decode_timed_runs(timed_runs: tuple[TimedRun, ...], config: DecoderConfig) -> list[_DecodedPath]:
    if not any(run.state is RunState.MARK for run in timed_runs):
        return []

    answers: list[_DecodedPath] = []
    for hard_runs in expand_unknown_runs(timed_runs):
        answers.extend(_decode_whole_runs(hard_runs, config))

    answers = _unique_best_text_answers(answers)
    answers.sort(key=_answer_sort_key)
    return answers


def _decode_whole_runs(hard_runs: list, config: DecoderConfig) -> list[_DecodedPath]:
    try:
        initial_unit_s = estimate_unit_s(hard_runs)
    except ValueError:
        return []

    output: list[_DecodedPath] = []
    for unit_s in unit_candidates(initial_unit_s, config.unit_candidate_spread, config.unit_candidate_steps):
        for decoded in decode_segments_with_unit(hard_runs, unit_s, config):
            if not decoded.text:
                continue
            output.append(
                _DecodedPath(
                    text=decoded.text,
                    unresolved_tokens=decoded.unresolved_tokens,
                    timing_quality=round(float(timing_quality_score(decoded)), 6),
                    unit_s=round(float(unit_s), 6),
                    tokens=decoded.decode_tokens,
                )
            )
    return output


def _to_public_answers(answers: list[_DecodedPath]) -> list[DecodedText]:
    return [
        DecodedText(
            text=answer.text,
            unresolved_tokens=answer.unresolved_tokens,
            tokens=answer.tokens,
            timing_quality=answer.timing_quality,
        )
        for answer in answers
    ]


def _unique_best_text_answers(answers: list[_DecodedPath]) -> list[_DecodedPath]:
    best_by_text: dict[str, _DecodedPath] = {}
    for answer in answers:
        text = answer.text.strip()
        if not text:
            continue
        current = best_by_text.get(text)
        if current is None or _answer_sort_key(answer) < _answer_sort_key(current):
            best_by_text[text] = answer
    return list(best_by_text.values())


def _answer_sort_key(answer: _DecodedPath) -> tuple[float]:
    # Decoder-local ordering is timing-only. Selection must never infer quality
    # from the answer's list position/rank.
    return (answer.timing_quality,)


__all__ = ["RunDecoder"]
