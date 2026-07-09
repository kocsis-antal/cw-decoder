from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from cw.decoder.api import DecodeResult
from cw.receiving.models import ChannelSignal
from cw.selection.debug import ChannelSelectionDebug
from cw.signal.models import SignalRun, SignalState, SignalTrack
from cw.decoder.tokens import DecodeToken, tokens_to_dicts


@dataclass(frozen=True)
class DebugDecodedText:
    text: str
    unresolved_tokens: int
    tokens: tuple[DecodeToken, ...] = ()


@dataclass(frozen=True)
class DebugDecoderOutput:
    decoder: str
    answers: tuple[DebugDecodedText, ...] = ()


@dataclass(frozen=True)
class DebugSignalOutput:
    analyzer: str
    unknown_ratio: float
    runs: str
    decoders: tuple[DebugDecoderOutput, ...] = ()


@dataclass(frozen=True)
class ChannelDebugOutput:
    time_s: float
    channel_id: int
    carrier_hz: float
    state: str
    selected_text: str = ""
    signals: tuple[DebugSignalOutput, ...] = ()
    selection: ChannelSelectionDebug | None = None


def channel_debug_output_from_layers(
    *,
    time_s: float,
    channel: ChannelSignal,
    tracks_with_results: Iterable[tuple[SignalTrack, tuple[DecodeResult, ...]]],
    selection: ChannelSelectionDebug | None,
) -> ChannelDebugOutput:
    return ChannelDebugOutput(
        time_s=time_s,
        channel_id=channel.channel_id,
        carrier_hz=channel.carrier_hz,
        state=channel.state.value,
        selected_text="" if selection is None else selection.selected_text,
        signals=tuple(_debug_signal(track, results) for track, results in tracks_with_results),
        selection=selection,
    )


def channel_debug_output_to_dict(output: ChannelDebugOutput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "debug": "channel",
        "time_s": _rounded_float(output.time_s),
        "channel_id": int(output.channel_id),
        "carrier_hz": _rounded_float(output.carrier_hz),
        "state": output.state,
        "selected_text": output.selected_text,
        "signals": [
            {
                "analyzer": signal.analyzer,
                "unknown_ratio": _rounded_float(signal.unknown_ratio),
                "runs": signal.runs,
                "decoders": [
                    {
                        "decoder": decoder.decoder,
                        "answers": [
                            {"text": answer.text, "tokens": tokens_to_dicts(answer.tokens), "unresolved_tokens": answer.unresolved_tokens}
                            for answer in decoder.answers
                        ],
                    }
                    for decoder in signal.decoders
                ],
            }
            for signal in output.signals
        ],
    }
    if output.selection is not None:
        payload["selection"] = {
            "selected_text": output.selection.selected_text,
            "kept_previous": output.selection.kept_previous,
            "available_track_count": output.selection.available_track_count,
            "groups": [
                {
                    "text": group.text,
                    "unresolved_tokens": group.unresolved_tokens,
                    "support_count": group.support_count,
                    "support_score": _rounded_float(group.support_score),
                    "unknown_penalty_score": _rounded_float(group.unknown_penalty_score),
                    "final_score": _rounded_float(group.final_score),
                    "neighbor_stability": group.neighbor_stability,
                    "selected": group.selected,
                    "kept_previous": group.kept_previous,
                    "eligible": group.eligible,
                    "rejection_reason": group.rejection_reason,
                    "paths": [
                        {
                            "analyzer": path.analyzer,
                            "decoder": path.decoder,
                            "unresolved_tokens": path.unresolved_tokens,
                        }
                        for path in group.paths
                    ],
                }
                for group in output.selection.groups
            ],
        }
    return payload


def channel_debug_output_to_json(output: ChannelDebugOutput) -> str:
    return json.dumps(channel_debug_output_to_dict(output), ensure_ascii=False, separators=(",", ":"))


def channel_debug_outputs_to_jsonl(outputs: Iterable[ChannelDebugOutput]) -> str:
    return "\n".join(channel_debug_output_to_json(output) for output in outputs)


def _debug_signal(track: SignalTrack, results: tuple[DecodeResult, ...]) -> DebugSignalOutput:
    return DebugSignalOutput(
        analyzer=track.analyzer,
        unknown_ratio=track.unknown_ratio,
        runs=_compact_runs(track.runs),
        decoders=tuple(
            DebugDecoderOutput(
                decoder=result.decoder,
                answers=tuple(
                    DebugDecodedText(text=answer.text, tokens=answer.tokens, unresolved_tokens=max(0, int(answer.unresolved_tokens)))
                    for answer in result.answers
                ),
            )
            for result in results
        ),
    )


def _compact_runs(runs: tuple[SignalRun, ...], *, limit: int = 48) -> str:
    parts = [_format_run(run) for run in runs[:limit]]
    if len(runs) > limit:
        parts.append(f"+{len(runs) - limit} more")
    return " ".join(parts)


def _format_run(run: SignalRun) -> str:
    prefix = {
        SignalState.MARK: "M",
        SignalState.SPACE: "S",
        SignalState.UNKNOWN: "U",
    }[run.state]
    return f"{prefix}{max(0.0, run.duration_s) * 1000:.0f}"


def _rounded_float(value: float, *, digits: int = 3) -> float:
    return round(float(value), digits)


__all__ = [
    "ChannelDebugOutput",
    "channel_debug_output_from_layers",
    "channel_debug_output_to_dict",
    "channel_debug_output_to_json",
    "channel_debug_outputs_to_jsonl",
]
