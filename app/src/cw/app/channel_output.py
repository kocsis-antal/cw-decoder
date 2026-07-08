from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cw.decoder.tokens import DecodeToken, tokens_from_dicts, tokens_to_text
from cw.receiving.models import ChannelSignal, ReceiveChunk
from cw.selection.models import ChannelWinner


@dataclass(frozen=True)
class ChannelOutput:
    """Public application output snapshot for one receiving channel."""

    channel_id: int
    carrier_hz: float
    state: str
    text: str = ""
    tokens: tuple[DecodeToken, ...] = ()
    stable_token_count: int = 0


def channel_outputs_from_states(receive_chunk: ReceiveChunk, winners: tuple[ChannelWinner, ...] = ()) -> tuple[ChannelOutput, ...]:
    winners_by_channel = {winner.channel_id: winner for winner in winners}
    return tuple(_output_from_channel(channel, winners_by_channel.get(channel.channel_id)) for channel in receive_chunk.channels)


def channel_output_from_dict(payload: dict[str, Any]) -> ChannelOutput | None:
    if "channel_id" not in payload:
        return None
    raw_tokens = payload.get("tokens") or ()
    tokens = tokens_from_dicts(raw_tokens)
    stable_count = sum(1 for token in raw_tokens if isinstance(token, dict) and token.get("stable"))
    output = ChannelOutput(
        channel_id=int(payload.get("channel_id") or 0),
        carrier_hz=float(payload.get("carrier_hz") or 0.0),
        state=str(payload.get("state") or ""),
        text=str(payload.get("text") or ""),
        tokens=tokens,
        stable_token_count=stable_count,
    )
    if not output.text:
        output = ChannelOutput(
            channel_id=output.channel_id,
            carrier_hz=output.carrier_hz,
            state=output.state,
            text=channel_output_display_text(output),
            tokens=output.tokens,
            stable_token_count=output.stable_token_count,
        )
    return output


def append_non_overlapping_tokens(
    committed: tuple[DecodeToken, ...],
    incoming: tuple[DecodeToken, ...],
) -> tuple[DecodeToken, ...]:
    if not incoming:
        return committed
    if not committed:
        return incoming
    max_overlap = min(len(committed), len(incoming))
    for size in range(max_overlap, 0, -1):
        if all(_same_token(left, right) for left, right in zip(committed[-size:], incoming[:size])):
            return committed + incoming[size:]
    return committed + incoming


def stable_tentative_display_text(
    stable_tokens: tuple[DecodeToken, ...],
    tentative_tokens: tuple[DecodeToken, ...],
) -> str:
    stable_text = tokens_to_text(stable_tokens)
    tentative_text = tokens_to_text(tentative_tokens)
    if tentative_text:
        return f"{stable_text} [{tentative_text}]" if stable_text else f"[{tentative_text}]"
    return stable_text


def _same_token(left: DecodeToken, right: DecodeToken) -> bool:
    if left.kind != right.kind or left.value != right.value:
        return False
    if left.start_s is None or right.start_s is None or left.end_s is None or right.end_s is None:
        return left.signature == right.signature
    return abs(left.start_s - right.start_s) <= 0.02 and abs(left.end_s - right.end_s) <= 0.02


def channel_output_display_text(output: ChannelOutput) -> str:
    if output.text:
        return output.text
    if not output.tokens:
        return ""
    stable = output.tokens[: max(0, int(output.stable_token_count))]
    tentative = output.tokens[max(0, int(output.stable_token_count)) :]
    return stable_tentative_display_text(stable, tentative)


def _output_from_channel(channel: ChannelSignal, winner: ChannelWinner | None) -> ChannelOutput:
    return ChannelOutput(
        channel_id=channel.channel_id,
        carrier_hz=channel.carrier_hz,
        state=channel.state.value,
        text="" if winner is None else winner.text,
        tokens=() if winner is None else winner.tokens,
        stable_token_count=0 if winner is None else max(0, int(winner.stable_token_count)),
    )
