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


def channel_output_display_text(output: ChannelOutput) -> str:
    if output.text:
        return output.text
    if not output.tokens:
        return ""
    stable = output.tokens[: max(0, int(output.stable_token_count))]
    tentative = output.tokens[max(0, int(output.stable_token_count)) :]
    stable_text = tokens_to_text(stable)
    tentative_text = tokens_to_text(tentative)
    if tentative_text:
        return f"{stable_text} [{tentative_text}]" if stable_text else f"[{tentative_text}]"
    return stable_text


def _output_from_channel(channel: ChannelSignal, winner: ChannelWinner | None) -> ChannelOutput:
    return ChannelOutput(
        channel_id=channel.channel_id,
        carrier_hz=channel.carrier_hz,
        state=channel.state.value,
        text="" if winner is None else winner.text,
        tokens=() if winner is None else winner.tokens,
        stable_token_count=0 if winner is None else max(0, int(winner.stable_token_count)),
    )
