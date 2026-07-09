from __future__ import annotations

import json
from typing import Any, Iterable

from cw.app.channel_output import ChannelOutput, channel_layer_info_to_dict
from cw.decoder.tokens import tokens_from_dicts, tokens_to_dicts, tokens_to_text


def channel_output_to_dict(output: ChannelOutput) -> dict[str, Any]:
    tokens = tokens_to_dicts(output.tokens)
    for index, token in enumerate(tokens):
        token["stable"] = index < int(output.stable_token_count)
    return {
        "channel_id": int(output.channel_id),
        "carrier_hz": _rounded_float(output.carrier_hz),
        "state": output.state,
        "tokens": tokens,
        "layers": channel_layer_info_to_dict(output.layers),
    }


def channel_output_from_dict(payload: dict[str, Any]) -> ChannelOutput | None:
    if "channel_id" not in payload:
        return None
    tokens = tokens_from_dicts(payload.get("tokens") or ())
    stable_count = sum(1 for token in (payload.get("tokens") or []) if isinstance(token, dict) and token.get("stable"))
    from cw.app.channel_output import channel_layer_info_from_dict

    return ChannelOutput(
        channel_id=int(payload.get("channel_id") or 0),
        carrier_hz=float(payload.get("carrier_hz") or 0.0),
        state=str(payload.get("state") or ""),
        text=tokens_to_text(tokens),
        tokens=tokens,
        stable_token_count=stable_count,
        layers=channel_layer_info_from_dict(payload.get("layers") or {}),
    )


def channel_output_to_json(output: ChannelOutput) -> str:
    return json.dumps(channel_output_to_dict(output), ensure_ascii=False, separators=(",", ":"))


def channel_outputs_to_jsonl(outputs: Iterable[ChannelOutput]) -> str:
    return "\n".join(channel_output_to_json(output) for output in outputs)


def _rounded_float(value: float, *, digits: int = 3) -> float:
    return round(float(value), digits)


__all__ = [
    "channel_output_from_dict",
    "channel_output_to_dict",
    "channel_output_to_json",
    "channel_outputs_to_jsonl",
]
