from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TextIO


def iter_formatted_debug_jsonl(lines: Iterable[str]) -> Iterable[str]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            yield line
            continue
        if isinstance(payload, dict) and payload.get("debug") == "channel":
            yield from _format_channel_debug(payload)
        else:
            yield line


def print_debug_output_view(input_stream: TextIO, output_stream: TextIO) -> None:
    for line in iter_formatted_debug_jsonl(input_stream):
        print(line, file=output_stream)


def print_debug_output_view_file(path: Path, output_stream: TextIO) -> None:
    with path.open("r", encoding="utf-8") as input_stream:
        print_debug_output_view(input_stream, output_stream)


def _format_channel_debug(payload: dict) -> list[str]:
    selected = payload.get("selected_text") or ""
    lines = [
        f"DEBUG t={float(payload.get('time_s') or 0.0):.2f}s "
        f"ch{int(payload.get('channel_id') or 0)} "
        f"{float(payload.get('carrier_hz') or 0.0):.1f}Hz "
        f"{payload.get('state') or ''} "
        f"selected={_quote(selected) if selected else '-'}"
    ]
    signals = payload.get("signals") or []
    if signals:
        lines.append("  signals:")
        for signal in signals:
            lines.extend(_format_signal(signal))
    selection = payload.get("selection") or {}
    groups = selection.get("groups") or []
    if groups:
        lines.append("  selection:")
        for group in groups:
            marker = "*" if group.get("selected") else " "
            kept = " kept" if group.get("kept_previous") else ""
            rejected = ""
            if group.get("eligible") is False:
                reason = str(group.get("rejection_reason") or "rejected")
                rejected = f" rejected={reason}"
            lines.append(
                f"   {marker} {_quote(str(group.get('text') or ''))} "
                f"bad={int(group.get('unresolved_tokens') or 0)} "
                f"support={int(group.get('support_count') or 0)} "
                f"families={int(group.get('family_count') or 0)} "
                f"neighbors={int(group.get('neighbor_stability') or 0)}{kept}{rejected}"
            )
    return lines


def _format_signal(signal: dict) -> list[str]:
    lines = [
        f"    {signal.get('analyzer') or ''} "
        f"unknown={float(signal.get('unknown_ratio') or 0.0):.3f} "
        f"runs={signal.get('runs') or '-'}"
    ]
    for decoder in signal.get("decoders") or []:
        answers = decoder.get("answers") or []
        rendered_answers = ", ".join(
            f"{_quote(str(answer.get('text') or ''))}/bad={int(answer.get('unresolved_tokens') or 0)}"
            for answer in answers
        )
        lines.append(f"      {decoder.get('decoder') or ''}: {rendered_answers or '-'}")
    return lines


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


__all__ = [
    "iter_formatted_debug_jsonl",
    "print_debug_output_view",
    "print_debug_output_view_file",
]
