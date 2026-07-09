from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

from cw.app.channel_output import (
    ChannelOutput,
    append_non_overlapping_tokens,
    channel_output_from_dict as _parse_channel_output,
    channel_output_display_text,
    stable_tentative_display_text,
)


ANSI_CLEAR_SCREEN = "\x1b[2J\x1b[H"
ANSI_CLEAR_TO_END = "\x1b[J"
ANSI_HIDE_CURSOR = "\x1b[?25l"
ANSI_SHOW_CURSOR = "\x1b[?25h"


@dataclass
class _ChannelRow:
    channel_id: int
    carrier_hz: float
    state: str = "candidate"
    text: str = ""
    committed_tokens: tuple = ()
    tentative_tokens: tuple = ()
    layers: object | None = None
    last_seen_s: float = 0.0
    last_text_s: float = 0.0


class HumanDashboardRenderer:
    """Stateful terminal view for public channel snapshots.

    The renderer consumes only ``ChannelOutput`` objects, the same public model
    used by JSONL output.  It does not know signal analyzers, decoder answers,
    selection scores or transcripts; it only keeps a small display cache so a
    human can watch multiple channels without append-only JSON noise.
    """

    def __init__(
        self,
        output_stream: TextIO,
        *,
        refresh_interval_s: float = 0.25,
        max_rows: int = 16,
        inactive_retention_s: float = 6.0,
        decoded_retention_s: float = 180.0,
        use_ansi: bool | None = None,
    ) -> None:
        self._output_stream = output_stream
        self._refresh_interval_s = max(0.0, refresh_interval_s)
        self._max_rows = max(1, max_rows)
        self._inactive_retention_s = max(0.0, inactive_retention_s)
        self._decoded_retention_s = max(self._inactive_retention_s, decoded_retention_s)
        self._use_ansi = _use_ansi_dashboard(output_stream) if use_ansi is None else use_ansi
        self._rows: dict[int, _ChannelRow] = {}
        self._current_time_s = 0.0
        self._last_render_s = -10**9
        self._started = False
        self._closed = False

    def start(self) -> None:
        self.render(force=True)

    def tick(self, time_s: float) -> None:
        self._current_time_s = max(self._current_time_s, float(time_s))
        self._drop_stale_rows()
        self.render()

    def emit(self, output: ChannelOutput) -> None:
        self._current_time_s = max(self._current_time_s, 0.0)
        if output.state == "dropped":
            self._rows.pop(output.channel_id, None)
            self.render(force=True)
            return

        row = self._row_for(output)
        row.state = output.state or row.state or "active"
        row.layers = output.layers
        row.last_seen_s = self._current_time_s
        text = self._update_row_text(row, output)
        if text:
            row.last_text_s = self._current_time_s

        self._drop_stale_rows()
        self.render(force=output.state == "dormant" or bool(text))

    def close(self) -> None:
        if self._closed:
            return
        self.render(force=True)
        if self._use_ansi and self._started:
            print(ANSI_SHOW_CURSOR, end="", file=self._output_stream, flush=True)
            print(file=self._output_stream, flush=True)
        self._closed = True

    def render(self, *, force: bool = False) -> None:
        if not force and self._current_time_s - self._last_render_s < self._refresh_interval_s:
            return
        self._last_render_s = self._current_time_s
        lines = self._build_lines()
        if self._use_ansi:
            if not self._started:
                print(ANSI_HIDE_CURSOR, end="", file=self._output_stream)
            print(ANSI_CLEAR_SCREEN, end="", file=self._output_stream)
            print("\n".join(lines), end="", file=self._output_stream)
            print(ANSI_CLEAR_TO_END, end="", file=self._output_stream, flush=True)
        else:
            print("\n".join(lines), file=self._output_stream, flush=True)
        self._started = True

    def _update_row_text(self, row: _ChannelRow, output: ChannelOutput) -> str:
        if output.tokens:
            stable_count = max(0, min(int(output.stable_token_count), len(output.tokens)))
            stable_tokens = tuple(output.tokens[:stable_count])
            tentative_tokens = tuple(output.tokens[stable_count:])
            if stable_tokens:
                row.committed_tokens = append_non_overlapping_tokens(row.committed_tokens, stable_tokens)
            row.tentative_tokens = tentative_tokens
            rendered = stable_tentative_display_text(row.committed_tokens, row.tentative_tokens)
            if rendered:
                row.text = rendered
            return row.text

        text = channel_output_display_text(output).strip()
        if text:
            # Backwards-compatible path for older callers/tests that only pass
            # pre-rendered text and no token stability metadata.
            row.text = text
        return row.text

    def _row_for(self, output: ChannelOutput) -> _ChannelRow:
        row = self._rows.get(output.channel_id)
        if row is None:
            row = _ChannelRow(channel_id=output.channel_id, carrier_hz=output.carrier_hz)
            self._rows[output.channel_id] = row
        else:
            row.carrier_hz = (row.carrier_hz * 0.8) + (output.carrier_hz * 0.2)
        return row

    def _drop_stale_rows(self) -> None:
        stale: list[int] = []
        for channel_id, row in self._rows.items():
            if row.text:
                reference_s = row.last_text_s or row.last_seen_s
                if self._current_time_s - reference_s > self._decoded_retention_s:
                    stale.append(channel_id)
            elif row.state == "dormant" and self._current_time_s - row.last_seen_s > self._inactive_retention_s:
                stale.append(channel_id)
        for channel_id in stale:
            del self._rows[channel_id]

    def _build_lines(self) -> list[str]:
        width = _dashboard_width()
        active_rows = [row for row in self._rows.values() if row.state != "dropped"]
        lines = [f"CW live monitor   t={_format_duration(self._current_time_s)}   channels={len(active_rows)}", ""]
        layer_width = 30
        text_width = max(10, width - 70)
        lines.append(_fit_columns([("ch", 4), ("freq", 10), ("state", 10), ("age", 7), ("layers", layer_width), ("text", text_width)]))
        lines.append(_fit_columns([("-" * 4, 4), ("-" * 10, 10), ("-" * 10, 10), ("-" * 7, 7), ("-" * layer_width, layer_width), ("-" * text_width, text_width)]))
        visible_rows = sorted(active_rows, key=lambda row: row.carrier_hz)[: self._max_rows]
        if not visible_rows:
            lines.append("listening... no confirmed CW channel yet")
        else:
            for row in visible_rows:
                lines.append(
                    _fit_columns(
                        [
                            (str(row.channel_id), 4),
                            (_format_carrier(row.carrier_hz), 10),
                            (row.state, 10),
                            (f"{self._row_age_s(row):4.1f}s", 7),
                            (_format_layers(row.layers), layer_width),
                            (_display_text(row), text_width),
                        ]
                    )
                )
            hidden_count = len(active_rows) - len(visible_rows)
            if hidden_count > 0:
                lines.append(f"... {hidden_count} more channel(s) hidden")
        return lines

    def _row_age_s(self, row: _ChannelRow) -> float:
        reference_s = row.last_text_s if row.text else row.last_seen_s
        return max(0.0, self._current_time_s - reference_s)


# Backwards compatible name for older callers.
HumanChannelOutputPrinter = HumanDashboardRenderer


def format_channel_output_dict(output: dict) -> str:
    parsed = _channel_output_from_dict(output)
    if parsed is None:
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    parts = [f"ch{parsed.channel_id}", f"{parsed.carrier_hz:7.1f} Hz", parsed.state]
    if parsed.text:
        parts.append(f"→ {parsed.text}")
    return " ".join(parts)


def iter_formatted_jsonl(lines: Iterable[str]) -> Iterable[str]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            yield line
            continue
        if isinstance(payload, dict) and _channel_output_from_dict(payload) is not None:
            yield format_channel_output_dict(payload)
        else:
            yield line


def print_channel_output_view(input_stream: TextIO, output_stream: TextIO) -> None:
    renderer = HumanDashboardRenderer(output_stream, use_ansi=_use_ansi_dashboard(output_stream))
    renderer.start()
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            renderer.close()
            print(line, file=output_stream, flush=True)
            continue
        if not isinstance(payload, dict):
            continue
        output = _channel_output_from_dict(payload)
        if output is not None:
            renderer.emit(output)
    renderer.close()


def print_channel_output_view_file(path: Path, output_stream: TextIO) -> None:
    with path.open("r", encoding="utf-8") as input_stream:
        print_channel_output_view(input_stream, output_stream)


def _channel_output_from_dict(payload: dict) -> ChannelOutput | None:
    return _parse_channel_output(payload)

def _use_ansi_dashboard(output_stream: TextIO) -> bool:
    if os.environ.get("CW_PLAIN_VIEW"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def _format_layers(layers: object | None) -> str:
    if layers is None:
        return "rx=- sig=- dec=- sel=-"
    audio_s = getattr(layers, "receiving_audio_s", 0.0) or 0.0
    tracks = getattr(layers, "signal_tracks", 0) or 0
    unknown = getattr(layers, "signal_best_unknown_ratio", None)
    answers = getattr(layers, "decoder_answers", 0) or 0
    support = getattr(layers, "selection_support", 0) or 0
    groups = getattr(layers, "selection_groups", 0) or 0
    if unknown is None:
        sig = f"sig{tracks}"
    else:
        sig = f"sig{tracks}/u{float(unknown):.2f}"
    return f"rx{float(audio_s):.1f}s {sig} dec{int(answers)} sel{int(support)}/{int(groups)}"


def _display_text(row: _ChannelRow) -> str:
    if row.text:
        return row.text
    return "—"


def _dashboard_width() -> int:
    explicit = os.environ.get("CW_VIEW_COLUMNS") or os.environ.get("COLUMNS")
    if explicit:
        try:
            return max(72, min(int(explicit), 320))
        except ValueError:
            pass
    return max(72, min(shutil.get_terminal_size((220, 30)).columns, 320))


def _format_carrier(carrier_hz: float) -> str:
    return f"{carrier_hz:7.1f} Hz"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    remain = seconds - (minutes * 60)
    if minutes <= 0:
        return f"{remain:04.1f}s"
    return f"{minutes:02d}:{remain:04.1f}"


def _fit_columns(columns: list[tuple[str, int]]) -> str:
    parts: list[str] = []
    last_index = len(columns) - 1
    for index, (value, width) in enumerate(columns):
        width = max(1, width)
        if index == last_index:
            text = _tail_ellipsize(str(value), width)
            parts.append(text)
        else:
            text = _ellipsize(str(value), width)
            parts.append(text.ljust(width))
    return " ".join(parts).rstrip()


def _ellipsize(value: str, width: int) -> str:
    value = value.replace("\n", " ")
    if len(value) <= width:
        return value
    if width <= 1:
        return "…"[:width]
    return value[: width - 1] + "…"


def _tail_ellipsize(value: str, width: int) -> str:
    value = value.replace("\n", " ")
    if len(value) <= width:
        return value
    if width <= 1:
        return "…"[:width]
    return "…" + value[-(width - 1) :]


__all__ = [
    "HumanChannelOutputPrinter",
    "HumanDashboardRenderer",
    "format_channel_output_dict",
    "iter_formatted_jsonl",
    "print_channel_output_view",
    "print_channel_output_view_file",
]
