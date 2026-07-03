from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO

from cw.stream_models import StreamEvent


ANSI_CLEAR_SCREEN = "\x1b[2J\x1b[H"
ANSI_CLEAR_TO_END = "\x1b[J"
ANSI_HIDE_CURSOR = "\x1b[?25l"
ANSI_SHOW_CURSOR = "\x1b[?25h"


@dataclass
class _CarrierRow:
    channel_id: int
    carrier_hz: float
    state: str = "signal"
    current_text: str = ""
    committed_text: str = ""
    finalized_texts: list[str] = field(default_factory=list)
    status_text: str = "waiting for text"
    preview: bool = False
    last_seen_s: float = 0.0
    last_text_s: float = 0.0
    score: float | None = None
    reason: str = ""


class HumanDashboardRenderer:
    """Live, stateful dashboard for people watching a CW stream.

    The decoder emits an append-only event stream.  That is perfect for JSONL,
    tests and future web/socket consumers, but it is hard to read when multiple
    carriers are active at the same time.  This renderer consumes the same
    events and maintains a small current-state table keyed internally by the
    decoder channel id, while showing only the radio-relevant information:
    carrier frequency, current state and decoded text.
    """

    def __init__(
        self,
        output_stream: TextIO,
        *,
        refresh_interval_s: float = 0.25,
        max_rows: int = 16,
        max_recent: int = 8,
        inactive_retention_s: float = 6.0,
        decoded_retention_s: float = 180.0,
        max_transcript_chars: int = 1600,
        use_ansi: bool | None = None,
    ) -> None:
        self._output_stream = output_stream
        self._refresh_interval_s = max(0.0, refresh_interval_s)
        self._max_rows = max(1, max_rows)
        _ = max_recent  # Kept as a harmless compatibility argument; recent list is no longer rendered.
        self._inactive_retention_s = max(0.0, inactive_retention_s)
        self._decoded_retention_s = max(self._inactive_retention_s, decoded_retention_s)
        self._max_transcript_chars = max(40, max_transcript_chars)
        self._use_ansi = _use_ansi_dashboard(output_stream) if use_ansi is None else use_ansi
        self._rows: dict[int, _CarrierRow] = {}
        self._current_time_s = 0.0
        self._last_render_s = -10**9
        self._started = False
        self._closed = False

    def start(self) -> None:
        self.render(force=True)

    def tick(self, time_s: float) -> None:
        self._current_time_s = max(self._current_time_s, float(time_s))
        self._drop_stale_inactive_rows()
        self.render()

    def emit(self, event: StreamEvent) -> None:
        self._current_time_s = max(self._current_time_s, event.time_s)
        if event.kind == "CHANNEL_STARTED":
            row = self._row_for(event)
            row.state = "signal"
            row.status_text = "waiting for text"
        elif event.kind == "SESSION_STARTED":
            row = self._row_for(event)
            row.state = "listening"
            row.current_text = ""
            row.committed_text = ""
            row.status_text = "waiting for text"
            row.preview = False
        elif event.kind == "SIGNAL_ACTIVE":
            row = self._row_for(event)
            if row.state not in {"decoding", "decoded"} or not self._transcript_text(row):
                row.state = "signal"
                row.status_text = _signal_status_text(event)
            row.reason = event.reason
        elif event.kind == "TEXT_PREVIEW":
            self._update_text(event, preview=True)
        elif event.kind == "TEXT_COMMITTED":
            self._update_text(event, preview=False)
        elif event.kind == "SESSION_FINAL":
            self._finalize_session(event)
        elif event.kind == "CHANNEL_DORMANT":
            self._mark_dormant(event)
        else:
            row = self._row_for(event)
            row.state = event.kind.lower().replace("_", " ")
            if event.text:
                row.current_text = event.text.strip()
        self._drop_stale_inactive_rows()
        self.render(force=event.kind in {"SESSION_FINAL", "CHANNEL_DORMANT"})

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

    def _row_for(self, event: StreamEvent) -> _CarrierRow:
        row = self._rows.get(event.channel_id)
        if row is None:
            row = _CarrierRow(channel_id=event.channel_id, carrier_hz=event.carrier_hz)
            self._rows[event.channel_id] = row
        else:
            # Keep the displayed carrier smooth but responsive.  The channel id
            # remains the internal key, so small live drift does not create a new
            # visual row.
            row.carrier_hz = (row.carrier_hz * 0.8) + (event.carrier_hz * 0.2)
        row.last_seen_s = max(row.last_seen_s, event.time_s)
        # A pure CHANNEL_DORMANT event often carries no decoded text, but it
        # still belongs to the already built transcript row.  Do not let it make
        # a finalized transcript look stale immediately; text freshness is kept
        # separately in last_text_s.
        if event.score is not None:
            row.score = event.score
        if event.reason:
            row.reason = event.reason
        return row

    def _update_text(self, event: StreamEvent, *, preview: bool) -> None:
        if not event.text:
            return
        row = self._row_for(event)
        row.state = "decoding"
        text = event.text.strip()
        row.current_text = text
        if not preview:
            row.committed_text = text
        row.preview = preview
        row.last_text_s = event.time_s
        self._trim_transcript(row)

    def _finalize_session(self, event: StreamEvent) -> None:
        row = self._row_for(event)
        text = event.text.strip() if event.text else row.current_text.strip()
        if not text:
            text = "<no decoded text>"
        row.state = "decoded"
        row.preview = False
        row.last_text_s = event.time_s
        if text and text != "<no decoded text>":
            self._append_finalized_text(row, text)
            row.current_text = ""
            row.committed_text = ""
        else:
            row.status_text = text

    def _mark_dormant(self, event: StreamEvent) -> None:
        row = self._row_for(event)
        if self._transcript_text(row):
            # For the operator this carrier is still interesting: it contains
            # decoded text.  Keep it visible in the upper table for a while.
            row.state = "decoded"
        else:
            row.state = "inactive"
            row.status_text = "inactive"
        row.preview = False

    def _drop_stale_inactive_rows(self) -> None:
        stale: list[int] = []
        for channel_id, row in self._rows.items():
            if self._transcript_text(row):
                # Transcript rows are the live operator's memory of a frequency.
                # Retain them much longer than bare signal-only rows so the top
                # table keeps acting as the operator's short-term memory.
                reference_s = row.last_text_s or row.last_seen_s
                if self._current_time_s - reference_s > self._decoded_retention_s:
                    stale.append(channel_id)
            elif row.state == "inactive" and self._current_time_s - row.last_seen_s > self._inactive_retention_s:
                stale.append(channel_id)
        for channel_id in stale:
            del self._rows[channel_id]

    def _append_finalized_text(self, row: _CarrierRow, text: str) -> None:
        if row.finalized_texts and row.finalized_texts[-1] == text:
            return
        row.finalized_texts.append(text)
        self._trim_transcript(row)

    def _display_text(self, row: _CarrierRow) -> str:
        text = self._transcript_text(row)
        return text if text else row.status_text

    def _transcript_text(self, row: _CarrierRow) -> str:
        parts = [*row.finalized_texts]
        current = self._current_display_text(row)
        if current:
            parts.append(current)
        return _join_transcript_segments(part for part in parts if part)

    def _current_display_text(self, row: _CarrierRow) -> str:
        current = row.current_text.strip()
        if not current:
            return ""
        if not row.preview:
            return current

        committed = row.committed_text.strip()
        if committed and current.startswith(committed):
            suffix = current[len(committed) :].strip()
            if suffix:
                return f"{committed} [{suffix}]"
            return committed
        return f"[{current}]"

    def _trim_transcript(self, row: _CarrierRow) -> None:
        # Keep enough history for a live operator, but bound memory and terminal
        # width pressure for long monitoring sessions.  Prefer dropping old
        # finalized text; never truncate the currently forming session unless it
        # alone is longer than the configured cap.
        while row.finalized_texts and len(self._transcript_text(row)) > self._max_transcript_chars:
            row.finalized_texts.pop(0)
        if len(self._transcript_text(row)) > self._max_transcript_chars and row.current_text:
            row.current_text = _left_ellipsize(row.current_text, self._max_transcript_chars)

    def _build_lines(self) -> list[str]:
        width = _dashboard_width()
        lines = [f"CW live monitor   t={_format_duration(self._current_time_s)}   carriers={len(self._rows)}", ""]
        lines.append(_fit_columns([("freq", 10), ("state", 10), ("last", 7), ("text", width - 31)]))
        lines.append(_fit_columns([("-" * 10, 10), ("-" * 10, 10), ("-" * 7, 7), ("-" * max(10, width - 31), width - 31)]))
        visible_rows = sorted(self._rows.values(), key=lambda row: row.carrier_hz)[: self._max_rows]
        if not visible_rows:
            lines.append("listening... no confirmed CW carrier yet")
        else:
            for row in visible_rows:
                age_s = self._row_age_s(row)
                text = self._display_text(row)
                lines.append(
                    _fit_columns(
                        [
                            (_format_carrier(row.carrier_hz), 10),
                            (row.state, 10),
                            (f"{age_s:4.1f}s", 7),
                            (text, width - 31),
                        ]
                    )
                )
            hidden_count = len(self._rows) - len(visible_rows)
            if hidden_count > 0:
                lines.append(f"... {hidden_count} more carrier(s) hidden")

        return lines

    def _row_age_s(self, row: _CarrierRow) -> float:
        # For decoded text the useful freshness is when the text last changed,
        # not when a late dormant/no-text event touched the carrier.  For signal
        # rows without decoded text, last_seen_s still describes liveness.
        reference_s = row.last_text_s if self._transcript_text(row) else row.last_seen_s
        return max(0.0, self._current_time_s - reference_s)


# Backwards compatible name for the previous compact human printer.  The new
# default is a dashboard, not an append-only channel/session log.
HumanStreamEventPrinter = HumanDashboardRenderer


def format_event_dict(event: dict) -> str:
    kind = str(event.get("type") or event.get("kind") or "EVENT")
    time_s = float(event.get("time_s") or 0.0)
    channel_id = event.get("channel_id")
    session_id = event.get("session_id")
    carrier_hz = float(event.get("carrier_hz") or 0.0)
    text = str(event.get("text") or "")
    score = event.get("score")
    reason = str(event.get("reason") or "")

    label = _kind_label(kind)
    parts = [f"[{time_s:7.2f}s]", f"ch{channel_id}", f"{carrier_hz:7.1f} Hz", label]
    if session_id is not None:
        parts.insert(3, f"s{session_id}")
    if score is not None:
        parts.append(f"score={float(score):.1f}")
    if reason:
        parts.append(f"({reason})")
    if text:
        parts.append(f"→ {text}")
    return " ".join(parts)


def iter_formatted_jsonl(lines: Iterable[str]) -> Iterable[str]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Keep stderr/status lines visible when a mixed log is piped in.
            yield line
            continue
        if isinstance(event, dict) and ("type" in event or "kind" in event):
            yield format_event_dict(event)
        else:
            yield line


def print_event_view(input_stream: TextIO, output_stream: TextIO) -> None:
    renderer = HumanDashboardRenderer(output_stream, use_ansi=_use_ansi_dashboard(output_stream))
    renderer.start()
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # Keep non-event diagnostic lines visible below the dashboard.
            renderer.close()
            print(line, file=output_stream, flush=True)
            continue
        event = _stream_event_from_dict(payload)
        if event is None:
            continue
        renderer.emit(event)
    renderer.close()


def print_event_view_file(path: Path, output_stream: TextIO) -> None:
    with path.open("r", encoding="utf-8") as input_stream:
        print_event_view(input_stream, output_stream)


def _stream_event_from_dict(payload: dict) -> StreamEvent | None:
    kind = payload.get("type") or payload.get("kind")
    if kind is None:
        return None
    return StreamEvent(
        time_s=float(payload.get("time_s") or 0.0),
        kind=str(kind),
        channel_id=int(payload.get("channel_id") or 0),
        session_id=None if payload.get("session_id") is None else int(payload.get("session_id")),
        carrier_hz=float(payload.get("carrier_hz") or 0.0),
        text=str(payload.get("text") or ""),
        score=None if payload.get("score") is None else float(payload.get("score")),
        reason=str(payload.get("reason") or ""),
    )


def _use_ansi_dashboard(output_stream: TextIO) -> bool:
    # Docker compose run -T often reports non-TTY inside the container even when
    # bytes still go to an interactive terminal on the host.  Keep ANSI enabled
    # by default for the live viewer, but provide a plain escape hatch for logs.
    if os.environ.get("CW_PLAIN_VIEW"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def _kind_label(kind: str) -> str:
    return {
        "CHANNEL_STARTED": "channel",
        "CHANNEL_DORMANT": "dormant",
        "SESSION_STARTED": "start",
        "SIGNAL_ACTIVE": "active",
        "TEXT_PREVIEW": "preview",
        "TEXT_COMMITTED": "text",
        "SESSION_FINAL": "final",
    }.get(kind, kind.lower())


def _signal_status_text(event: StreamEvent) -> str:
    if event.reason == "awaiting_decodable_text":
        return "waiting for text"
    if event.reason == "carrier_detected":
        return "carrier detected"
    if event.reason:
        return event.reason.replace("_", " ")
    return "signal detected"


def _join_transcript_segments(parts: Iterable[str]) -> str:
    # Each finalized segment represents a decoder session boundary, normally a
    # longer silence gap, not a transmitted character.  Use visual whitespace
    # instead of /, ?, *, etc. so the monitor does not invent CW punctuation.
    return "   ".join(part for part in parts if part)


def _dashboard_width() -> int:
    explicit = os.environ.get("CW_VIEW_COLUMNS") or os.environ.get("COLUMNS")
    if explicit:
        try:
            return max(72, min(int(explicit), 320))
        except ValueError:
            pass
    # When running through docker compose run -T stdout is often not a TTY, so
    # Python cannot see the host terminal size. Prefer a generous fallback; if
    # the real terminal is narrower it will wrap, but wide terminals no longer
    # lose half the available text column.
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
    for index, (value, width) in enumerate(columns):
        width = max(1, width)
        text = _ellipsize(str(value), width)
        if index == 0:
            parts.append(text.ljust(width))
        elif index == len(columns) - 1:
            parts.append(text)
        else:
            parts.append(text.ljust(width))
    return " ".join(parts).rstrip()


def _ellipsize(value: str, width: int) -> str:
    value = value.replace("\n", " ")
    if len(value) <= width:
        return value
    if width <= 1:
        return "…"[:width]
    return value[: width - 1] + "…"


def _left_ellipsize(value: str, width: int) -> str:
    value = value.replace("\n", " ")
    if len(value) <= width:
        return value
    if width <= 1:
        return "…"[:width]
    return "…" + value[-(width - 1) :]
