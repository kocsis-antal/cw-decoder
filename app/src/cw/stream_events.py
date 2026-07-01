from __future__ import annotations

import json
from typing import Any, Iterable

from cw.stream_models import StreamEvent, StreamSimulationResult

STREAM_EVENT_SCHEMA = "cw.stream.event.v1"


def stream_event_to_dict(event: StreamEvent) -> dict[str, Any]:
    """Convert one stream lifecycle event to a stable JSON-friendly payload.

    The external field name is ``type`` to match event-stream conventions.  The
    internal dataclass keeps ``kind`` because it is a Python object, not the wire
    format.  Keep all top-level keys present so downstream consumers do not need
    to special-case CHANNEL_STARTED vs SESSION_FINAL records.
    """

    return {
        "schema": STREAM_EVENT_SCHEMA,
        "time_s": _rounded_float(event.time_s),
        "type": event.kind,
        "channel_id": int(event.channel_id),
        "session_id": event.session_id if event.session_id is None else int(event.session_id),
        "carrier_hz": _rounded_float(event.carrier_hz),
        "text": event.text,
        "score": None if event.score == 0.0 else _rounded_float(event.score),
        "reason": event.reason,
    }


def stream_event_to_json(event: StreamEvent) -> str:
    return json.dumps(stream_event_to_dict(event), ensure_ascii=False, sort_keys=True)


def stream_events_to_jsonl(events: Iterable[StreamEvent]) -> str:
    return "\n".join(stream_event_to_json(event) for event in events)


def stream_result_events_to_jsonl(result: StreamSimulationResult) -> str:
    return stream_events_to_jsonl(result.events)


def _rounded_float(value: float) -> float:
    return round(float(value), 3)
