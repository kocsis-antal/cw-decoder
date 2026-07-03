import json
from cw.stream_events import STREAM_EVENT_SCHEMA, stream_event_to_dict, stream_events_to_jsonl
from cw.stream_models import StreamEvent


def test_stream_event_to_dict_uses_stable_wire_schema() -> None:
    event = StreamEvent(
        time_s=1.23456,
        kind="SESSION_FINAL",
        channel_id=2,
        session_id=3,
        carrier_hz=700.98765,
        text="CQ DE YU7NKA",
        score=6.789,
        reason="end_of_stream",
    )

    assert stream_event_to_dict(event) == {
        "schema": STREAM_EVENT_SCHEMA,
        "time_s": 1.235,
        "type": "SESSION_FINAL",
        "channel_id": 2,
        "session_id": 3,
        "carrier_hz": 700.988,
        "text": "CQ DE YU7NKA",
        "score": 6.789,
        "reason": "end_of_stream",
    }


def test_stream_events_to_jsonl_round_trips_event_lines() -> None:
    events = [
        StreamEvent(0.0, "CHANNEL_STARTED", 1, None, 700.0),
        StreamEvent(0.5, "TEXT_COMMITTED", 1, 1, 700.0, text="CQ", score=5.5),
    ]

    lines = stream_events_to_jsonl(events).splitlines()

    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert [payload["type"] for payload in payloads] == ["CHANNEL_STARTED", "TEXT_COMMITTED"]
    assert payloads[0]["session_id"] is None
    assert payloads[0]["score"] is None
    assert payloads[1]["text"] == "CQ"


def test_stream_events_to_jsonl_includes_new_preview_and_activity_events() -> None:
    events = [
        StreamEvent(1.0, "SIGNAL_ACTIVE", 1, None, 700.0, reason="carrier_detected"),
        StreamEvent(1.5, "TEXT_PREVIEW", 1, 1, 700.0, text="CQ", score=12.0, reason="awaiting_stable_prefix"),
    ]

    payloads = [json.loads(line) for line in stream_events_to_jsonl(events).splitlines()]

    assert [payload["type"] for payload in payloads] == ["SIGNAL_ACTIVE", "TEXT_PREVIEW"]
    assert payloads[0]["reason"] == "carrier_detected"
    assert payloads[1]["text"] == "CQ"


def test_stream_event_to_dict_preserves_explicit_zero_score() -> None:
    event = StreamEvent(1.0, "TEXT_COMMITTED", 1, 1, 700.0, text="CQ", score=0.0)

    assert stream_event_to_dict(event)["score"] == 0.0
