import json
import sys
from pathlib import Path

from cw.multi_generator import parse_source_spec, write_multi_sample
from cw.stream_events import STREAM_EVENT_SCHEMA, stream_event_to_dict, stream_events_to_jsonl
from cw.stream_models import StreamEvent, StreamingConfig
from cw.streaming import simulate_stream_from_wav


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


def test_stream_sim_events_are_serializable_as_jsonl(tmp_path: Path) -> None:
    wav_path = tmp_path / "two.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ DE YU7NKA", index=0, sample_rate=8000),
        parse_source_spec(
            "id=two;freq=1000;preset=straight;text=CQ DE YT7MK;start=0.2;amplitude=0.45",
            index=1,
            sample_rate=8000,
        ),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    result = simulate_stream_from_wav(wav_path, StreamingConfig(max_tracks=3, emit_interval_s=0.5))
    payloads = [json.loads(line) for line in stream_events_to_jsonl(result.events).splitlines()]

    assert payloads
    assert {payload["type"] for payload in payloads} >= {
        "CHANNEL_STARTED",
        "SESSION_STARTED",
        "TEXT_COMMITTED",
        "SESSION_FINAL",
        "CHANNEL_DORMANT",
    }
    assert {payload["text"] for payload in payloads if payload["type"] == "SESSION_FINAL"} == {
        "CQ DE YU7NKA",
        "CQ DE YT7MK",
    }


def test_stream_sim_cli_can_emit_json_events(tmp_path: Path, capsys, monkeypatch) -> None:
    from cw.cli import main

    wav_path = tmp_path / "one.wav"
    sources = [
        parse_source_spec("id=one;freq=700;preset=field;text=CQ DE YU7NKA", index=0, sample_rate=8000),
    ]
    write_multi_sample(sources, wav_path, sample_rate=8000)

    monkeypatch.setattr(sys, "argv", ["cw", "stream-sim", str(wav_path), "--json-events"])
    main()

    output = capsys.readouterr().out.strip()
    payloads = [json.loads(line) for line in output.splitlines()]
    assert payloads
    assert "duration_s=" not in output
    assert all(payload["schema"] == STREAM_EVENT_SCHEMA for payload in payloads)
    assert any(payload["type"] == "SESSION_FINAL" and payload["text"] == "CQ DE YU7NKA" for payload in payloads)


def test_stream_event_to_dict_preserves_explicit_zero_score() -> None:
    event = StreamEvent(1.0, "TEXT_COMMITTED", 1, 1, 700.0, text="CQ", score=0.0)

    assert stream_event_to_dict(event)["score"] == 0.0
