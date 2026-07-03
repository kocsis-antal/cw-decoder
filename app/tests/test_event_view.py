from cw.event_view import format_event_dict, iter_formatted_jsonl


def test_format_event_dict_is_human_readable() -> None:
    line = format_event_dict(
        {
            "time_s": 12.345,
            "type": "TEXT_COMMITTED",
            "channel_id": 2,
            "session_id": 1,
            "carrier_hz": 975.0,
            "score": 4.2,
            "text": "CQ DE TEST",
            "reason": "",
        }
    )

    assert "ch2" in line
    assert "975.0 Hz" in line
    assert "text" in line
    assert "CQ DE TEST" in line


def test_iter_formatted_jsonl_keeps_non_json_lines() -> None:
    lines = list(iter_formatted_jsonl(['{"type":"SIGNAL_ACTIVE","time_s":1,"channel_id":1,"carrier_hz":700}\n', "live stats\n"]))

    assert "active" in lines[0]
    assert lines[1] == "live stats"


def test_view_events_cli_reads_stdin(capsys, monkeypatch) -> None:
    import io
    import sys

    from cw.cli import main

    monkeypatch.setattr(sys, "argv", ["cw", "view-events"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"TEXT_COMMITTED","time_s":1,"channel_id":1,"session_id":1,"carrier_hz":700,"text":"CQ"}\n'))

    main()

    assert "CQ" in capsys.readouterr().out


def test_human_dashboard_renderer_shows_frequency_without_internal_ids() -> None:
    import io

    from cw.event_view import HumanDashboardRenderer
    from cw.stream_models import StreamEvent

    output = io.StringIO()
    renderer = HumanDashboardRenderer(output, use_ansi=False, refresh_interval_s=0.0)

    for event in [
        StreamEvent(1.0, "CHANNEL_STARTED", 8, None, 1444.1),
        StreamEvent(1.2, "SIGNAL_ACTIVE", 8, None, 1444.1, reason="awaiting_decodable_text"),
        StreamEvent(1.5, "TEXT_COMMITTED", 8, 1, 1443.0, text="ROL", score=2.5),
        StreamEvent(2.0, "TEXT_PREVIEW", 8, 1, 1442.0, text="ROLF 7I", score=2.5),
        StreamEvent(3.0, "SESSION_FINAL", 8, 1, 1440.0, text="ROLF 73", score=2.5, reason="silence_gap"),
    ]:
        renderer.emit(event)
    renderer.close()

    rendered = output.getvalue()

    assert "144" in rendered
    assert "ROL [F 7I]" in rendered
    assert "ROLF 73" in rendered
    assert "recent decoded" not in rendered
    assert "ch8" not in rendered
    assert "s1" not in rendered
    assert "kind=" not in rendered


def test_human_dashboard_keeps_interleaved_carriers_on_separate_rows() -> None:
    import io

    from cw.event_view import HumanDashboardRenderer
    from cw.stream_models import StreamEvent

    output = io.StringIO()
    renderer = HumanDashboardRenderer(output, use_ansi=False, refresh_interval_s=0.0)

    for event in [
        StreamEvent(1.0, "CHANNEL_STARTED", 3, None, 1037.5),
        StreamEvent(1.1, "TEXT_COMMITTED", 3, 1, 1037.5, text="R HIETLII"),
        StreamEvent(1.2, "CHANNEL_STARTED", 4, None, 2062.5),
        StreamEvent(1.3, "TEXT_COMMITTED", 4, 1, 2062.5, text="CQCQ DE EV8"),
        StreamEvent(1.4, "SESSION_FINAL", 3, 1, 1037.5, text="R HIETLII", reason="channel_inactive"),
    ]:
        renderer.emit(event)
    renderer.close()

    rendered = output.getvalue()

    assert "1037" in rendered and "R HIETLII" in rendered
    assert "2062" in rendered and "CQCQ DE EV8" in rendered
    assert "ch3" not in rendered
    assert "ch4" not in rendered


def test_human_dashboard_shows_channel_transcript_across_sessions() -> None:
    import io

    from cw.event_view import HumanDashboardRenderer
    from cw.stream_models import StreamEvent

    output = io.StringIO()
    renderer = HumanDashboardRenderer(output, use_ansi=False, refresh_interval_s=0.0)

    for event in [
        StreamEvent(1.0, "CHANNEL_STARTED", 3, None, 975.0),
        StreamEvent(1.1, "SESSION_STARTED", 3, 1, 975.0),
        StreamEvent(1.5, "TEXT_COMMITTED", 3, 1, 975.0, text="CQ CQ DE TEST"),
        StreamEvent(2.0, "SESSION_FINAL", 3, 1, 975.0, text="CQ CQ DE TEST", reason="silence_gap"),
        StreamEvent(2.5, "SESSION_STARTED", 3, 2, 975.0),
        StreamEvent(3.0, "TEXT_PREVIEW", 3, 2, 975.0, text="PSE K", reason="awaiting_stable_prefix"),
    ]:
        renderer.emit(event)
    renderer.close()

    rendered = output.getvalue()

    assert "CQ CQ DE TEST   [PSE K]" in rendered
    assert "ch3" not in rendered
    assert "s1" not in rendered
    assert "s2" not in rendered


def test_human_dashboard_bounds_very_long_transcript() -> None:
    import io

    from cw.event_view import HumanDashboardRenderer
    from cw.stream_models import StreamEvent

    output = io.StringIO()
    renderer = HumanDashboardRenderer(output, use_ansi=False, refresh_interval_s=0.0, max_transcript_chars=50)

    for event in [
        StreamEvent(1.0, "CHANNEL_STARTED", 4, None, 1000.0),
        StreamEvent(1.2, "TEXT_COMMITTED", 4, 1, 1000.0, text="A" * 80),
    ]:
        renderer.emit(event)
    renderer.close()

    rendered = output.getvalue()

    assert "…" in rendered
    assert "A" * 40 in rendered


def test_human_dashboard_retains_decoded_transcript_after_dormant() -> None:
    import io

    from cw.event_view import HumanDashboardRenderer
    from cw.stream_models import StreamEvent

    output = io.StringIO()
    renderer = HumanDashboardRenderer(
        output,
        use_ansi=False,
        refresh_interval_s=0.0,
        inactive_retention_s=1.0,
        decoded_retention_s=30.0,
    )

    for event in [
        StreamEvent(1.0, "CHANNEL_STARTED", 1, None, 625.0),
        StreamEvent(2.0, "SESSION_FINAL", 1, 1, 625.0, text="CO EM0WWA", reason="silence_gap"),
        StreamEvent(3.0, "CHANNEL_DORMANT", 1, None, 625.0, reason="channel_inactive"),
    ]:
        renderer.emit(event)
    renderer.tick(10.0)
    renderer.close()

    rendered = output.getvalue()

    assert "625.0 Hz" in rendered
    assert "CO EM0WWA" in rendered
    assert "decoded" in rendered


def test_human_dashboard_drops_signal_only_rows_quickly() -> None:
    import io

    from cw.event_view import HumanDashboardRenderer
    from cw.stream_models import StreamEvent

    output = io.StringIO()
    renderer = HumanDashboardRenderer(output, use_ansi=False, refresh_interval_s=0.0, inactive_retention_s=1.0)

    for event in [
        StreamEvent(1.0, "CHANNEL_STARTED", 1, None, 625.0),
        StreamEvent(2.0, "CHANNEL_DORMANT", 1, None, 625.0, reason="channel_inactive"),
    ]:
        renderer.emit(event)
    renderer.tick(4.0)
    renderer.close()

    rendered = output.getvalue()

    assert "625.0 Hz" not in rendered.split("CW live monitor")[-1]
