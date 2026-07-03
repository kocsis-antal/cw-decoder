from cw.qso_generator import ContestQsoConfig, build_contest_qso_sources


def test_build_contest_qso_sources_creates_sequential_turns() -> None:
    sources = build_contest_qso_sources(
        ContestQsoConfig(caller_call="YU7NKA", responder_call="YT7MK", seed=123)
    )

    assert [source.text for source in sources] == [
        "CQ TEST YU7NKA",
        "YU7NKA YT7MK",
        "YT7MK 599 001",
        "TU 599 002",
        "TU",
    ]
    assert [source.start_s for source in sources] == sorted(source.start_s for source in sources)
    assert sources[0].config.tone_hz == 700.0
    assert sources[1].config.tone_hz == 706.0
    assert sources[0].config.wpm != sources[1].config.wpm
