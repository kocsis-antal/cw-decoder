from __future__ import annotations

import numpy as np

from cw.app.config import ProcessingConfig
from cw.app.transcript import ChannelTranscript, apply_transcript_update, safe_commit_prefix_len
from cw.app.channel_output import ChannelOutput, channel_output_display_text
from cw.decoder.tokens import char_token, gap_token, tokens_to_text
from cw.receiving.models import ChannelSignal, ChannelState
from cw.selection.models import ChannelWinner


def _tokens(text: str, *, start: float = 0.0, step: float = 0.2):
    out = []
    t = start
    pending_gap = False
    for ch in text:
        if ch == " ":
            pending_gap = True
            continue
        if pending_gap and out:
            out.append(gap_token("word_gap", start_s=t, end_s=t + step))
            t += step
            pending_gap = False
        out.append(char_token(ch, start_s=t, end_s=t + step))
        t += step
    return tuple(out)


def _winner(text: str, *, channel_id: int = 1, carrier_hz: float = 700.0, time_s: float = 1.0, tokens=None):
    return ChannelWinner(channel_id, carrier_hz, text, "selected", time_s, tokens=tokens if tokens is not None else _tokens(text))


def _active_channel(*, start_s: float = 10.0, end_s: float = 13.9):
    return ChannelSignal(1, 700.0, start_s, end_s, np.ones(100, dtype=np.float32), 8000, ChannelState.ACTIVE)


def test_channel_transcript_keeps_trailing_active_tokens_tentative_and_returns_trim() -> None:
    channel = _active_channel(start_s=10.0, end_s=13.9)
    winner = _winner("CQ DE", time_s=13.9, tokens=_tokens("CQ DE", start=0.0, step=0.4))
    transcript = ChannelTranscript()

    update = apply_transcript_update(transcript, channel, winner, ProcessingConfig(commit_hold_chars=2))

    assert tokens_to_text(transcript.committed_tokens) == "CQ"
    assert tokens_to_text(transcript.tentative_tokens) == "DE"
    assert channel_output_display_text(ChannelOutput(1, 700.0, "active", tokens=update.winner.tokens, stable_token_count=update.winner.stable_token_count)) == "CQ [DE]"
    assert update.trim_before_s is not None


def test_channel_transcript_uses_token_time_not_repeated_string_prefix() -> None:
    channel = _active_channel(start_s=0.0, end_s=5.0)
    transcript = ChannelTranscript()
    config = ProcessingConfig(commit_hold_chars=3, commit_fallback_after_chars=6)

    first = apply_transcript_update(transcript, channel, _winner("55N= MY NAME", time_s=3.0, tokens=_tokens("55N= MY NAME", step=0.2)), config)
    # The useful phrase has a safe gap boundary, so it becomes stable even if the next tail starts elsewhere.
    assert "MY" in tokens_to_text(first.winner.tokens)
    assert transcript.committed_tokens

    second = apply_transcript_update(transcript, channel, _winner("MY NAME IS MARK", time_s=4.0, tokens=_tokens("MY NAME IS MARK", start=1.0, step=0.2)), config)
    rendered = channel_output_display_text(ChannelOutput(1, 700.0, "active", tokens=second.winner.tokens, stable_token_count=second.winner.stable_token_count))
    assert "MY" in rendered
    assert "MARK" in rendered


def test_safe_commit_uses_latest_non_empty_word_boundary_and_ignores_leading_separator() -> None:
    assert safe_commit_prefix_len((gap_token("word_gap"),) + _tokens("CEUCQCD"), 8, fallback_after_chars=6) == 4
    assert safe_commit_prefix_len(_tokens("CQ DE F"), len(_tokens("CQ DE")), fallback_after_chars=6) == 3
    assert safe_commit_prefix_len(_tokens("CQCQDE"), 6, fallback_after_chars=6) == 3
    assert safe_commit_prefix_len((gap_token("word_gap"), gap_token("word_gap")), 2, fallback_after_chars=6) == 0


def test_channel_transcript_does_not_replace_committed_text_with_single_glitch() -> None:
    channel = _active_channel(start_s=0.0, end_s=20.0)
    transcript = ChannelTranscript()
    config = ProcessingConfig(commit_hold_chars=3, commit_fallback_after_chars=6)

    first = apply_transcript_update(transcript, channel, _winner("DRIR2NZRK", time_s=20.0), config)
    assert tokens_to_text(transcript.committed_tokens)

    second = apply_transcript_update(transcript, channel, _winner("D", time_s=20.5), config)
    text = tokens_to_text(second.winner.tokens)
    assert len(text) > 1
    assert text.startswith(tokens_to_text(transcript.committed_tokens))


def test_channel_transcript_commits_trailing_session_gap_between_bursts() -> None:
    transcript = ChannelTranscript()
    config = ProcessingConfig(commit_hold_chars=0, commit_audio_context_chars=0)
    channel = _active_channel(start_s=0.0, end_s=4.0)
    first_tokens = (
        char_token("C", start_s=0.0, end_s=0.3),
        char_token("Q", start_s=0.4, end_s=0.8),
        gap_token("session_gap", start_s=0.8, end_s=3.0),
    )

    first = apply_transcript_update(transcript, channel, _winner("", tokens=first_tokens), config)

    assert transcript.committed_tokens[-1].kind == "session_gap"
    assert first.winner.tokens[-1].kind == "session_gap"
    assert first.winner.stable_token_count == len(first.winner.tokens)

    second_tokens = (
        char_token("D", start_s=3.0, end_s=3.3),
        char_token("E", start_s=3.4, end_s=3.7),
    )
    second = apply_transcript_update(transcript, channel, _winner("", tokens=second_tokens), config)

    rendered = tokens_to_text(second.winner.tokens)
    assert rendered == "CQ   DE"


def test_channel_transcript_does_not_duplicate_overlap_content_after_gap_commit() -> None:
    transcript = ChannelTranscript()
    config = ProcessingConfig(commit_hold_chars=0, commit_audio_context_chars=2)
    channel = _active_channel(start_s=0.0, end_s=4.0)
    first_tokens = (
        char_token("C", start_s=0.0, end_s=0.3),
        char_token("Q", start_s=0.4, end_s=0.8),
        gap_token("session_gap", start_s=0.8, end_s=1.5),
    )
    apply_transcript_update(transcript, channel, _winner("", tokens=first_tokens), config)

    overlap_tokens = (
        char_token("Q", start_s=0.4, end_s=0.805),
        gap_token("session_gap", start_s=0.805, end_s=2.0),
        char_token("D", start_s=2.0, end_s=2.3),
    )
    second = apply_transcript_update(transcript, channel, _winner("", tokens=overlap_tokens), config)

    values = [(token.kind, token.value) for token in second.winner.tokens]
    assert values.count(("char", "Q")) == 1
    assert tokens_to_text(second.winner.tokens).startswith("CQ   D")


def test_session_gap_trim_is_hard_cut_without_audio_context() -> None:
    from cw.app.transcript import trim_time_from_committed_tokens

    tokens = (
        char_token("A", start_s=0.0, end_s=0.2),
        char_token("B", start_s=0.3, end_s=0.5),
        gap_token("session_gap", start_s=0.5, end_s=1.8),
    )

    assert trim_time_from_committed_tokens(tokens, audio_context_chars=2) == 1.8


def test_audio_context_never_crosses_last_session_gap() -> None:
    from cw.app.transcript import trim_time_from_committed_tokens

    tokens = (
        char_token("A", start_s=0.0, end_s=0.2),
        gap_token("session_gap", start_s=0.2, end_s=1.2),
        char_token("B", start_s=1.2, end_s=1.5),
    )

    assert trim_time_from_committed_tokens(tokens, audio_context_chars=5) == 1.2
