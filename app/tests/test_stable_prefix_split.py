from __future__ import annotations

import numpy as np

from cw.app.channel_output import ChannelOutput, channel_output_display_text
from cw.app.config import ProcessingConfig
from cw.app.transcript import (
    TranscriptConfig,
    safe_commit_prefix_len,
    split_transcript_tokens,
    split_winner_tokens,
    trim_time_from_stable_tokens,
)
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


def test_stable_prefix_split_keeps_trailing_active_tokens_tentative_and_returns_trim() -> None:
    split = split_transcript_tokens(
        _tokens("CQ DE", start=10.0, step=0.4),
        active=True,
        config=TranscriptConfig(hold_chars=2),
    )

    assert tokens_to_text(split.stable_tokens) == "CQ"
    assert tokens_to_text(split.carried_tokens) == "DE"
    assert channel_output_display_text(ChannelOutput(1, 700.0, "active", tokens=split.tokens, stable_token_count=split.stable_token_count)) == "CQ [DE]"
    assert split.trim_before_s is not None


def test_split_winner_outputs_absolute_token_times_and_receiver_trim_point() -> None:
    channel = _active_channel(start_s=10.0, end_s=13.9)
    winner = _winner("CQ DE", time_s=13.9, tokens=_tokens("CQ DE", start=0.0, step=0.4))

    split_winner, trim_before_s = split_winner_tokens(channel, winner, ProcessingConfig(stable_prefix_hold_chars=2))

    assert tokens_to_text(split_winner.tokens[: split_winner.stable_token_count]) == "CQ"
    assert split_winner.tokens[0].start_s == 10.0
    assert trim_before_s is not None


def test_safe_commit_uses_latest_non_empty_word_boundary_and_ignores_leading_separator() -> None:
    assert safe_commit_prefix_len((gap_token("word_gap"),) + _tokens("CEUCQCD"), 8, fallback_after_chars=6) == 4
    assert safe_commit_prefix_len(_tokens("CQ DE F"), len(_tokens("CQ DE")), fallback_after_chars=6) == 3
    assert safe_commit_prefix_len(_tokens("CQCQDE"), 6, fallback_after_chars=6) == 3
    assert safe_commit_prefix_len((gap_token("word_gap"), gap_token("word_gap")), 2, fallback_after_chars=6) == 0


def test_stable_prefix_can_include_trailing_session_gap() -> None:
    tokens = (
        char_token("C", start_s=0.0, end_s=0.3),
        char_token("Q", start_s=0.4, end_s=0.8),
        gap_token("session_gap", start_s=0.8, end_s=3.0),
    )

    split = split_transcript_tokens(tokens, active=True, config=TranscriptConfig(hold_chars=0, audio_context_chars=0))

    assert split.stable_tokens[-1].kind == "session_gap"
    assert split.stable_token_count == len(split.tokens)
    assert split.trim_before_s == 3.0


def test_session_gap_trim_is_hard_cut_without_audio_context() -> None:
    tokens = (
        char_token("A", start_s=0.0, end_s=0.2),
        char_token("B", start_s=0.3, end_s=0.5),
        gap_token("session_gap", start_s=0.5, end_s=1.8),
    )

    assert trim_time_from_stable_tokens(tokens, audio_context_chars=2) == 1.8


def test_audio_context_never_crosses_last_session_gap() -> None:
    tokens = (
        char_token("A", start_s=0.0, end_s=0.2),
        gap_token("session_gap", start_s=0.2, end_s=1.2),
        char_token("B", start_s=1.2, end_s=1.5),
    )

    assert trim_time_from_stable_tokens(tokens, audio_context_chars=5) == 1.2
