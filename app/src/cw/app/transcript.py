from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from cw.decoder.tokens import (
    CONTENT_KINDS,
    GAP_KINDS,
    DecodeToken,
    TOKEN_UNKNOWN,
    gap_token,
    token_signature,
    tokens_to_text,
)
from cw.receiving.models import ChannelSignal, ChannelState
from cw.selection.models import ChannelWinner


@dataclass
class ChannelTranscript:
    """Persistent decoded-token state owned by one tracked channel.

    The transcript stores stable tokens, the current mutable tail and the audio
    trim point for the channel.  It never uses rendered text as state: gaps and
    characters stay separate tokens with timing.
    """

    committed_tokens: tuple[DecodeToken, ...] = ()
    tentative_tokens: tuple[DecodeToken, ...] = ()
    audio_trim_before_s: float = 0.0
    last_channel_state: ChannelState | None = None
    pending_separator: DecodeToken | None = None

    def visible_tokens(self) -> tuple[DecodeToken, ...]:
        return normalize_gap_tokens(self.committed_tokens + self.tentative_tokens)

    @property
    def stable_token_count(self) -> int:
        return len(self.committed_tokens)


@dataclass(frozen=True)
class TranscriptUpdate:
    winner: ChannelWinner
    trim_before_s: float | None = None


@dataclass(frozen=True)
class TranscriptConfig:
    enabled: bool = True
    hold_chars: int = 3
    fallback_after_chars: int = 6
    commit_unresolved: bool = False
    audio_context_chars: int = 2

    @classmethod
    def from_config(cls, config) -> "TranscriptConfig":
        return cls(
            enabled=bool(getattr(config, "incremental_commit_enabled", True)),
            hold_chars=max(0, int(getattr(config, "commit_hold_chars", 3))),
            fallback_after_chars=max(0, int(getattr(config, "commit_fallback_after_chars", 6))),
            commit_unresolved=bool(getattr(config, "commit_unresolved", False)),
            audio_context_chars=max(0, int(getattr(config, "commit_audio_context_chars", 2))),
        )


def apply_transcript_update(
    transcript: ChannelTranscript,
    channel: ChannelSignal,
    winner: ChannelWinner,
    config,
) -> TranscriptUpdate:
    """Commit the stable part of one selected decode into the channel state.

    Stability is decided from token timing and safe token boundaries, not from a
    repeated rendered string prefix.  The selected winner is still the current
    tail source, but already committed token-time ranges are ignored.
    """

    settings = TranscriptConfig.from_config(config)
    raw_tokens = absolute_tokens(winner.tokens, offset_s=channel.start_s)
    if not settings.enabled:
        return TranscriptUpdate(_winner_from_tokens(winner, raw_tokens, stable_count=0))

    tail = uncommitted_tail(transcript, raw_tokens)
    tail = add_pending_separator_if_needed(transcript, channel, tail)

    if channel.state is not ChannelState.ACTIVE and transcript.tentative_tokens:
        tail = preserve_non_degraded_tentative(transcript.tentative_tokens, tail)

    commit_len = commit_prefix_len(
        tail,
        active=channel.state is ChannelState.ACTIVE,
        hold_chars=settings.hold_chars,
        fallback_after_chars=settings.fallback_after_chars,
        commit_unresolved=settings.commit_unresolved,
        allow_gap_only=bool(transcript.committed_tokens),
    )

    trim_before_s: float | None = None
    if commit_len > 0:
        committed_part = tail[:commit_len]
        transcript.committed_tokens = normalize_gap_tokens(transcript.committed_tokens + committed_part)
        if transcript.pending_separator is not None and transcript.pending_separator in committed_part:
            transcript.pending_separator = None
        trim_before_s = trim_time_from_committed_tokens(transcript.committed_tokens, settings.audio_context_chars)
        if trim_before_s is not None and trim_before_s > transcript.audio_trim_before_s:
            transcript.audio_trim_before_s = trim_before_s

    transcript.tentative_tokens = normalize_gap_tokens(tail[commit_len:])
    transcript.last_channel_state = channel.state

    visible = transcript.visible_tokens()
    return TranscriptUpdate(
        _winner_from_tokens(winner, visible, stable_count=len(transcript.committed_tokens)),
        trim_before_s=trim_before_s,
    )


def winner_from_transcript(channel: ChannelSignal, transcript: ChannelTranscript, *, time_s: float) -> ChannelWinner | None:
    if not transcript.committed_tokens and not transcript.tentative_tokens:
        return None
    if channel.state is not ChannelState.ACTIVE and transcript.tentative_tokens:
        transcript.committed_tokens = normalize_gap_tokens(transcript.committed_tokens + transcript.tentative_tokens)
        transcript.tentative_tokens = ()
        transcript.pending_separator = None
    tokens = transcript.visible_tokens()
    transcript.last_channel_state = channel.state
    return ChannelWinner(
        channel_id=channel.channel_id,
        carrier_hz=channel.carrier_hz,
        text="",
        state="committed",
        updated_at_s=time_s,
        tokens=tokens,
        stable_token_count=len(transcript.committed_tokens),
    )


def absolute_tokens(tokens: Iterable[DecodeToken], *, offset_s: float) -> tuple[DecodeToken, ...]:
    output: list[DecodeToken] = []
    for token in tokens:
        start = None if token.start_s is None else float(token.start_s) + float(offset_s)
        end = None if token.end_s is None else float(token.end_s) + float(offset_s)
        output.append(replace(token, start_s=_round_optional(start), end_s=_round_optional(end)))
    return tuple(output)


def relative_tokens(tokens: Iterable[DecodeToken], *, offset_s: float) -> tuple[DecodeToken, ...]:
    output: list[DecodeToken] = []
    for token in tokens:
        start = None if token.start_s is None else float(token.start_s) - float(offset_s)
        end = None if token.end_s is None else float(token.end_s) - float(offset_s)
        output.append(replace(token, start_s=_round_optional(start), end_s=_round_optional(end)))
    return tuple(output)


def uncommitted_tail(transcript: ChannelTranscript, raw_tokens: tuple[DecodeToken, ...]) -> tuple[DecodeToken, ...]:
    cutoff = last_committed_time_s(transcript)
    if cutoff is None:
        return raw_tokens
    tail: list[DecodeToken] = []
    eps = 1e-3
    for token in raw_tokens:
        token_start = token.start_s
        token_end = token.end_s if token.end_s is not None else token.start_s
        if token_end is None:
            tail.append(token)
            continue
        if token.is_content:
            # A re-decoded overlap may slightly change a character end time.
            # If the character starts in already committed time, it is the same
            # historical character and must not be appended again.
            if token_start is not None and token_start < cutoff - eps:
                continue
            if token_end <= cutoff + eps:
                continue
            tail.append(token)
            continue
        # Gap tokens are exactly the boundary information we need to keep.
        # A gap starting at/just before the commit cutoff and ending after it
        # separates the committed text from future content, so keep it.
        if token_end > cutoff + eps:
            tail.append(token)
    return tuple(tail)


def add_pending_separator_if_needed(
    transcript: ChannelTranscript,
    channel: ChannelSignal,
    tail: tuple[DecodeToken, ...],
) -> tuple[DecodeToken, ...]:
    resumed_after_pause = (
        transcript.last_channel_state is not None
        and transcript.last_channel_state is not ChannelState.ACTIVE
        and channel.state is ChannelState.ACTIVE
    )
    if resumed_after_pause and transcript.committed_tokens and tail and not tokens_start_with_gap(tail):
        transcript.pending_separator = gap_token("session_gap")
    if transcript.pending_separator is not None and tail and not tokens_start_with_gap(tail) and not tokens_end_with_gap(transcript.committed_tokens):
        return (transcript.pending_separator,) + tail
    return tail


def commit_prefix_len(
    tokens: tuple[DecodeToken, ...],
    *,
    active: bool,
    hold_chars: int,
    fallback_after_chars: int,
    commit_unresolved: bool,
    allow_gap_only: bool = False,
) -> int:
    if not tokens:
        return 0
    max_len = len(tokens) if not active else length_before_last_content_tokens(tokens, hold_chars)
    if max_len <= 0 and active and allow_gap_only and all(token.is_gap for token in tokens):
        max_len = len(tokens)
    if max_len <= 0:
        return 0
    if active:
        safe_len = safe_commit_prefix_len(
            tokens,
            max_len,
            fallback_after_chars=fallback_after_chars,
            allow_gap_only=allow_gap_only,
        )
    else:
        safe_len = max_len
    if not commit_unresolved:
        safe_len = resolved_prefix_len(tokens, safe_len)
    if safe_len > 0 and not any(token.is_content for token in tokens[:safe_len]):
        return safe_len if allow_gap_only and any(token.is_gap for token in tokens[:safe_len]) else 0
    return safe_len


def safe_commit_prefix_len(
    tokens: tuple[DecodeToken, ...],
    max_len: int,
    *,
    fallback_after_chars: int,
    allow_gap_only: bool = False,
) -> int:
    prefix = tokens[: max(0, max_len)]
    if not prefix:
        return 0
    for index in range(len(prefix) - 1, -1, -1):
        if prefix[index].kind in GAP_KINDS and any(token.is_content for token in prefix[:index]):
            return index + 1
    if allow_gap_only and all(token.is_gap for token in prefix):
        return len(prefix)
    if fallback_after_chars <= 0:
        return 0
    content_positions = [index for index, token in enumerate(prefix) if token.is_content]
    if len(content_positions) < fallback_after_chars:
        return 0
    keep_count = max(1, len(content_positions) // 2)
    return content_positions[keep_count - 1] + 1


def length_before_last_content_tokens(tokens: tuple[DecodeToken, ...], hold_count: int) -> int:
    if hold_count <= 0:
        return len(tokens)
    seen = 0
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index].is_content:
            seen += 1
            if seen >= hold_count:
                return index
    return 0


def resolved_prefix_len(tokens: tuple[DecodeToken, ...], max_len: int) -> int:
    for index, token in enumerate(tokens[:max_len]):
        if token.kind == TOKEN_UNKNOWN:
            return index
    return max_len


def trim_time_from_committed_tokens(tokens: tuple[DecodeToken, ...], audio_context_chars: int) -> float | None:
    last_session_gap_end = _last_session_gap_time_s(tokens)
    if tokens and tokens[-1].kind == "session_gap":
        return last_session_gap_end

    timed_content = [
        token
        for token in tokens
        if token.is_content
        and (token.start_s is not None or token.end_s is not None)
        and (last_session_gap_end is None or _token_time_s(token) is None or _token_time_s(token) >= last_session_gap_end)
    ]
    if not timed_content:
        return last_session_gap_end

    context_chars = max(0, int(audio_context_chars))
    if context_chars <= 0:
        trim = timed_content[-1].end_s if timed_content[-1].end_s is not None else timed_content[-1].start_s
    else:
        keep_index = max(0, len(timed_content) - context_chars)
        keep_token = timed_content[keep_index]
        trim = keep_token.start_s if keep_token.start_s is not None else keep_token.end_s
    if last_session_gap_end is not None and trim is not None:
        return max(float(last_session_gap_end), float(trim))
    return trim


def _last_session_gap_time_s(tokens: tuple[DecodeToken, ...]) -> float | None:
    for token in reversed(tokens):
        if token.kind != "session_gap":
            continue
        if token.end_s is not None:
            return token.end_s
        if token.start_s is not None:
            return token.start_s
    return None


def _token_time_s(token: DecodeToken) -> float | None:
    if token.start_s is not None:
        return token.start_s
    return token.end_s


def last_committed_time_s(transcript: ChannelTranscript) -> float | None:
    for token in reversed(transcript.committed_tokens):
        if token.end_s is not None:
            return token.end_s
        if token.start_s is not None:
            return token.start_s
    return None


def preserve_non_degraded_tentative(previous: tuple[DecodeToken, ...], current: tuple[DecodeToken, ...]) -> tuple[DecodeToken, ...]:
    if not previous:
        return current
    if not current:
        return previous
    previous_content = sum(1 for token in previous if token.is_content)
    current_content = sum(1 for token in current if token.is_content)
    if current_content == 0:
        return previous
    common = common_prefix_len_tokens(previous, current)
    if common == len(previous):
        return current
    if common == len(current) and current_content + 2 < previous_content:
        return previous
    if common == 0 and current_content * 2 < previous_content:
        return previous
    return current


def common_prefix_len_tokens(left: tuple[DecodeToken, ...], right: tuple[DecodeToken, ...]) -> int:
    left_sig = token_signature(left)
    right_sig = token_signature(right)
    size = min(len(left_sig), len(right_sig))
    for index in range(size):
        if left_sig[index] != right_sig[index]:
            return index
    return size


def normalize_gap_tokens(tokens: tuple[DecodeToken, ...]) -> tuple[DecodeToken, ...]:
    out: list[DecodeToken] = []
    for token in tokens:
        if token.kind in GAP_KINDS:
            if not out:
                continue
            if out[-1].kind in GAP_KINDS:
                if out[-1].kind != "session_gap" and token.kind == "session_gap":
                    out[-1] = token
                continue
        out.append(token)
    # Do not strip trailing gaps. A trailing word/session gap is real state:
    # it separates already committed content from the next received content.
    # Rendering may hide a final blank, but the token must survive.
    return tuple(out)


def tokens_start_with_gap(tokens: tuple[DecodeToken, ...]) -> bool:
    return bool(tokens and tokens[0].kind in GAP_KINDS)


def tokens_end_with_gap(tokens: tuple[DecodeToken, ...]) -> bool:
    return bool(tokens and tokens[-1].kind in GAP_KINDS)


def _winner_from_tokens(winner: ChannelWinner, tokens: tuple[DecodeToken, ...], *, stable_count: int) -> ChannelWinner:
    return replace(
        winner,
        text="",
        tokens=tokens,
        stable_token_count=max(0, int(stable_count)),
    )


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


__all__ = [
    "ChannelTranscript",
    "TranscriptConfig",
    "TranscriptUpdate",
    "apply_transcript_update",
    "winner_from_transcript",
    "absolute_tokens",
    "relative_tokens",
    "uncommitted_tail",
    "normalize_gap_tokens",
    "safe_commit_prefix_len",
]
