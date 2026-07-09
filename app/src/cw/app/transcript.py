from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from cw.decoder.tokens import (
    GAP_KINDS,
    DecodeToken,
    TOKEN_UNKNOWN,
    token_signature,
    tokens_to_text,
)
from cw.receiving.models import ChannelSignal, ChannelState
from cw.selection.models import ChannelWinner


@dataclass(frozen=True)
class TranscriptConfig:
    """Application-level rules for splitting the current winner.

    This object is intentionally not a transcript memory.  It only controls how
    the current selected token sequence is split into a stable prefix and a
    carried/tentative tail.
    """

    enabled: bool = True
    hold_chars: int = 3
    fallback_after_chars: int = 6
    commit_unresolved: bool = False
    audio_context_chars: int = 2

    @classmethod
    def from_config(cls, config) -> "TranscriptConfig":
        return cls(
            enabled=bool(getattr(config, "stable_prefix_enabled", True)),
            hold_chars=max(0, int(getattr(config, "stable_prefix_hold_chars", 3))),
            fallback_after_chars=max(0, int(getattr(config, "stable_prefix_fallback_after_chars", 6))),
            commit_unresolved=bool(getattr(config, "stable_prefix_commit_unresolved", False)),
            audio_context_chars=max(0, int(getattr(config, "stable_audio_context_chars", 2))),
        )


@dataclass(frozen=True)
class TranscriptSplit:
    """Stable/carry split for one currently selected decoded token sequence."""

    stable_tokens: tuple[DecodeToken, ...] = ()
    carried_tokens: tuple[DecodeToken, ...] = ()
    trim_before_s: float | None = None

    @property
    def tokens(self) -> tuple[DecodeToken, ...]:
        return self.stable_tokens + self.carried_tokens

    @property
    def stable_token_count(self) -> int:
        return len(self.stable_tokens)


# Backwards-compatible alias for older tests/imports.  It is no longer a
# persistent per-channel object.
ChannelTranscript = TranscriptSplit


def split_winner_tokens(
    channel: ChannelSignal,
    winner: ChannelWinner,
    config,
    *,
    force_trim_before_s: float | None = None,
) -> tuple[ChannelWinner, float | None]:
    """Split the selected winner and return a display-ready winner plus trim time.

    The input winner carries tokens relative to the channel audio window.  The
    output winner carries absolute-time tokens because the JSON/debug stream is
    an application snapshot, while the trim time is fed back into receiving.
    """

    settings = TranscriptConfig.from_config(config)
    absolute = absolute_tokens(winner.tokens, offset_s=channel.start_s)
    split = split_transcript_tokens(
        absolute,
        active=channel.state is ChannelState.ACTIVE,
        config=settings,
        force_trim_before_s=force_trim_before_s,
    )
    return (
        replace(
            winner,
            text="",
            tokens=split.tokens,
            stable_token_count=split.stable_token_count,
        ),
        split.trim_before_s,
    )


def split_transcript_tokens(
    tokens: Iterable[DecodeToken],
    *,
    active: bool,
    config: TranscriptConfig,
    force_trim_before_s: float | None = None,
) -> TranscriptSplit:
    """Split one current token sequence into stable and carried parts.

    No channel history is stored here.  The stable prefix tells the app what can
    be marked stable in JSON and where receiving may trim already processed
    audio.  The carried tail remains visible/tentative and will stay available
    because receiving keeps the corresponding audio tail.
    """

    normalized = normalize_gap_tokens(tuple(tokens))
    if not config.enabled:
        return TranscriptSplit(stable_tokens=(), carried_tokens=normalized)

    stable_len = stable_prefix_len(
        normalized,
        active=active,
        hold_chars=config.hold_chars,
        fallback_after_chars=config.fallback_after_chars,
        commit_unresolved=config.commit_unresolved,
    )
    stable_len = forced_stable_prefix_len(
        normalized,
        stable_len,
        force_trim_before_s=force_trim_before_s,
        audio_context_chars=config.audio_context_chars,
    )
    stable_prefix = normalize_gap_tokens(normalized[:stable_len])
    trim_before_s = trim_time_from_stable_tokens(stable_prefix, config.audio_context_chars) if stable_prefix else None
    if stable_prefix and force_trim_before_s is not None and (trim_before_s is None or trim_before_s < float(force_trim_before_s)):
        emergency_trim = trim_time_from_stable_tokens(stable_prefix, 0)
        if emergency_trim is not None:
            trim_before_s = emergency_trim

    # ``audio_context_chars`` means that the last stable content tokens remain
    # in the receiver audio tail.  Those retained context tokens must not be
    # emitted as stable/committed output yet, because the next audio window will
    # decode them again.  Commit only the stable tokens that are strictly before
    # the trim point; everything from the trim point onward is carried.
    commit_len = committable_prefix_len(normalized, stable_len, trim_before_s)
    stable = normalize_gap_tokens(normalized[:commit_len])
    carried = normalize_gap_tokens(normalized[commit_len:])
    return TranscriptSplit(
        stable_tokens=stable,
        carried_tokens=carried,
        trim_before_s=trim_before_s,
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


def stable_prefix_len(
    tokens: tuple[DecodeToken, ...],
    *,
    active: bool,
    hold_chars: int,
    fallback_after_chars: int,
    commit_unresolved: bool,
) -> int:
    if not tokens:
        return 0
    max_len = len(tokens) if not active else length_before_last_content_tokens(tokens, hold_chars)
    if max_len <= 0:
        return 0
    safe_len = (
        safe_commit_prefix_len(tokens, max_len, fallback_after_chars=fallback_after_chars)
        if active
        else max_len
    )
    if not commit_unresolved:
        safe_len = resolved_prefix_len(tokens, safe_len)
    if safe_len > 0 and not any(token.is_content for token in tokens[:safe_len]):
        return 0
    return safe_len


# Kept under the old name because the tests and the concept still use the same
# safe-boundary rule; the surrounding transcript object has been removed.
def forced_stable_prefix_len(
    tokens: tuple[DecodeToken, ...],
    current_len: int,
    *,
    force_trim_before_s: float | None,
    audio_context_chars: int,
) -> int:
    """Raise the stable prefix when history pressure needs a safe trim.

    This is an emergency policy for ``max_history_s``.  It never chooses an
    arbitrary audio cut; it only promotes the current selected token sequence to
    stable at token boundaries until ``trim_time_from_stable_tokens()`` can
    reach the requested floor.  If no token boundary can satisfy the floor, the
    largest available token prefix is returned and receiving will retain extra
    audio rather than cut blindly.
    """

    if force_trim_before_s is None or not tokens:
        return max(0, min(int(current_len), len(tokens)))

    start_len = max(0, min(int(current_len), len(tokens)))
    target = float(force_trim_before_s)

    def trim_at(length: int, context_chars: int) -> float | None:
        stable = normalize_gap_tokens(tokens[:length])
        return trim_time_from_stable_tokens(stable, context_chars) if stable else None

    for length in range(start_len, len(tokens) + 1):
        trim = trim_at(length, audio_context_chars)
        if trim is not None and trim >= target:
            return length

    # If the normal context-retention rule keeps too much old audio, sacrifice
    # context before sacrificing character boundaries.  This branch is rare and
    # exists only to make the global history limit an emergency commit rather
    # than a character cutter.
    if audio_context_chars > 0:
        for length in range(start_len, len(tokens) + 1):
            trim = trim_at(length, 0)
            if trim is not None and trim >= target:
                return length

    return len(tokens)



def committable_prefix_len(
    tokens: tuple[DecodeToken, ...],
    stable_len: int,
    trim_before_s: float | None,
    *,
    tolerance_s: float = 1e-6,
) -> int:
    """Return the stable prefix that is not retained as audio context.

    ``stable_len`` is the prefix that may be trusted by the current selection.
    ``trim_before_s`` is the absolute receiver trim point.  Tokens at or after
    that point are kept in audio context and therefore stay carried/tentative in
    public output; otherwise a later window could commit the same physical
    characters again with a different temporary gap decision.
    """

    max_len = max(0, min(int(stable_len), len(tokens)))
    if max_len <= 0:
        return 0
    if trim_before_s is None:
        return max_len

    floor = float(trim_before_s) + float(tolerance_s)
    commit_len = 0
    for index, token in enumerate(tokens[:max_len]):
        token_time = _token_end_time_s(token)
        if token_time is None:
            # Timeless tokens cannot be proven to be retained context.  Keep the
            # old stable behavior for such synthetic/test tokens.
            commit_len = index + 1
            continue
        if token_time <= floor:
            commit_len = index + 1
            continue
        break
    return commit_len

def safe_commit_prefix_len(
    tokens: tuple[DecodeToken, ...],
    max_len: int,
    *,
    fallback_after_chars: int,
) -> int:
    prefix = tokens[: max(0, max_len)]
    if not prefix:
        return 0
    for index in range(len(prefix) - 1, -1, -1):
        if prefix[index].kind in GAP_KINDS and any(token.is_content for token in prefix[:index]):
            return index + 1
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


def trim_time_from_stable_tokens(tokens: tuple[DecodeToken, ...], audio_context_chars: int) -> float | None:
    last_gap_end = _last_gap_time_s(tokens)
    if tokens and tokens[-1].kind in GAP_KINDS:
        return last_gap_end

    timed_content = [
        token
        for token in tokens
        if token.is_content
        and (token.start_s is not None or token.end_s is not None)
        and (last_gap_end is None or _token_time_s(token) is None or _token_time_s(token) >= last_gap_end)
    ]
    if not timed_content:
        return last_gap_end

    context_chars = max(0, int(audio_context_chars))
    if context_chars <= 0:
        trim = timed_content[-1].end_s if timed_content[-1].end_s is not None else timed_content[-1].start_s
    else:
        keep_index = max(0, len(timed_content) - context_chars)
        keep_token = timed_content[keep_index]
        trim = keep_token.start_s if keep_token.start_s is not None else keep_token.end_s
    if last_gap_end is not None and trim is not None:
        return max(float(last_gap_end), float(trim))
    return trim


# Old public name retained as an alias to avoid needless churn in callers/tests.
trim_time_from_committed_tokens = trim_time_from_stable_tokens


def _last_session_gap_time_s(tokens: tuple[DecodeToken, ...]) -> float | None:
    for token in reversed(tokens):
        if token.kind != "session_gap":
            continue
        if token.end_s is not None:
            return token.end_s
        if token.start_s is not None:
            return token.start_s
    return None


def _last_gap_time_s(tokens: tuple[DecodeToken, ...]) -> float | None:
    for token in reversed(tokens):
        if token.kind not in GAP_KINDS:
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


def _token_end_time_s(token: DecodeToken) -> float | None:
    if token.end_s is not None:
        return float(token.end_s)
    if token.start_s is not None:
        return float(token.start_s)
    return None


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
    return tuple(out)


def common_prefix_len_tokens(left: tuple[DecodeToken, ...], right: tuple[DecodeToken, ...]) -> int:
    left_sig = token_signature(left)
    right_sig = token_signature(right)
    size = min(len(left_sig), len(right_sig))
    for index in range(size):
        if left_sig[index] != right_sig[index]:
            return index
    return size


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


__all__ = [
    "ChannelTranscript",
    "TranscriptConfig",
    "TranscriptSplit",
    "absolute_tokens",
    "common_prefix_len_tokens",
    "normalize_gap_tokens",
    "relative_tokens",
    "safe_commit_prefix_len",
    "split_transcript_tokens",
    "split_winner_tokens",
    "stable_prefix_len",
    "trim_time_from_committed_tokens",
    "trim_time_from_stable_tokens",
    "committable_prefix_len",
]
