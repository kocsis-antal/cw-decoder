from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from cw.morse_table import DECODE_ERROR_MARKER

TOKEN_CHAR = "char"
TOKEN_WORD_GAP = "word_gap"
TOKEN_SESSION_GAP = "session_gap"
TOKEN_UNKNOWN = "unknown"
GAP_KINDS = {TOKEN_WORD_GAP, TOKEN_SESSION_GAP}
CONTENT_KINDS = {TOKEN_CHAR, TOKEN_UNKNOWN}


@dataclass(frozen=True)
class DecodeToken:
    """One decoded CW token with optional timing relative to the channel window."""

    kind: str
    value: str = ""
    start_s: float | None = None
    end_s: float | None = None

    @property
    def is_gap(self) -> bool:
        return self.kind in GAP_KINDS

    @property
    def is_content(self) -> bool:
        return self.kind in CONTENT_KINDS

    @property
    def signature(self) -> tuple[str, str]:
        return (self.kind, self.value if self.kind in {TOKEN_CHAR, TOKEN_UNKNOWN} else "")


def char_token(value: str, *, start_s: float | None = None, end_s: float | None = None) -> DecodeToken:
    return DecodeToken(TOKEN_CHAR, str(value), _round_optional(start_s), _round_optional(end_s))


def unknown_token(*, start_s: float | None = None, end_s: float | None = None) -> DecodeToken:
    return DecodeToken(TOKEN_UNKNOWN, DECODE_ERROR_MARKER, _round_optional(start_s), _round_optional(end_s))


def gap_token(kind: str, *, start_s: float | None = None, end_s: float | None = None) -> DecodeToken:
    if kind not in GAP_KINDS:
        raise ValueError(f"not a gap token kind: {kind!r}")
    return DecodeToken(kind, "", _round_optional(start_s), _round_optional(end_s))


def tokens_to_text(tokens: Iterable[DecodeToken], *, brackets: bool = False) -> str:
    parts: list[str] = []
    previous_gap = False
    for token in tokens:
        if token.kind == TOKEN_CHAR:
            parts.append(token.value)
            previous_gap = False
        elif token.kind == TOKEN_UNKNOWN:
            parts.append(DECODE_ERROR_MARKER)
            previous_gap = False
        elif token.kind == TOKEN_WORD_GAP:
            if parts and not previous_gap:
                parts.append(" ")
                previous_gap = True
        elif token.kind == TOKEN_SESSION_GAP:
            if parts and not previous_gap:
                parts.append("   ")
                previous_gap = True
    text = "".join(parts).strip()
    return f"[{text}]" if brackets and text else text


def token_signature(tokens: Iterable[DecodeToken]) -> tuple[tuple[str, str], ...]:
    return tuple(token.signature for token in tokens)


def tokens_to_dicts(tokens: Iterable[DecodeToken]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for token in tokens:
        item: dict[str, Any] = {"kind": token.kind}
        if token.value:
            item["value"] = token.value
        if token.start_s is not None:
            item["start_s"] = _round_optional(token.start_s)
        if token.end_s is not None:
            item["end_s"] = _round_optional(token.end_s)
        output.append(item)
    return output


def token_from_dict(payload: dict[str, Any]) -> DecodeToken:
    return DecodeToken(
        kind=str(payload.get("kind") or ""),
        value=str(payload.get("value") or ""),
        start_s=_float_or_none(payload.get("start_s")),
        end_s=_float_or_none(payload.get("end_s")),
    )


def tokens_from_dicts(payloads: Iterable[dict[str, Any]]) -> tuple[DecodeToken, ...]:
    return tuple(token_from_dict(item) for item in payloads if isinstance(item, dict))


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DecodeToken",
    "TOKEN_CHAR",
    "TOKEN_WORD_GAP",
    "TOKEN_SESSION_GAP",
    "TOKEN_UNKNOWN",
    "GAP_KINDS",
    "CONTENT_KINDS",
    "char_token",
    "unknown_token",
    "gap_token",
    "tokens_to_text",
    "token_signature",
    "tokens_to_dicts",
    "tokens_from_dicts",
]
