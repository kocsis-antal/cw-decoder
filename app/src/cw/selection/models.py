from __future__ import annotations

from dataclasses import dataclass

from cw.decoder.api import DecodeResult
from cw.decoder.tokens import DecodeToken


@dataclass(frozen=True)
class TrackDecodedTexts:
    """Decoded answers produced from one signal track.

    The analyzer string identifies the signal-track source.  Selection may use
    it internally for support/diversity checks, but it is not a public score.
    """

    analyzer: str
    results: tuple[DecodeResult, ...] = ()
    unknown_ratio: float = 0.0
    rejected: bool = False
    rejection_reason: str = ""


@dataclass(frozen=True)
class ChannelDecodedTexts:
    """All decoded answers currently available for one receiving channel."""

    channel_id: int
    carrier_hz: float
    tracks: tuple[TrackDecodedTexts, ...] = ()


@dataclass(frozen=True)
class SelectionInput:
    """Batch input for the selection layer.

    The application composition layer only wires preceding layers together.
    Grouping, support counting, scoring and winner choice belong to selection.
    """

    channels: tuple[ChannelDecodedTexts, ...] = ()


@dataclass(frozen=True)
class ChannelWinner:
    channel_id: int
    carrier_hz: float
    text: str
    state: str
    updated_at_s: float
    tokens: tuple[DecodeToken, ...] = ()
    stable_token_count: int = 0


@dataclass(frozen=True)
class SelectionChunk:
    time_s: float
    winners: tuple[ChannelWinner, ...]


__all__ = [
    "TrackDecodedTexts",
    "ChannelDecodedTexts",
    "SelectionInput",
    "ChannelWinner",
    "SelectionChunk",
]
