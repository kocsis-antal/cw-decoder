from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cw.signal.models import SignalTrack
from cw.decoder.tokens import DecodeToken


@dataclass(frozen=True)
class DecodedText:
    """One text returned by a decoder.

    This is not a ranking. ``timing_quality`` is the decoder-owned physical
    timing-fit value; lower is better. ``unresolved_tokens`` is descriptive
    output only and must not be used as a hidden decoder rank. Selection owns
    the comparison between candidates.
    """

    text: str
    unresolved_tokens: int = 0
    tokens: tuple[DecodeToken, ...] = ()
    timing_quality: float = 0.0


@dataclass(frozen=True)
class DecodeResult:
    decoder: str
    answers: tuple[DecodedText, ...] = ()

    @property
    def text(self) -> str:
        return self.answers[0].text if self.answers else ""


class Decoder(Protocol):
    name: str

    def decode(self, track: SignalTrack) -> DecodeResult: ...


__all__ = ["DecodedText", "DecodeResult", "Decoder"]
