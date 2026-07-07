from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cw.signal.models import SignalTrack
from cw.decoder.tokens import DecodeToken


@dataclass(frozen=True)
class DecodedText:
    """One text returned by a decoder.

    This is not a score and not a ranking.  The only decoder-owned quality fact
    exposed here is how many Morse tokens could not be mapped to a valid
    character.  Invalid tokens are visible in text with the decode error marker;
    a literal '?' remains a valid Morse character.
    """

    text: str
    unresolved_tokens: int = 0
    tokens: tuple[DecodeToken, ...] = ()


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
