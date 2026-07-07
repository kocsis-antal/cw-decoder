from __future__ import annotations

from cw.decoder.api import DecodeResult, Decoder
from cw.signal.models import SignalTrack


class DecoderBank:
    def __init__(self, decoders: tuple[Decoder, ...]) -> None:
        self.decoders = decoders

    def decode(self, track: SignalTrack) -> tuple[DecodeResult, ...]:
        return tuple(decoder.decode(track) for decoder in self.decoders)
