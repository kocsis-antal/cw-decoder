from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ChannelState(Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    DROPPED = "dropped"


@dataclass(frozen=True)
class CarrierObservation:
    """A raw carrier observation before stable channel tracking.

    This is receiving-internal input to the channel tracker. ``relative_power``
    is measured against the strongest accepted spectrum peak in the current
    observation window; it is not a decoder quality value.
    """

    carrier_hz: float
    relative_power: float
    snr_db: float = 0.0
    power: float = 0.0


@dataclass(frozen=True)
class ChannelSignal:
    """Current receiving-layer snapshot of one tracked channel.

    The channel id is stable even when the carrier drifts.  ``state`` is a
    typed lifecycle state, not an event.  Receiving-only measurements used for
    channel tracking stay inside the receiving layer.
    """

    channel_id: int
    carrier_hz: float
    start_s: float
    end_s: float
    audio_window: np.ndarray
    sample_rate: int
    state: ChannelState = ChannelState.ACTIVE

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def has_audio(self) -> bool:
        # Final DORMANT snapshots may still carry the last active audio window
        # so the decoder can flush the tail when the stream closes.  Candidate
        # and dropped channels still have an empty window.
        return len(self.audio_window) > 0

    @property
    def samples(self) -> np.ndarray:
        """Convenience alias for code that still speaks in audio samples."""
        return self.audio_window


@dataclass(frozen=True)
class ReceivingStats:
    processed_duration_s: float = 0.0
    retained_audio_s: float = 0.0
    input_rms: float = 0.0
    input_peak: float = 0.0


@dataclass(frozen=True)
class ReceiveChunk:
    time_s: float
    channels: tuple[ChannelSignal, ...] = ()
    stats: ReceivingStats = field(default_factory=ReceivingStats)


__all__ = [
    "CarrierObservation",
    "ChannelSignal",
    "ChannelState",
    "ReceiveChunk",
    "ReceivingStats",
]
