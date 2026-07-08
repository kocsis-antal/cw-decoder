from __future__ import annotations

from dataclasses import dataclass

from cw.receiving.models import ChannelState


@dataclass
class TrackedChannel:
    """Mutable receiving-internal channel state.

    Public consumers should see ``ChannelSignal`` snapshots, not this object.
    """

    channel_id: int
    carrier_hz: float
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    channel_started: bool = False
    dormant: bool = False
    relative_power: float | None = None
    snr_db: float | None = None
    power: float | None = None
    state: ChannelState = ChannelState.CANDIDATE
    dropped_by_limit: bool = False
    audio_trim_before_s: float = 0.0
