from __future__ import annotations

from dataclasses import dataclass

from cw.receiving.config import ReceivingConfig, strict_channel_match_hz
from cw.receiving.models import CarrierObservation, ChannelState
from cw.receiving.state import TrackedChannel


@dataclass(frozen=True)
class ChannelTrackerResult:
    channels: tuple[TrackedChannel, ...]


class ChannelTracker:
    """Tracks raw carrier observations as stable receiving-channel snapshots."""

    def __init__(self, config: ReceivingConfig) -> None:
        self.config = config
        self._channels: list[TrackedChannel] = []
        self._next_channel_id = 1

    @property
    def channels(self) -> tuple[TrackedChannel, ...]:
        return tuple(self._channels)

    def update(self, observations: tuple[CarrierObservation, ...], *, time_s: float) -> ChannelTrackerResult:
        observed: set[int] = set()
        report_channels: list[TrackedChannel] = []
        reported_ids: set[int] = set()

        for observation in observations:
            channel = self._channel_for_observation(observation, time_s=time_s)
            if channel is None:
                continue
            observed.add(channel.channel_id)
            self._update_seen_channel_state(channel, time_s=time_s)
            if channel.channel_id not in reported_ids:
                report_channels.append(channel)
                reported_ids.add(channel.channel_id)

        for channel in self.finalize_inactive(time_s=time_s, observed_channel_ids=observed):
            if channel.channel_id not in reported_ids:
                report_channels.append(channel)
                reported_ids.add(channel.channel_id)

        return ChannelTrackerResult(tuple(report_channels))

    def finish(self, *, time_s: float) -> tuple[TrackedChannel, ...]:
        closed: list[TrackedChannel] = []
        for channel in self._channels:
            if channel.state in {ChannelState.DORMANT, ChannelState.DROPPED}:
                continue
            channel.dormant = True
            channel.state = ChannelState.DORMANT if channel.channel_started else ChannelState.DROPPED
            closed.append(channel)
        return tuple(closed)

    def finalize_inactive(self, *, time_s: float, observed_channel_ids: set[int] | frozenset[int] | None = None) -> tuple[TrackedChannel, ...]:
        observed_channel_ids = observed_channel_ids or set()
        closed: list[TrackedChannel] = []
        for channel in self._channels:
            if channel.channel_id in observed_channel_ids or channel.dormant:
                continue
            if time_s - channel.last_seen_s < self.config.max_track_gap_s:
                continue
            channel.dormant = True
            channel.hits = 0
            channel.state = ChannelState.DORMANT if channel.channel_started else ChannelState.DROPPED
            closed.append(channel)
        return tuple(closed)

    def _update_seen_channel_state(self, channel: TrackedChannel, *, time_s: float) -> None:
        if channel.hits < self.config.min_track_hits:
            channel.state = ChannelState.CANDIDATE
            return

        if self._is_alias_of_recent_channel(channel, time_s=time_s):
            channel.dormant = True
            channel.hits = 0
            channel.state = ChannelState.DROPPED
            return

        channel.channel_started = True
        channel.dormant = False
        channel.state = ChannelState.ACTIVE

    def _channel_for_observation(self, observation: CarrierObservation, *, time_s: float) -> TrackedChannel | None:
        channel = self._best_channel_for_carrier(observation.carrier_hz, time_s=time_s)
        if channel is None:
            channel = TrackedChannel(
                channel_id=self._next_channel_id,
                carrier_hz=round(float(observation.carrier_hz), 3),
                first_seen_s=time_s,
                last_seen_s=time_s,
                relative_power=observation.relative_power,
                snr_db=observation.snr_db,
                power=observation.power,
            )
            self._next_channel_id += 1
            self._channels.append(channel)
            return channel
        channel.hits += 1
        channel.carrier_hz = _smooth_carrier(channel.carrier_hz, observation.carrier_hz, self.config.carrier_smoothing)
        channel.last_seen_s = time_s
        channel.relative_power = observation.relative_power
        channel.snr_db = observation.snr_db
        channel.power = observation.power
        if channel.dormant:
            channel.dormant = False
        return channel

    def _best_channel_for_carrier(self, carrier_hz: float, *, time_s: float) -> TrackedChannel | None:
        strict_match_hz = strict_channel_match_hz(self.config)
        reacquire_hz = _channel_reacquire_hz(self.config)
        matches: list[tuple[int, float, TrackedChannel]] = []
        for channel in self._channels:
            delta_hz = abs(channel.carrier_hz - carrier_hz)
            if delta_hz <= strict_match_hz:
                matches.append((0, delta_hz, channel))
                continue
            recently_seen = time_s - channel.last_seen_s <= self.config.channel_reacquire_s
            if recently_seen and delta_hz <= reacquire_hz:
                matches.append((1, delta_hz, channel))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1], -item[2].hits))
        return matches[0][2]

    def _is_alias_of_recent_channel(self, channel: TrackedChannel, *, time_s: float) -> bool:
        if not self.config.alias_suppression:
            return False
        alias_hz = self.config.channel_alias_hz
        if alias_hz is None or alias_hz <= 0:
            return False
        for other in self._channels:
            if other is channel or other.dormant or not other.channel_started:
                continue
            if time_s - other.last_seen_s > self.config.channel_alias_s:
                continue
            if abs(other.carrier_hz - channel.carrier_hz) > alias_hz:
                continue
            if other.hits >= channel.hits + 2 or (other.relative_power or 0.0) > (channel.relative_power or 0.0) + 0.20:
                return True
        return False



def _channel_reacquire_hz(config: ReceivingConfig) -> float:
    return float(config.channel_reacquire_hz)


def _smooth_carrier(previous: float, current: float, smoothing: float) -> float:
    alpha = min(1.0, max(0.0, float(smoothing)))
    return round(previous * (1.0 - alpha) + current * alpha, 3)
