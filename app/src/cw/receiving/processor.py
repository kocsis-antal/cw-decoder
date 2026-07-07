from __future__ import annotations

import numpy as np

from cw.io.models import AudioBlock
from cw.receiving.audio_history import AudioRingBuffer
from cw.receiving.carrier_observer import CarrierObserver
from cw.receiving.channel_tracker import ChannelTracker
from cw.receiving.config import ReceivingConfig
from cw.receiving.models import ChannelSignal, ChannelState, ReceiveChunk, ReceivingStats
from cw.receiving.state import TrackedChannel


class Receiver:
    """Converts audio blocks into channel-centric receiving snapshots.

    This layer stops before signal segmentation and Morse/text decoding.  It
    emits stable channel ids and current channel state snapshots, not events.
    """

    def __init__(self, sample_rate: int, config: ReceivingConfig) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(sample_rate)
        self.config = config
        self.processed_duration_s = 0.0
        self.last_input_rms = 0.0
        self.last_input_peak = 0.0

        self._audio = AudioRingBuffer(self.sample_rate, self.config.max_history_s)
        self._last_observe_s = 0.0
        self._carrier_observer = CarrierObserver(self.sample_rate, self.config)
        self._channel_tracker = ChannelTracker(self.config)

    @property
    def _window_start_s(self) -> float:
        return self._audio.start_s

    def push(self, block: AudioBlock) -> ReceiveChunk:
        if block.sample_rate != self.sample_rate:
            raise ValueError("audio block sample_rate changed during stream")
        samples = np.asarray(block.samples, dtype=np.float32)
        if len(samples) == 0:
            return ReceiveChunk(time_s=self.processed_duration_s, stats=self._stats())

        self.last_input_rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        self.last_input_peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        self._audio.append(samples)
        self.processed_duration_s = max(block.end_s, self.processed_duration_s + len(samples) / self.sample_rate)

        if self.processed_duration_s - self._last_observe_s < max(0.05, self.config.emit_interval_s):
            return ReceiveChunk(time_s=self.processed_duration_s, stats=self._stats())
        self._last_observe_s = self.processed_duration_s
        return self._observe_channels()

    def commit_channel_audio(self, channel_id: int, *, before_s: float) -> None:
        channel = self._tracked_channel(int(channel_id))
        if channel is None:
            return
        previous = max(self._window_start_s, float(channel.audio_trim_before_s or 0.0))
        channel.audio_trim_before_s = max(previous, float(before_s))

    def tracked_channel(self, channel_id: int) -> TrackedChannel | None:
        return self._tracked_channel(channel_id)

    def _tracked_channel(self, channel_id: int) -> TrackedChannel | None:
        for channel in self._channel_tracker.channels:
            if channel.channel_id == channel_id:
                return channel
        return None

    def finish(self, *, final_time_s: float | None = None) -> ReceiveChunk:
        if final_time_s is not None:
            self.processed_duration_s = max(self.processed_duration_s, float(final_time_s))

        # Build the final snapshots while channels are still ACTIVE.  Closing
        # them first would make _channel_signal() return an empty audio window,
        # so the last buffered Morse element/character would never reach the
        # signal and decoder layers.
        final_signals = tuple(self._final_channel_signal(channel) for channel in self._channel_tracker.channels)
        self._channel_tracker.finish(time_s=self.processed_duration_s)
        return ReceiveChunk(
            time_s=self.processed_duration_s,
            channels=final_signals,
            stats=self._stats(),
        )

    def _observe_channels(self) -> ReceiveChunk:
        candidate_signal, _candidate_start_s = self._recent_window(self.config.carrier_window_s)
        observations = self._carrier_observer.observe(candidate_signal)
        tracker_result = self._channel_tracker.update(observations, time_s=self.processed_duration_s)

        return ReceiveChunk(
            time_s=self.processed_duration_s,
            channels=tuple(self._channel_signal(channel) for channel in tracker_result.channels),
            stats=self._stats(),
        )

    def _final_channel_signal(self, channel: TrackedChannel) -> ChannelSignal:
        if channel.state in {ChannelState.DORMANT, ChannelState.DROPPED}:
            return self._channel_signal(channel)

        final_state = ChannelState.DORMANT if channel.channel_started else ChannelState.DROPPED
        if channel.state is ChannelState.ACTIVE and channel.channel_started:
            signal, start_s = self._final_channel_window(channel)
        else:
            signal = np.asarray([], dtype=np.float32)
            start_s = self.processed_duration_s

        return ChannelSignal(
            channel_id=channel.channel_id,
            carrier_hz=channel.carrier_hz,
            start_s=start_s,
            end_s=start_s + len(signal) / self.sample_rate,
            audio_window=signal,
            sample_rate=self.sample_rate,
            state=final_state,
        )

    def _channel_signal(self, channel: TrackedChannel) -> ChannelSignal:
        if channel.state is ChannelState.ACTIVE:
            signal, start_s = self._channel_window(channel)
        else:
            signal = np.asarray([], dtype=np.float32)
            start_s = self.processed_duration_s
        return ChannelSignal(
            channel_id=channel.channel_id,
            carrier_hz=channel.carrier_hz,
            start_s=start_s,
            end_s=start_s + len(signal) / self.sample_rate,
            audio_window=signal,
            sample_rate=self.sample_rate,
            state=channel.state,
        )

    def _recent_window(self, target_s: float) -> tuple[np.ndarray, float]:
        target_s = float(target_s)
        if self.config.max_history_s is not None:
            target_s = min(target_s, float(self.config.max_history_s))
        return self._audio.recent(target_s)

    def _channel_earliest_start_s(self, channel: TrackedChannel) -> float:
        return max(
            self._window_start_s,
            float(channel.audio_trim_before_s or self._window_start_s),
            channel.first_seen_s - max(self.config.history_margin_s, self.config.carrier_window_s),
        )

    def _channel_window(self, channel: TrackedChannel) -> tuple[np.ndarray, float]:
        earliest_start_s = self._channel_earliest_start_s(channel)
        latest_start_s = self.processed_duration_s - self.config.channel_window_s
        start_s = max(earliest_start_s, latest_start_s)
        return self._audio.from_time(start_s)

    def _final_channel_window(self, channel: TrackedChannel) -> tuple[np.ndarray, float]:
        # The live rolling window is intentionally bounded for CPU use, but a
        # finish snapshot should flush the whole retained channel when possible.
        # Otherwise an 8.3 s CQ with an 8.0 s channel window loses its first
        # element and the final dormant decode can become worse than the last
        # active line.
        return self._audio.from_time(self._channel_earliest_start_s(channel))

    def _stats(self) -> ReceivingStats:
        return ReceivingStats(
            processed_duration_s=self.processed_duration_s,
            retained_audio_s=self._audio.duration_s,
            input_rms=self.last_input_rms,
            input_peak=self.last_input_peak,
        )
