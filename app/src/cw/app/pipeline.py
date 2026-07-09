from __future__ import annotations

from dataclasses import dataclass, field

from cw.app.channel_output import ChannelLayerInfo, ChannelOutput, channel_outputs_from_states
from cw.app.config import ProcessingConfig
from cw.app.debug_output import ChannelDebugOutput, channel_debug_output_from_layers
from cw.app.transcript import split_winner_tokens
from cw.decoder.api import DecodeResult
from cw.decoder.bank import DecoderBank
from cw.decoder.run_decoder import RunDecoder
from cw.io.models import AudioBlock
from cw.receiving.models import ChannelSignal, ReceiveChunk, ReceivingStats
from cw.receiving.processor import Receiver
from cw.selection.arbiter import ChannelResultSelector
from cw.selection.debug import ChannelSelectionDebug
from cw.selection.models import ChannelDecodedTexts, ChannelWinner, SelectionInput, TrackDecodedTexts
from cw.signal.models import SignalTrack
from cw.signal.segmenters import SignalSegmenterBank


@dataclass(frozen=True)
class OutputChunk:
    time_s: float
    outputs: tuple[ChannelOutput, ...] = ()
    debug_outputs: tuple[ChannelDebugOutput, ...] = ()
    receiving: ReceiveChunk | None = None
    stats: ReceivingStats = field(default_factory=ReceivingStats)


@dataclass(frozen=True)
class _SegmentedChannel:
    channel: ChannelSignal
    tracks: tuple[SignalTrack, ...]


@dataclass(frozen=True)
class _TrackDecodeJob:
    channel_id: int
    track: SignalTrack


@dataclass(frozen=True)
class _DecodedTrack:
    channel_id: int
    track: SignalTrack
    results: tuple[DecodeResult, ...]


class ProcessingPipeline:
    """Application composition pipeline: wires layers and owns app-only policy.

    The app receives audio input, pushes it through receiving/signal/decoder/
    selection, then turns the selected current tokens into JSON-ready channel
    output.  Stable-prefix splitting is not a separate memory layer: it marks
    the current winner and feeds the receiver an audio trim point for the next
    iteration.
    """

    def __init__(self, sample_rate: int, config: ProcessingConfig, *, debug: bool = False) -> None:
        self.config = config
        self.receiver = Receiver(sample_rate, config)
        self.segmenters = SignalSegmenterBank.default(config)
        self.decoders = DecoderBank((RunDecoder(config),))
        self.selector = ChannelResultSelector(config=config)
        self.debug = debug

    @property
    def processed_duration_s(self) -> float:
        return self.receiver.processed_duration_s

    @property
    def last_input_rms(self) -> float:
        return self.receiver.last_input_rms

    @property
    def last_input_peak(self) -> float:
        return self.receiver.last_input_peak

    def close(self) -> None:
        return None

    def push(self, block: AudioBlock) -> OutputChunk:
        receive_chunk = self.receiver.push(block)
        return self._process_receive_chunk(receive_chunk)

    def finish(self, *, final_time_s: float | None = None) -> OutputChunk:
        receive_chunk = self.receiver.finish(final_time_s=final_time_s)
        return self._process_receive_chunk(receive_chunk)

    def _process_receive_chunk(self, receive_chunk: ReceiveChunk) -> OutputChunk:
        segmented_channels = self._segment_channels(receive_chunk.channels)
        decoded_channels, debug_track_results_by_channel = self._decode_segmented_channels(segmented_channels)
        selection_input = SelectionInput(channels=tuple(decoded_channels))
        selected, selection_debug = self.selector.select_with_debug(selection_input, time_s=receive_chunk.time_s)
        force_trim_before_s = self.receiver.history_force_commit_before_s()
        channel_winners = self._split_selected_winners(receive_chunk, selected.winners, force_trim_before_s=force_trim_before_s)
        self.receiver.enforce_audio_history_limit()
        layer_info_by_channel = self._layer_infos(receive_chunk, segmented_channels, decoded_channels, selection_debug)
        outputs = channel_outputs_from_states(receive_chunk, channel_winners, layer_info_by_channel)
        debug_outputs = self._debug_outputs(receive_chunk, debug_track_results_by_channel, selection_debug) if self.debug else ()
        return OutputChunk(
            time_s=receive_chunk.time_s,
            outputs=outputs,
            debug_outputs=debug_outputs,
            receiving=receive_chunk,
            stats=receive_chunk.stats,
        )

    def _split_selected_winners(
        self,
        receive_chunk: ReceiveChunk,
        winners: tuple[ChannelWinner, ...],
        *,
        force_trim_before_s: float | None = None,
    ) -> tuple[ChannelWinner, ...]:
        channels_by_id = {channel.channel_id: channel for channel in receive_chunk.channels}
        output: list[ChannelWinner] = []
        for winner in winners:
            signal = channels_by_id.get(winner.channel_id)
            if signal is None:
                output.append(winner)
                continue
            forced_floor = force_trim_before_s if force_trim_before_s is not None and signal.start_s < force_trim_before_s else None
            split_winner, trim_before_s = split_winner_tokens(
                signal,
                winner,
                self.config,
                force_trim_before_s=forced_floor,
            )
            if trim_before_s is not None:
                self.receiver.trim_channel_audio_before(winner.channel_id, before_s=trim_before_s)
            output.append(split_winner)
        return tuple(output)

    def _segment_channels(self, channels: tuple[ChannelSignal, ...]) -> tuple[_SegmentedChannel, ...]:
        if not channels:
            return ()
        return self._map_items(self._segment_one_channel, channels)

    def _segment_one_channel(self, channel: ChannelSignal) -> _SegmentedChannel:
        return _SegmentedChannel(channel=channel, tracks=self.segmenters.segment_channel(channel))


    def _layer_infos(
        self,
        receive_chunk: ReceiveChunk,
        segmented_channels: tuple[_SegmentedChannel, ...],
        decoded_channels: list[ChannelDecodedTexts],
        selection_debug,
    ) -> dict[int, ChannelLayerInfo]:
        segmented_by_channel = {item.channel.channel_id: item for item in segmented_channels}
        decoded_by_channel = {item.channel_id: item for item in decoded_channels}
        selection_by_channel = {item.channel_id: item for item in selection_debug.channels}
        infos: dict[int, ChannelLayerInfo] = {}
        for channel in receive_chunk.channels:
            segmented = segmented_by_channel.get(channel.channel_id)
            decoded = decoded_by_channel.get(channel.channel_id)
            selection = selection_by_channel.get(channel.channel_id)
            tracks = () if segmented is None else segmented.tracks
            unknown_ratios = [track.unknown_ratio for track in tracks]
            longest_mark_s = max(
                (run.duration_s for track in tracks for run in track.runs if run.state.value == "mark"),
                default=0.0,
            )
            decoder_answers = 0
            if decoded is not None:
                decoder_answers = sum(
                    len(result.answers)
                    for track in decoded.tracks
                    for result in track.results
                )
            selected_group = None
            if selection is not None:
                selected_group = next((group for group in selection.groups if group.selected), None)
            infos[channel.channel_id] = ChannelLayerInfo(
                receiving_audio_s=channel.duration_s if channel.has_audio else 0.0,
                signal_tracks=len(tracks),
                signal_best_unknown_ratio=min(unknown_ratios) if unknown_ratios else None,
                signal_longest_mark_s=longest_mark_s,
                decoder_answers=decoder_answers,
                selection_groups=0 if selection is None else len(selection.groups),
                selection_support=0 if selected_group is None else selected_group.support_count,
                selected=selected_group is not None,
            )
        return infos

    def _decode_segmented_channels(
        self,
        segmented_channels: tuple[_SegmentedChannel, ...],
    ) -> tuple[list[ChannelDecodedTexts], dict[int, list[tuple[SignalTrack, tuple[DecodeResult, ...]]]]]:
        jobs: list[_TrackDecodeJob] = []
        for segmented in segmented_channels:
            for track in segmented.tracks:
                jobs.append(_TrackDecodeJob(channel_id=segmented.channel.channel_id, track=track))

        decoded_tracks = self._decode_tracks(jobs) if jobs else ()
        decoded_by_channel: dict[int, list[_DecodedTrack]] = {}
        for decoded in decoded_tracks:
            decoded_by_channel.setdefault(decoded.channel_id, []).append(decoded)

        decoded_channels: list[ChannelDecodedTexts] = []
        debug_track_results_by_channel: dict[int, list[tuple[SignalTrack, tuple[DecodeResult, ...]]]] = {}
        for segmented in segmented_channels:
            channel_decoded_tracks = decoded_by_channel.get(segmented.channel.channel_id, [])
            decoded_channels.append(
                ChannelDecodedTexts(
                    channel_id=segmented.channel.channel_id,
                    carrier_hz=segmented.channel.carrier_hz,
                    tracks=tuple(
                        TrackDecodedTexts(
                            analyzer=decoded.track.analyzer,
                            results=decoded.results,
                            unknown_ratio=decoded.track.unknown_ratio,
                        )
                        for decoded in channel_decoded_tracks
                    ),
                )
            )
            if self.debug:
                debug_track_results_by_channel[segmented.channel.channel_id] = [
                    (decoded.track, decoded.results) for decoded in channel_decoded_tracks
                ]
        return decoded_channels, debug_track_results_by_channel

    def _decode_tracks(self, jobs: list[_TrackDecodeJob]) -> tuple[_DecodedTrack, ...]:
        if not jobs:
            return ()
        return self._map_items(self._decode_one_track, jobs)

    def _decode_one_track(self, job: _TrackDecodeJob) -> _DecodedTrack:
        return _DecodedTrack(
            channel_id=job.channel_id,
            track=job.track,
            results=self.decoders.decode(job.track),
        )

    def _map_items(self, func, items) -> tuple:
        # Keep the live pipeline single-threaded after the reader/capture stage.
        # The expensive part is reduced by feeding stable-prefix trim points back
        # into receiving, not by Python worker threads.
        return tuple(func(item) for item in tuple(items))

    def _debug_outputs(
        self,
        receive_chunk: ReceiveChunk,
        track_results_by_channel: dict[int, list[tuple[SignalTrack, tuple[DecodeResult, ...]]]],
        selection_debug,
    ) -> tuple[ChannelDebugOutput, ...]:
        selection_by_channel: dict[int, ChannelSelectionDebug] = {}
        if selection_debug is not None:
            selection_by_channel = {channel.channel_id: channel for channel in selection_debug.channels}
        return tuple(
            channel_debug_output_from_layers(
                time_s=receive_chunk.time_s,
                channel=channel,
                tracks_with_results=track_results_by_channel.get(channel.channel_id, ()),
                selection=selection_by_channel.get(channel.channel_id),
            )
            for channel in receive_chunk.channels
        )
