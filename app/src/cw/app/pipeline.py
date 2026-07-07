from __future__ import annotations

from dataclasses import dataclass, field, replace

from cw.decoder.bank import DecoderBank
from cw.decoder.tokens import TOKEN_UNKNOWN, tokens_to_text
from cw.decoder.run_decoder import RunDecoder
from cw.io.models import AudioBlock
from cw.app.config import ProcessingConfig
from cw.app.debug_output import ChannelDebugOutput, channel_debug_output_from_layers
from cw.app.transcript import ChannelTranscript, absolute_tokens, apply_transcript_update, relative_tokens, uncommitted_tail, winner_from_transcript
from cw.receiving.models import ChannelSignal, ReceiveChunk, ReceivingStats
from cw.receiving.processor import Receiver
from cw.signal.segmenters import SignalSegmenterBank
from cw.selection.arbiter import ChannelResultSelector
from cw.selection.debug import ChannelSelectionDebug
from cw.selection.models import ChannelDecodedTexts, ChannelWinner, SelectionInput, TrackDecodedTexts
from cw.app.channel_output import ChannelOutput, channel_outputs_from_states
from cw.signal.models import SignalTrack
from cw.decoder.api import DecodedText, DecodeResult



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
    """Application composition pipeline: it wires layers, but owns no DSP policy.

    The input reader/capture may run ahead of the decoder, but the receiving,
    signal, decoder and selection layers are intentionally kept ordered.  Live
    CPU use is reduced by committing stable per-channel text prefixes and
    trimming the audio that produced them, not by Python worker threads.
    """

    def __init__(self, sample_rate: int, config: ProcessingConfig, *, debug: bool = False) -> None:
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
        decoded_channels_for_selection = self._decoded_uncommitted_channels(decoded_channels, receive_chunk.channels)

        selection_input = SelectionInput(channels=tuple(decoded_channels_for_selection))
        if self.debug:
            selected, selection_debug = self.selector.select_with_debug(selection_input, time_s=receive_chunk.time_s)
        else:
            selected = self.selector.select(selection_input, time_s=receive_chunk.time_s)
            selection_debug = None
        channel_winners = self._apply_channel_transcripts(receive_chunk, selected.winners)
        outputs = channel_outputs_from_states(receive_chunk, channel_winners)
        debug_outputs = self._debug_outputs(receive_chunk, debug_track_results_by_channel, selection_debug) if self.debug else ()
        return OutputChunk(
            time_s=receive_chunk.time_s,
            outputs=outputs,
            debug_outputs=debug_outputs,
            receiving=receive_chunk,
            stats=receive_chunk.stats,
        )



    def _decoded_uncommitted_channels(
        self,
        decoded_channels: list[ChannelDecodedTexts],
        signals: tuple[ChannelSignal, ...],
    ) -> list[ChannelDecodedTexts]:
        signals_by_id = {signal.channel_id: signal for signal in signals}
        output: list[ChannelDecodedTexts] = []
        for decoded_channel in decoded_channels:
            signal = signals_by_id.get(decoded_channel.channel_id)
            tracked = self.receiver.tracked_channel(decoded_channel.channel_id)
            if signal is None or tracked is None:
                output.append(decoded_channel)
                continue
            transcript = _channel_transcript(tracked)
            output.append(_trim_decoded_channel_to_uncommitted(decoded_channel, signal, transcript))
        return output

    def _apply_channel_transcripts(self, receive_chunk: ReceiveChunk, winners: tuple[ChannelWinner, ...]) -> tuple[ChannelWinner, ...]:
        channels_by_id = {channel.channel_id: channel for channel in receive_chunk.channels}
        winners_by_id = {winner.channel_id: winner for winner in winners}
        output: list[ChannelWinner] = []
        config = self.decoders.decoders[0].config
        for winner in winners:
            signal = channels_by_id.get(winner.channel_id)
            tracked = self.receiver.tracked_channel(winner.channel_id)
            if signal is None or tracked is None:
                output.append(winner)
                continue
            transcript = _channel_transcript(tracked)
            update = apply_transcript_update(transcript, signal, winner, config)
            if update.trim_before_s is not None:
                self.receiver.commit_channel_audio(winner.channel_id, before_s=update.trim_before_s)
            output.append(update.winner)
        for signal in receive_chunk.channels:
            if signal.channel_id in winners_by_id:
                continue
            tracked = self.receiver.tracked_channel(signal.channel_id)
            if tracked is None:
                continue
            transcript = _channel_transcript(tracked)
            transcript_winner = winner_from_transcript(signal, transcript, time_s=receive_chunk.time_s)
            if transcript_winner is not None:
                output.append(transcript_winner)
        return tuple(output)

    def _segment_channels(self, channels: tuple[ChannelSignal, ...]) -> tuple[_SegmentedChannel, ...]:
        if not channels:
            return ()
        return self._map_items(self._segment_one_channel, channels)

    def _segment_one_channel(self, channel: ChannelSignal) -> _SegmentedChannel:
        return _SegmentedChannel(channel=channel, tracks=self.segmenters.segment_channel(channel))

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
        # The expensive part is reduced by committing stable channel prefixes and
        # trimming their audio, not by Python worker threads.
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




def _trim_decoded_channel_to_uncommitted(
    decoded_channel: ChannelDecodedTexts,
    signal: ChannelSignal,
    transcript: ChannelTranscript,
) -> ChannelDecodedTexts:
    tracks: list[TrackDecodedTexts] = []
    for track in decoded_channel.tracks:
        results: list[DecodeResult] = []
        for result in track.results:
            answers: list[DecodedText] = []
            for answer in result.answers:
                if not answer.tokens:
                    answers.append(answer)
                    continue
                absolute = absolute_tokens(answer.tokens, offset_s=signal.start_s)
                tail_absolute = uncommitted_tail(transcript, absolute)
                tail_relative = relative_tokens(tail_absolute, offset_s=signal.start_s)
                text = tokens_to_text(tail_relative)
                if not text:
                    continue
                answers.append(
                    DecodedText(
                        text=text,
                        unresolved_tokens=sum(1 for token in tail_relative if token.kind == TOKEN_UNKNOWN),
                        tokens=tail_relative,
                    )
                )
            if answers:
                results.append(DecodeResult(decoder=result.decoder, answers=tuple(answers)))
        tracks.append(replace(track, results=tuple(results)))
    return replace(decoded_channel, tracks=tuple(tracks))


def _channel_transcript(tracked_channel) -> ChannelTranscript:
    transcript = getattr(tracked_channel, "transcript", None)
    if not isinstance(transcript, ChannelTranscript):
        transcript = ChannelTranscript()
        tracked_channel.transcript = transcript
    return transcript
