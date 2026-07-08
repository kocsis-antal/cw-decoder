from __future__ import annotations

import json

import numpy as np

from cw.app.jsonl import channel_output_to_json
from cw.app.channel_output import ChannelOutput
from cw.decoder.tokens import char_token, gap_token
from cw.io.array_source import ArrayAudioSource
from cw.io.pcm import decode_raw_pcm
from cw.app.config import ProcessingConfig
from cw.receiving.processor import Receiver
from cw.signal.models import SignalRun, SignalState
from cw.signal.segmenters import SignalSegmenterBank
from cw.decoder.run_decoder import RunDecoder
from cw.selection.models import ChannelDecodedTexts, SelectionInput, TrackDecodedTexts
from cw.selection.arbiter import ChannelResultSelector
from cw.ui.dashboard import _channel_output_from_dict


def test_channel_output_json_roundtrip() -> None:
    output = ChannelOutput(
        channel_id=3,
        carrier_hz=702.5,
        state="active",
        tokens=(char_token("C"), char_token("Q"), gap_token("word_gap"), char_token("D"), char_token("E")),
        stable_token_count=3,
    )
    encoded = channel_output_to_json(output)
    payload = json.loads(encoded)
    assert payload["channel_id"] == 3
    assert payload["carrier_hz"] == 702.5
    assert payload["state"] == "active"
    assert set(payload) == {"channel_id", "carrier_hz", "state", "tokens"}
    assert payload["tokens"][2] == {"kind": "word_gap", "stable": True}
    parsed = _channel_output_from_dict(payload)
    assert parsed is not None
    assert parsed.text == "CQ [DE]"


def test_raw_pcm_decode_s16le() -> None:
    raw = np.array([0, 32767, -32768], dtype="<i2").tobytes()
    samples = decode_raw_pcm(raw, "s16le", 1)
    assert samples.dtype == np.float32
    assert samples[0] == 0.0
    assert samples[1] > 0.99
    assert samples[2] == -1.0


def test_receiving_accepts_audio_blocks_and_outputs_channel_signals() -> None:
    sample_rate = 8000
    t = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    signal = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    source = ArrayAudioSource(signal, sample_rate, block_ms=20.0)
    receiver = Receiver(sample_rate, ProcessingConfig(max_tracks=1, emit_interval_s=0.1))
    chunks = []
    for block in source:
        chunks.append(receiver.push(block))
    chunks.append(receiver.finish())
    assert all(chunk.time_s >= 0 for chunk in chunks)
    assert any(chunk.channels for chunk in chunks)


def test_signal_track_has_only_signal_runs_and_unknown_time_ratio() -> None:
    from dataclasses import fields
    from cw.signal.models import SignalRun, SignalState, SignalTrack

    assert {field.name for field in fields(SignalTrack)} == {"analyzer", "runs", "unknown_ratio"}

    track = SignalTrack(
        analyzer="unit-test",
        runs=(
            SignalRun(SignalState.MARK, 0.1),
            SignalRun(SignalState.UNKNOWN, 0.1),
            SignalRun(SignalState.SPACE, 0.2),
        ),
        unknown_ratio=0.25,
    )

    assert track.unknown_ratio == 0.25
    assert not hasattr(track, "confidence")
    assert not hasattr(track, "metadata")


def test_signal_layer_can_generate_multiple_tracks_from_channel() -> None:
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    keying = ((t % 0.20) < 0.10).astype(np.float32)
    samples = (0.2 * keying * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    source = ArrayAudioSource(samples, sample_rate, block_ms=1000.0)
    receiver = Receiver(sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1))
    chunk = receiver.push(next(iter(source)))
    config = ProcessingConfig(
        signal_threshold_ratios=(0.25, 0.35, 0.45),
        signal_distribution_acceptance_probabilities=(0.70, 0.90),
    )
    bank = SignalSegmenterBank.default(config)
    tracks = bank.segment_channel(chunk.channels[0])

    assert len(tracks) == 5
    assert {track.analyzer for track in tracks} == {
        "threshold_activity:threshold=0.25",
        "threshold_activity:threshold=0.35",
        "threshold_activity:threshold=0.45",
        "energy_distribution:p=0.70",
        "energy_distribution:p=0.90",
    }
    assert all(not hasattr(track, "channel") for track in tracks)
    assert all(not hasattr(track, "signal") for track in tracks)
    assert any(track.runs for track in tracks)
    assert all(0.0 <= track.unknown_ratio <= 1.0 for track in tracks)
    assert all(run.state in {SignalState.MARK, SignalState.SPACE, SignalState.UNKNOWN} for track in tracks for run in track.runs)
    assert all(run.duration_s > 0 for track in tracks for run in track.runs)


def test_signal_segmenter_uses_signal_thresholds_not_decoder_thresholds() -> None:
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    keying = ((t % 0.20) < 0.10).astype(np.float32)
    samples = (0.2 * keying * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    source = ArrayAudioSource(samples, sample_rate, block_ms=1000.0)
    receiver = Receiver(sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1))
    chunk = receiver.push(next(iter(source)))

    config = ProcessingConfig(
        signal_threshold_ratios=(0.33, 0.44),
        signal_distribution_acceptance_probabilities=(0.80,),
    )

    tracks = SignalSegmenterBank.default(config).segment_channel(chunk.channels[0])

    threshold_tracks = [track.analyzer for track in tracks if track.analyzer.startswith("threshold_activity:")]
    distribution_tracks = [track.analyzer for track in tracks if track.analyzer.startswith("energy_distribution:")]

    assert threshold_tracks == [
        "threshold_activity:threshold=0.33",
        "threshold_activity:threshold=0.44",
    ]
    assert distribution_tracks == ["energy_distribution:p=0.80"]



def test_threshold_activity_states_can_emit_unknown_inside_uncertainty_band() -> None:
    from cw.signal.segmenters import _threshold_activity_states

    states = _threshold_activity_states(
        np.asarray([0.10, 0.45, 0.50, 0.55, 0.90]),
        threshold=0.50,
        noise=0.0,
        signal=1.0,
        uncertainty_ratio=0.10,
    )

    assert states == [
        SignalState.SPACE,
        SignalState.UNKNOWN,
        SignalState.UNKNOWN,
        SignalState.UNKNOWN,
        SignalState.MARK,
    ]

def test_distribution_signal_segmenter_uses_energy_distribution() -> None:
    from cw.receiving.models import ChannelSignal, ChannelState
    from cw.signal.segmenters import DistributionSignalSegmenter

    sample_rate = 8000
    carrier_hz = 700.0
    segment_samples = sample_rate // 4
    amplitudes = np.concatenate(
        (
            np.zeros(segment_samples, dtype=np.float32),
            np.full(segment_samples, 0.30, dtype=np.float32),
            np.zeros(segment_samples, dtype=np.float32),
            np.full(segment_samples, 0.30, dtype=np.float32),
        )
    )
    t = np.arange(len(amplitudes), dtype=np.float32) / sample_rate
    samples = (amplitudes * np.sin(2 * np.pi * carrier_hz * t)).astype(np.float32)
    channel = ChannelSignal(
        channel_id=1,
        carrier_hz=carrier_hz,
        start_s=0.0,
        end_s=len(samples) / sample_rate,
        audio_window=samples,
        sample_rate=sample_rate,
        state=ChannelState.ACTIVE,
    )
    config = ProcessingConfig(signal_distribution_acceptance_probabilities=(0.70, 0.90))

    tracks = DistributionSignalSegmenter(config).segment(channel)

    assert [track.analyzer for track in tracks] == ["energy_distribution:p=0.70", "energy_distribution:p=0.90"]
    assert all(0.0 <= track.unknown_ratio <= 1.0 for track in tracks)
    for track in tracks:
        states = {run.state for run in track.runs}
        assert SignalState.MARK in states
        assert SignalState.SPACE in states


def test_distribution_activity_states_can_emit_unknown_for_uncertain_frames() -> None:
    from cw.signal.segmenters import _distribution_activity_states

    states = _distribution_activity_states(np.asarray([0.02, 0.45, 0.55, 0.98]), acceptance_probability=0.90)

    assert states == [SignalState.SPACE, SignalState.UNKNOWN, SignalState.UNKNOWN, SignalState.MARK]

def test_decoder_and_selection_are_separate_from_receiving() -> None:
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    source = ArrayAudioSource(samples, sample_rate, block_ms=1000.0)
    config = ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1)
    receiver = Receiver(sample_rate, config)
    receive_chunk = receiver.push(next(iter(source)))
    channel = receive_chunk.channels[0]
    tracks = SignalSegmenterBank.default(config).segment_channel(channel)
    selection_input = SelectionInput(
        channels=(
            ChannelDecodedTexts(
                channel_id=channel.channel_id,
                carrier_hz=channel.carrier_hz,
                tracks=tuple(
                    TrackDecodedTexts(
                        analyzer=track.analyzer,
                        results=(RunDecoder(config).decode(track),),
                    )
                    for track in tracks[:1]
                ),
            ),
        )
    )
    selection = ChannelResultSelector().select(selection_input, time_s=receive_chunk.time_s)
    assert all(state.channel_id > 0 for state in selection.winners)


def test_processing_pipeline_runs_default_decoder_without_crashing() -> None:
    from cw.app.pipeline import ProcessingPipeline

    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    source = ArrayAudioSource(samples, sample_rate, block_ms=1000.0)
    pipeline = ProcessingPipeline(sample_rate, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1))

    chunk = pipeline.push(next(iter(source)))

    assert chunk.time_s > 0
    assert chunk.receiving is not None


def test_channel_candidate_is_a_channel_output_state() -> None:
    output = ChannelOutput(
        channel_id=4,
        carrier_hz=701.2,
        state="candidate",
    )
    payload = json.loads(channel_output_to_json(output))
    assert payload["channel_id"] == 4
    assert payload["state"] == "candidate"


def test_decoder_does_not_depend_on_offline_analysis_tool() -> None:
    from pathlib import Path

    decoder = Path(__file__).parents[1] / "src" / "cw" / "decoder"
    offenders = []
    for path in decoder.glob("*.py"):
        if "cw.prob_analysis" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_dashboard_renderer_is_channel_snapshot_view_only() -> None:
    from io import StringIO
    from cw.ui.dashboard import HumanDashboardRenderer

    stream = StringIO()
    renderer = HumanDashboardRenderer(stream, use_ansi=False, refresh_interval_s=0.0)
    renderer.start()
    renderer.tick(1.0)
    renderer.emit(ChannelOutput(channel_id=2, carrier_hz=701.25, state="active", text="CQ"))
    renderer.tick(2.0)
    renderer.emit(ChannelOutput(channel_id=3, carrier_hz=812.0, state="candidate", text=""))
    renderer.emit(ChannelOutput(channel_id=3, carrier_hz=812.0, state="dropped", text=""))
    renderer.close()

    rendered = stream.getvalue()
    assert "ch" in rendered
    assert "freq" in rendered
    assert "state" in rendered
    assert "text" in rendered
    assert "CQ" in rendered
    assert "701.2 Hz" in rendered
    assert "session" not in rendered.lower()
    assert "stable" not in rendered.lower()
    assert "unstable" not in rendered.lower()
    assert "score" not in rendered.lower()




def test_dashboard_appends_stable_json_tokens_instead_of_overwriting_with_audio_tail() -> None:
    from io import StringIO
    from cw.ui.dashboard import HumanDashboardRenderer

    stream = StringIO()
    renderer = HumanDashboardRenderer(stream, use_ansi=False, refresh_interval_s=0.0)
    renderer.start()
    renderer.emit(
        ChannelOutput(
            channel_id=1,
            carrier_hz=700.0,
            state="active",
            tokens=(
                char_token("C", start_s=0.0, end_s=0.1),
                char_token("Q", start_s=0.2, end_s=0.3),
                gap_token("word_gap", start_s=0.3, end_s=0.6),
                char_token("D", start_s=0.6, end_s=0.7),
            ),
            stable_token_count=3,
        )
    )
    renderer.emit(
        ChannelOutput(
            channel_id=1,
            carrier_hz=700.0,
            state="active",
            tokens=(
                char_token("Q", start_s=0.2, end_s=0.3),
                gap_token("word_gap", start_s=0.3, end_s=0.6),
                char_token("D", start_s=0.6, end_s=0.7),
                char_token("E", start_s=0.8, end_s=0.9),
            ),
            stable_token_count=4,
        )
    )
    renderer.close()

    rendered = stream.getvalue()
    assert "CQ DE" in rendered
    assert "Q DE" in rendered

def test_channel_output_view_ignores_non_public_json_fields() -> None:
    from io import StringIO
    from cw.ui.dashboard import iter_formatted_jsonl

    lines = ['{"channel_id":4,"carrier_hz":702.0,"state":"active","text":"CQ","score":999,"stable_text":"NO"}']

    rendered = list(iter_formatted_jsonl(lines))

    assert rendered == ["ch4   702.0 Hz active → CQ"]


def test_signal_layer_merges_micro_spaces_inside_marks() -> None:
    from cw.app.config import ProcessingConfig
    from cw.signal.models import SignalRun, SignalState
    from cw.signal.segmenters import _clean_signal_runs

    runs = [
        SignalRun(SignalState.MARK, 0.040),
        SignalRun(SignalState.SPACE, 0.005),
        SignalRun(SignalState.MARK, 0.045),
        SignalRun(SignalState.SPACE, 0.060),
        SignalRun(SignalState.MARK, 0.040),
    ]

    cleaned = _clean_signal_runs(runs, ProcessingConfig(signal_max_cpm=200.0))

    assert cleaned == [
        SignalRun(SignalState.MARK, 0.090),
        SignalRun(SignalState.SPACE, 0.060),
        SignalRun(SignalState.MARK, 0.040),
    ]


def test_decoder_runtime_does_not_split_runs_into_segments() -> None:
    import inspect
    import cw.decoder.run_decoder as run_decoder

    source = inspect.getsource(run_decoder)

    assert "split_runs_into_segments" not in source
    assert "decode_segment_gap" not in source


def test_pipeline_has_no_legacy_publishable_winner_filter() -> None:
    import inspect
    import cw.app.pipeline as pipeline

    source = inspect.getsource(pipeline)

    assert "_publishable_winners" not in source


def test_signal_layer_filters_tracks_with_too_much_unknown_time() -> None:
    from cw.signal.segmenters import _gate_signal_track
    from cw.signal.models import SignalRun, SignalTrack, SignalState

    config = ProcessingConfig(signal_max_unknown_ratio=0.10)
    track = SignalTrack(
        analyzer="unit-test",
        runs=(SignalRun(SignalState.UNKNOWN, 0.2), SignalRun(SignalState.MARK, 0.8)),
        unknown_ratio=0.20,
    )

    assert _gate_signal_track(track, config) is None


def test_signal_layer_does_not_filter_tracks_only_because_they_have_many_runs() -> None:
    from cw.signal.segmenters import _gate_signal_track
    from cw.signal.models import SignalRun, SignalTrack, SignalState

    config = ProcessingConfig()
    track = SignalTrack(
        analyzer="unit-test",
        runs=(
            SignalRun(SignalState.MARK, 0.1),
            SignalRun(SignalState.SPACE, 0.1),
            SignalRun(SignalState.MARK, 0.1),
        ),
        unknown_ratio=0.0,
    )

    assert _gate_signal_track(track, config) is track


def test_signal_layer_rejects_unkeyed_tone_as_not_cw_activity() -> None:
    from cw.receiving.models import ChannelSignal, ChannelState

    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    channel = ChannelSignal(
        channel_id=1,
        carrier_hz=700.0,
        start_s=0.0,
        end_s=1.0,
        audio_window=samples,
        sample_rate=sample_rate,
        state=ChannelState.ACTIVE,
    )

    tracks = SignalSegmenterBank.default(ProcessingConfig()).segment_channel(channel)

    assert tracks == ()


def test_signal_layer_rejects_hiss_without_separable_keying() -> None:
    from cw.receiving.models import ChannelSignal, ChannelState

    sample_rate = 8000
    rng = np.random.default_rng(42)
    samples = rng.normal(0.0, 0.02, sample_rate).astype(np.float32)
    channel = ChannelSignal(
        channel_id=1,
        carrier_hz=700.0,
        start_s=0.0,
        end_s=1.0,
        audio_window=samples,
        sample_rate=sample_rate,
        state=ChannelState.ACTIVE,
    )

    tracks = SignalSegmenterBank.default(ProcessingConfig()).segment_channel(channel)

    assert tracks == ()


def test_signal_speed_gate_drops_implausibly_fast_marks() -> None:
    from cw.signal.segmenters import _clean_signal_runs

    runs = [
        SignalRun(SignalState.MARK, 0.005),
        SignalRun(SignalState.SPACE, 0.010),
        SignalRun(SignalState.MARK, 0.010),
    ]

    cleaned = _clean_signal_runs(runs, ProcessingConfig(signal_max_cpm=200.0))

    assert cleaned == [SignalRun(SignalState.SPACE, 0.025)]


def test_signal_speed_gate_keeps_200_cpm_dot_duration() -> None:
    from cw.signal.segmenters import _clean_signal_runs

    runs = [
        SignalRun(SignalState.MARK, 0.030),
        SignalRun(SignalState.SPACE, 0.030),
        SignalRun(SignalState.MARK, 0.090),
    ]

    cleaned = _clean_signal_runs(runs, ProcessingConfig(signal_max_cpm=200.0))

    assert cleaned == runs
