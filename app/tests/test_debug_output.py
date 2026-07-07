from __future__ import annotations

import json

import numpy as np

from cw.app.config import ProcessingConfig
from cw.app.debug_output import channel_debug_output_to_json
from cw.app.jsonl import channel_output_to_json
from cw.app.pipeline import ProcessingPipeline
from cw.decoder.api import DecodedText, DecodeResult
from cw.decoder.tokens import char_token
from cw.io.models import AudioBlock
from cw.receiving.models import ChannelSignal, ChannelState
from cw.selection.debug import ChannelSelectionDebug, SelectionGroupDebug, SelectionPathDebug
from cw.signal.models import SignalRun, SignalState, SignalTrack
from cw.ui.debug_view import iter_formatted_debug_jsonl


def test_public_json_stays_minimal_when_debug_exists() -> None:
    pipeline = ProcessingPipeline(8000, ProcessingConfig(max_tracks=1, min_track_hits=1, emit_interval_s=0.1), debug=True)
    t = np.arange(8000, dtype=np.float32) / 8000
    samples = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)

    chunk = pipeline.push(AudioBlock(samples=samples, sample_rate=8000, start_s=0.0, duration_s=1.0, index=0))

    assert chunk.outputs
    assert chunk.debug_outputs
    public_payload = json.loads(channel_output_to_json(chunk.outputs[0]))
    assert set(public_payload) == {"channel_id", "carrier_hz", "state", "tokens"}


def test_debug_json_contains_signal_decoder_and_selection_details() -> None:
    channel = ChannelSignal(
        channel_id=4,
        carrier_hz=702.0,
        start_s=0.0,
        end_s=1.0,
        audio_window=np.array([], dtype=np.float32),
        sample_rate=8000,
        state=ChannelState.ACTIVE,
    )
    track = SignalTrack(
        analyzer="threshold_activity:threshold=0.30",
        runs=(
            SignalRun(SignalState.MARK, 0.08),
            SignalRun(SignalState.UNKNOWN, 0.03),
            SignalRun(SignalState.SPACE, 0.09),
        ),
        unknown_ratio=0.15,
    )
    selection = ChannelSelectionDebug(
        channel_id=4,
        selected_text="CQ",
        groups=(
            SelectionGroupDebug(
                text="CQ",
                unresolved_tokens=0,
                support_count=2,
                family_count=1,
                neighbor_stability=1,
                selected=True,
                paths=(SelectionPathDebug(analyzer=track.analyzer, decoder="run_decoder", unresolved_tokens=0),),
            ),
        ),
    )
    from cw.app.debug_output import channel_debug_output_from_layers

    debug = channel_debug_output_from_layers(
        time_s=1.0,
        channel=channel,
        tracks_with_results=((track, (DecodeResult(decoder="run_decoder", answers=(DecodedText("CQ", 0, tokens=(char_token("C"), char_token("Q"))),)),)),),
        selection=selection,
    )

    payload = json.loads(channel_debug_output_to_json(debug))

    assert payload["debug"] == "channel"
    assert payload["selected_text"] == "CQ"
    assert payload["signals"][0]["unknown_ratio"] == 0.15
    assert payload["signals"][0]["runs"] == "M80 U30 S90"
    assert payload["signals"][0]["decoders"][0]["answers"][0]["text"] == "CQ"
    assert payload["signals"][0]["decoders"][0]["answers"][0]["tokens"] == [
        {"kind": "char", "value": "C"},
        {"kind": "char", "value": "Q"},
    ]
    assert payload["selection"]["groups"][0]["support_count"] == 2


def test_debug_view_formats_selection_reason() -> None:
    line = json.dumps(
        {
            "debug": "channel",
            "time_s": 1.25,
            "channel_id": 4,
            "carrier_hz": 702.0,
            "state": "active",
            "selected_text": "CQ",
            "signals": [
                {
                    "analyzer": "energy_distribution:p=0.80",
                    "unknown_ratio": 0.04,
                    "runs": "M80 S90",
                    "decoders": [{"decoder": "run_decoder", "answers": [{"text": "CQ", "unresolved_tokens": 0}]}],
                }
            ],
            "selection": {
                "groups": [
                    {
                        "text": "CQ",
                        "unresolved_tokens": 0,
                        "support_count": 3,
                        "family_count": 2,
                        "neighbor_stability": 1,
                        "selected": True,
                    }
                ]
            },
        }
    )

    rendered = list(iter_formatted_debug_jsonl([line]))

    assert rendered[0].startswith("DEBUG t=1.25s ch4 702.0Hz active selected=\"CQ\"")
    assert any("energy_distribution:p=0.80" in item for item in rendered)
    assert any('bad=0 support=3 families=2 neighbors=1' in item for item in rendered)
