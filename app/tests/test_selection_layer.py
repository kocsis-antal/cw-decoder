from __future__ import annotations

import json

from cw.app.jsonl import channel_output_to_json
from cw.app.channel_output import channel_outputs_from_states
from cw.decoder.api import DecodedText, DecodeResult
from cw.decoder.tokens import char_token, gap_token
from cw.selection.arbiter import ChannelResultSelector
from cw.selection.config import SelectionConfig
from cw.selection.models import ChannelDecodedTexts, SelectionInput, TrackDecodedTexts
from cw.receiving.models import ChannelSignal, ChannelState, ReceiveChunk


def _tokens(text: str):
    output = []
    pending_gap = False
    for ch in text:
        if ch == " ":
            pending_gap = True
            continue
        if pending_gap and output:
            output.append(gap_token("word_gap"))
            pending_gap = False
        output.append(char_token(ch))
    return tuple(output)


def _channel(*tracks: TrackDecodedTexts, channel_id: int = 7) -> ChannelDecodedTexts:
    return ChannelDecodedTexts(channel_id=channel_id, carrier_hz=701.5, tracks=tracks)


def _track(analyzer: str, *answers: DecodedText, decoder: str = "run_decoder") -> TrackDecodedTexts:
    patched = tuple(DecodedText(answer.text, answer.unresolved_tokens, answer.tokens or _tokens(answer.text)) for answer in answers)
    return TrackDecodedTexts(analyzer=analyzer, results=(DecodeResult(decoder=decoder, answers=patched),))


def test_selection_chooses_fewer_unresolved_tokens() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.30", DecodedText(text="C□", unresolved_tokens=1)),
                _track("energy_distribution:p=0.80", DecodedText(text="CQ", unresolved_tokens=0)),
                _track("energy_distribution:p=0.90", DecodedText(text="CQ", unresolved_tokens=0)),
            ),
        )
    )

    chunk = ChannelResultSelector().select(selection, time_s=3.0)

    assert len(chunk.winners) == 1
    assert chunk.winners[0].channel_id == 7
    assert chunk.winners[0].text == "CQ"
    assert not hasattr(chunk.winners[0], "score")


def test_selection_groups_identical_text_support() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="CO", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="CQ", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.35", DecodedText(text="CQ", unresolved_tokens=0)),
            ),
        )
    )

    chunk = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=())).select(selection, time_s=1.0)

    assert chunk.winners[0].text == "CQ"


def test_selection_prefers_model_family_diversity() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="AAAA", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="BBBB", unresolved_tokens=0)),
                _track("energy_distribution:p=0.80", DecodedText(text="BBBB", unresolved_tokens=0)),
            ),
        )
    )

    chunk = ChannelResultSelector().select(selection, time_s=1.0)

    assert chunk.winners[0].text == "BBBB"


def test_selection_uses_parameter_neighbor_stability() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="LEFT", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="RIGHT", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.35", DecodedText(text="RIGHT", unresolved_tokens=0)),
            ),
        )
    )

    chunk = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=())).select(selection, time_s=1.0)

    assert chunk.winners[0].text == "RIGHT"


def test_selection_is_stateless_and_uses_current_encounter_order_as_tiebreaker() -> None:
    selector = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=()))
    first = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.30", DecodedText(text="OLD", unresolved_tokens=0)),
            ),
        )
    )
    selector.select(first, time_s=1.0)

    second = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.30", DecodedText(text="OLD", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.35", DecodedText(text="NEW", unresolved_tokens=0)),
            ),
        )
    )

    chunk = selector.select(second, time_s=2.0)

    assert chunk.winners[0].text == "OLD"


def test_selection_is_stateless_and_prefers_current_lower_error_candidate() -> None:
    selector = ChannelResultSelector()
    first = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.30", DecodedText(text="OLD", unresolved_tokens=0)),
            ),
        )
    )
    selector.select(first, time_s=1.0)

    second = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.30", DecodedText(text="OLD", unresolved_tokens=1)),
                _track("energy_distribution:p=0.80", DecodedText(text="NEW", unresolved_tokens=0)),
                _track("energy_distribution:p=0.90", DecodedText(text="NEW", unresolved_tokens=0)),
            ),
        )
    )

    chunk = selector.select(second, time_s=2.0)

    assert chunk.winners[0].text == "NEW"


def test_selection_winner_is_merged_into_channel_update_json() -> None:
    selected = ChannelResultSelector().select(
        SelectionInput(
            channels=(
                _channel(
                    _track("energy_distribution:p=0.80", DecodedText(text="CQ", unresolved_tokens=0)),
                    channel_id=4,
                ),
            )
        ),
        time_s=1.5,
    )
    receive_chunk = ReceiveChunk(
        time_s=1.5,
        channels=(
            ChannelSignal(
                channel_id=4,
                carrier_hz=702.0,
                start_s=0.0,
                end_s=1.5,
                audio_window=__import__("numpy").array([], dtype="float32"),
                sample_rate=8000,
                state=ChannelState.ACTIVE,
            ),
        ),
    )
    outputs = channel_outputs_from_states(receive_chunk, selected.winners)

    assert len(outputs) == 1
    payload = json.loads(channel_output_to_json(outputs[0]))
    assert payload["channel_id"] == 4
    assert payload["carrier_hz"] == 702.0
    assert payload["state"] == "active"
    assert [token["kind"] for token in payload["tokens"]] == ["char", "char"]
    assert "text" not in payload
    assert set(payload) == {"channel_id", "carrier_hz", "state", "tokens"}


def test_selection_treats_family_diversity_as_ranking_not_a_hard_filter() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="NOISE", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="NOISE", unresolved_tokens=0)),
                _track("energy_distribution:p=0.80"),
            ),
        )
    )

    selected, debug = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=())).select_with_debug(selection, time_s=1.0)

    assert len(selected.winners) == 1
    assert selected.winners[0].text == "NOISE"
    assert debug.channels[0].available_family_count == 2
    assert debug.channels[0].groups[0].eligible is True
    assert debug.channels[0].groups[0].rejection_reason == ""


def test_selection_allows_single_family_result_when_only_one_family_is_configured() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="CQ", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="CQ", unresolved_tokens=0)),
            ),
        )
    )

    chunk = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=("threshold_activity",))).select(selection, time_s=1.0)

    assert len(chunk.winners) == 1
    assert chunk.winners[0].text == "CQ"




def test_default_selection_keeps_threshold_debug_but_does_not_select_it() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="CQ", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="CQ", unresolved_tokens=0)),
            ),
        )
    )

    selected, debug = ChannelResultSelector().select_with_debug(selection, time_s=1.0)

    assert selected.winners == ()
    assert debug.channels[0].groups[0].text == "CQ"
    assert debug.channels[0].groups[0].eligible is False
    assert debug.channels[0].groups[0].rejection_reason == "selection_family_not_allowed"

def test_selection_can_require_multiple_supporting_paths() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.25", DecodedText(text="NOISE", unresolved_tokens=0)),
            ),
        )
    )

    selected, debug = ChannelResultSelector(config=SelectionConfig(selection_min_support_count=2, selection_candidate_families=())).select_with_debug(selection, time_s=1.0)

    assert selected.winners == ()
    assert debug.channels[0].groups[0].eligible is False
    assert debug.channels[0].groups[0].rejection_reason == "support_count<2"


def test_selection_is_stateless_and_does_not_hold_previous_when_absent() -> None:
    selector = ChannelResultSelector()
    selector.select(
        SelectionInput(
            channels=(
                _channel(_track("threshold_activity:threshold=0.30", DecodedText(text="OLD", unresolved_tokens=0))),
            )
        ),
        time_s=1.0,
    )

    chunk = selector.select(
        SelectionInput(
            channels=(
                _channel(_track("threshold_activity:threshold=0.30")),
            )
        ),
        time_s=1.5,
    )

    assert chunk.winners == ()


def test_selection_does_not_let_many_same_family_variants_outvote_two_families() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.20", DecodedText(text="NO_GAP", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.25", DecodedText(text="NO_GAP", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.30", DecodedText(text="NO_GAP", unresolved_tokens=0)),
                _track("threshold_activity:threshold=0.35", DecodedText(text="NO_GAP", unresolved_tokens=0)),
                _track("energy_distribution:p=0.80", DecodedText(text="WITH GAP", unresolved_tokens=0)),
                _track("adaptive_timing", DecodedText(text="WITH GAP", unresolved_tokens=0)),
            ),
        )
    )

    selected, debug = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=())).select_with_debug(selection, time_s=1.0)

    assert selected.winners[0].text == "WITH GAP"
    winning_group = next(group for group in debug.channels[0].groups if group.selected)
    assert winning_group.family_count == 2
    assert winning_group.support_count == 2
    assert winning_group.final_score == 2.0


def test_selection_unknown_penalty_is_ranking_not_absolute_veto() -> None:
    selection = SelectionInput(
        channels=(
            _channel(
                _track("threshold_activity:threshold=0.30", DecodedText(text="CLEAN", unresolved_tokens=0)),
                _track("energy_distribution:p=0.80", DecodedText(text="BETTER□", unresolved_tokens=1)),
                _track("adaptive_timing", DecodedText(text="BETTER□", unresolved_tokens=1)),
            ),
        )
    )

    selected, debug = ChannelResultSelector(config=SelectionConfig(selection_candidate_families=())).select_with_debug(selection, time_s=1.0)

    assert selected.winners[0].text == "BETTER□"
    winning_group = next(group for group in debug.channels[0].groups if group.selected)
    assert winning_group.family_count == 2
    assert winning_group.unresolved_tokens == 1
    assert winning_group.unknown_penalty_score > 0
    assert winning_group.final_score > 1.0
