from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from cw.decoder.tokens import DecodeToken, TOKEN_UNKNOWN, token_signature
from cw.morse_table import DECODE_ERROR_MARKER
from cw.selection.config import SelectionConfig
from cw.selection.debug import (
    ChannelSelectionDebug,
    SelectionDebugChunk,
    SelectionGroupDebug,
    SelectionPathDebug,
)
from cw.selection.models import ChannelDecodedTexts, ChannelWinner, SelectionChunk, SelectionInput


@dataclass(frozen=True)
class _DecodedPath:
    text: str
    tokens: tuple[DecodeToken, ...]
    timing_quality: float
    analyzer: str
    decoder: str
    encounter_order: int
    unknown_ratio: float = 0.0


@dataclass(frozen=True)
class _AnalyzerInfo:
    analyzer: str
    family: str
    parameter_name: str | None
    parameter_value: float | None


@dataclass
class _TextGroup:
    text: str
    tokens: tuple[DecodeToken, ...]
    paths: list[_DecodedPath] = field(default_factory=list)
    encounter_order: int = 0


@dataclass(frozen=True)
class _GroupScore:
    unresolved_tokens: int
    support_count: int
    support_score: float
    timing_quality: float
    unknown_penalty_score: float
    final_score: float
    neighbor_stability: int
    encounter_order: int

    def ranking_tuple(self) -> tuple[float, float, int, int]:
        """Rank independent facts once.

        ``final_score`` already contains the selection-owned UNKNOWN penalty.
        Timing quality is a separate decoder-owned physical tie-break.
        """

        return (
            self.final_score,
            -self.timing_quality,
            self.neighbor_stability,
            self.support_count,
        )


@dataclass(frozen=True)
class _ScoredGroup:
    text: str
    tokens: tuple[DecodeToken, ...]
    score: _GroupScore
    best_paths: tuple[_DecodedPath, ...]
    eligible: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class _SelectionOutcome:
    winner: ChannelWinner | None
    debug: ChannelSelectionDebug


class ChannelResultSelector:
    """Chooses one current decoded token stream per channel.

    The selector is intentionally stateless.  It ranks only the current
    candidates provided by the application layer.  Persistent text,
    stable/tentative marking and receiver audio trimming belong to the app, not
    to selection.
    """

    def __init__(self, *, hysteresis: bool = False, config: SelectionConfig | None = None) -> None:
        # ``hysteresis`` is accepted for older tests/call sites, but deliberately
        # ignored.  Selection must not keep previous text.
        self.hysteresis = False
        self.config = config or SelectionConfig()

    def select(self, selection_input: SelectionInput, *, time_s: float) -> SelectionChunk:
        selected, _debug = self.select_with_debug(selection_input, time_s=time_s)
        return selected

    def select_with_debug(self, selection_input: SelectionInput, *, time_s: float) -> tuple[SelectionChunk, SelectionDebugChunk]:
        winners: list[ChannelWinner] = []
        debug_channels: list[ChannelSelectionDebug] = []
        for channel in selection_input.channels:
            outcome = self._select_channel(channel, time_s=time_s)
            if outcome.winner is not None:
                winners.append(outcome.winner)
            debug_channels.append(outcome.debug)
        return (
            SelectionChunk(time_s=time_s, winners=tuple(winners)),
            SelectionDebugChunk(time_s=time_s, channels=tuple(debug_channels)),
        )

    def _select_channel(self, channel: ChannelDecodedTexts, *, time_s: float) -> _SelectionOutcome:
        groups = _groups_by_text(channel)
        analyzer_infos = _analyzer_infos(channel)
        available_track_count = len([track for track in channel.tracks if not track.rejected])
        if not groups:
            return _SelectionOutcome(
                winner=None,
                debug=_selection_debug(
                    channel.channel_id,
                    "",
                    False,
                    [],
                    available_track_count=available_track_count,
                ),
            )

        scored = [_score_group(group, analyzer_infos, self.config) for group in groups.values()]
        eligible_scored = [group for group in scored if group.eligible]
        if not eligible_scored:
            return _SelectionOutcome(
                winner=None,
                debug=_selection_debug(
                    channel.channel_id,
                    "",
                    False,
                    scored,
                    available_track_count=available_track_count,
                ),
            )
        selected = _best_group(eligible_scored)
        if selected is None:
            return _SelectionOutcome(
                winner=None,
                debug=_selection_debug(
                    channel.channel_id,
                    "",
                    False,
                    scored,
                    available_track_count=available_track_count,
                ),
            )

        winner = ChannelWinner(
            channel_id=channel.channel_id,
            carrier_hz=channel.carrier_hz,
            text=selected.text,
            state="selected",
            updated_at_s=time_s,
            tokens=selected.tokens,
        )
        return _SelectionOutcome(
            winner=winner,
            debug=_selection_debug(
                channel.channel_id,
                selected.text,
                False,
                scored,
                available_track_count=available_track_count,
            ),
        )


def _groups_by_text(channel: ChannelDecodedTexts) -> dict[tuple[tuple[str, str], ...], _TextGroup]:
    groups: dict[tuple[tuple[str, str], ...], _TextGroup] = {}
    encounter_order = 0
    for track in channel.tracks:
        for result in track.results:
            for answer in result.answers:
                text = answer.text.strip()
                if not text:
                    continue
                tokens = answer.tokens
                key = token_signature(tokens) if tokens else (("text", text),)
                group = groups.get(key)
                if group is None:
                    group = _TextGroup(text=text, tokens=tokens, encounter_order=encounter_order)
                    groups[key] = group
                group.paths.append(
                    _DecodedPath(
                        text=text,
                        tokens=tokens,
                        timing_quality=_normalized_timing_quality(answer.timing_quality),
                        analyzer=track.analyzer,
                        decoder=result.decoder,
                        encounter_order=encounter_order,
                        unknown_ratio=max(0.0, float(track.unknown_ratio)),
                    )
                )
                encounter_order += 1
    return groups


def _score_group(group: _TextGroup, analyzer_infos: dict[str, _AnalyzerInfo], config: SelectionConfig) -> _ScoredGroup:
    best_paths = _best_paths_by_analyzer(tuple(group.paths))
    unresolved_tokens = _unresolved_token_count(group.tokens, group.text)
    support_count = len(best_paths)
    support_score = float(support_count)
    timing_quality = _mean_timing_quality(best_paths)
    unknown_penalty_score = _unknown_penalty(unresolved_tokens)
    final_score = support_score - unknown_penalty_score
    neighbor_stability = _neighbor_stability(list(best_paths), analyzer_infos)
    score = _GroupScore(
        unresolved_tokens=unresolved_tokens,
        support_count=support_count,
        support_score=support_score,
        timing_quality=timing_quality,
        unknown_penalty_score=unknown_penalty_score,
        final_score=final_score,
        neighbor_stability=neighbor_stability,
        encounter_order=group.encounter_order,
    )
    eligible, reason = _eligibility(score, config)
    return _ScoredGroup(
        text=group.text,
        tokens=group.tokens,
        best_paths=best_paths,
        score=score,
        eligible=eligible,
        rejection_reason=reason,
    )


def _best_paths_by_analyzer(paths: tuple[_DecodedPath, ...]) -> tuple[_DecodedPath, ...]:
    best: dict[str, _DecodedPath] = {}
    for path in paths:
        current = best.get(path.analyzer)
        if current is None or (path.timing_quality, path.encounter_order) < (
            current.timing_quality,
            current.encounter_order,
        ):
            best[path.analyzer] = path
    return tuple(best.values())


def _unresolved_token_count(tokens: tuple[DecodeToken, ...], text: str) -> int:
    # Tokens are the source of truth in the runtime. The text fallback keeps
    # hand-built/test candidates honest without trusting decoder-owned ranking data.
    token_count = sum(1 for token in tokens if token.kind == TOKEN_UNKNOWN)
    return max(token_count, text.count(DECODE_ERROR_MARKER))


def _normalized_timing_quality(value: float) -> float:
    quality = float(value)
    return quality if isfinite(quality) and quality >= 0.0 else float("inf")


def _mean_timing_quality(paths: tuple[_DecodedPath, ...]) -> float:
    if not paths:
        return float("inf")
    return sum(path.timing_quality for path in paths) / len(paths)


def _unknown_penalty(unresolved_tokens: int) -> float:
    # Unknowns are a ranking penalty, not an automatic veto.
    return 0.35 * max(0, int(unresolved_tokens))


def _eligibility(score: _GroupScore, config: SelectionConfig) -> tuple[bool, str]:
    if score.support_count < config.selection_min_support_count:
        return False, f"support_count<{config.selection_min_support_count}"
    return True, ""


def _best_group(groups: list[_ScoredGroup]) -> _ScoredGroup | None:
    best: _ScoredGroup | None = None
    for group in groups:
        if best is None or _is_better(group.score, best.score):
            best = group
    return best


def _is_better(left: _GroupScore, right: _GroupScore) -> bool:
    left_tuple = left.ranking_tuple()
    right_tuple = right.ranking_tuple()
    if left_tuple != right_tuple:
        return left_tuple > right_tuple
    # Stable deterministic fallback: keep the earliest decoded occurrence.
    # This avoids lexicographic or length-based preference.
    return left.encounter_order < right.encounter_order


def _selection_debug(
    channel_id: int,
    selected_text: str,
    kept_previous: bool,
    scored: list[_ScoredGroup],
    *,
    available_track_count: int,
) -> ChannelSelectionDebug:
    groups = tuple(
        SelectionGroupDebug(
            text=group.text,
            unresolved_tokens=group.score.unresolved_tokens,
            support_count=group.score.support_count,
            timing_quality=group.score.timing_quality,
            support_score=group.score.support_score,
            unknown_penalty_score=group.score.unknown_penalty_score,
            final_score=group.score.final_score,
            neighbor_stability=group.score.neighbor_stability,
            selected=bool(selected_text) and group.text == selected_text,
            kept_previous=False,
            eligible=group.eligible,
            rejection_reason=group.rejection_reason,
            paths=tuple(
                SelectionPathDebug(
                    analyzer=path.analyzer,
                    decoder=path.decoder,
                    unresolved_tokens=group.score.unresolved_tokens,
                    timing_quality=path.timing_quality,
                )
                for path in group.best_paths
            ),
        )
        for group in sorted(scored, key=lambda item: item.score.ranking_tuple(), reverse=True)
    )
    return ChannelSelectionDebug(
        channel_id=channel_id,
        selected_text=selected_text,
        kept_previous=False,
        available_track_count=available_track_count,
        groups=groups,
    )


def _analyzer_infos(channel: ChannelDecodedTexts) -> dict[str, _AnalyzerInfo]:
    return {track.analyzer: _parse_analyzer(track.analyzer) for track in channel.tracks}


def _parse_analyzer(analyzer: str) -> _AnalyzerInfo:
    family = _analyzer_family(analyzer)
    parameter_name: str | None = None
    parameter_value: float | None = None
    if ":" in analyzer:
        _family, tail = analyzer.split(":", 1)
        if "=" in tail:
            name, raw_value = tail.split("=", 1)
            raw_value = raw_value.strip()
            try:
                value = float(raw_value)
            except ValueError:
                value = None
            if value is not None and isfinite(value):
                parameter_name = name.strip() or None
                parameter_value = value
    return _AnalyzerInfo(
        analyzer=analyzer,
        family=family,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )


def _analyzer_family(analyzer: str) -> str:
    return analyzer.split(":", 1)[0].strip() or analyzer.strip()


def _neighbor_stability(best_paths: list[_DecodedPath], analyzer_infos: dict[str, _AnalyzerInfo]) -> int:
    supported = {path.analyzer for path in best_paths}
    by_family_param: dict[tuple[str, str], list[_AnalyzerInfo]] = {}
    for info in analyzer_infos.values():
        if info.parameter_name is None or info.parameter_value is None:
            continue
        by_family_param.setdefault((info.family, info.parameter_name), []).append(info)

    stability = 0
    for infos in by_family_param.values():
        ordered = sorted(infos, key=lambda item: (item.parameter_value, item.analyzer))
        for left, right in zip(ordered, ordered[1:]):
            if left.analyzer in supported and right.analyzer in supported:
                stability += 1
    return stability


__all__ = ["ChannelResultSelector"]
