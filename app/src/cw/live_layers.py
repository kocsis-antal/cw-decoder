from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np

from cw.nextgen import NextgenCarrierResult, NextgenCandidate, NextgenSession, _detect_carriers_nextgen, decode_signal_carrier_nextgen
from cw.stream_models import StreamingConfig, effective_tracker_frame_ms, effective_tracker_hop_ms


class LiveCarrierDetector:
    """Short-window spectrum detector for the live receiver front-end."""

    def __init__(self, sample_rate: int, config: StreamingConfig) -> None:
        self.sample_rate = int(sample_rate)
        self.config = config

    def detect(self, signal: np.ndarray) -> tuple[float, ...]:
        if len(signal) < max(1, int(self.sample_rate * 0.20)):
            return ()
        detected = _detect_carriers_nextgen(
            signal,
            self.sample_rate,
            min_tone_hz=self.config.min_tone_hz,
            max_tone_hz=self.config.max_tone_hz,
            max_carriers=self.config.max_tracks,
            min_separation_hz=self.config.min_separation_hz,
            relative_threshold=self.config.peak_relative_threshold,
            frame_ms=effective_tracker_frame_ms(self.config),
            hop_ms=effective_tracker_hop_ms(self.config),
        )
        return tuple(float(candidate.carrier_hz) for candidate in detected)


class LiveCarrierDecoder:
    """Per-carrier text decoder.  It sees only a tracked carrier window."""

    def __init__(self, sample_rate: int, config: StreamingConfig) -> None:
        self.sample_rate = int(sample_rate)
        self.config = config

    def decode(
        self,
        signal: np.ndarray,
        *,
        start_s: float,
        carrier_hz: float,
        decode_config: StreamingConfig,
    ) -> NextgenCarrierResult:
        return decode_signal_carrier_nextgen(
            signal,
            self.sample_rate,
            carrier_hz=carrier_hz,
            start_s=start_s,
            threshold_ratios=decode_config.threshold_ratios or (decode_config.threshold_ratio,),
            lowpass_ms=max(5.0, decode_config.frame_ms / 2.5),
            envelope_hop_ms=decode_config.hop_ms,
            session_gap_s=decode_config.min_session_gap_s,
            min_session_evidence_score=0.0,
            config=decode_config,
            max_candidates=decode_config.max_tracks,
            max_candidates_per_session=4,
        )


class _LiveSessionLike(Protocol):
    committed_text: str
    last_candidate_text: str


class LiveSessionHypothesisArbiter:
    """Choose the live-visible candidate from a session's parallel hypotheses.

    Offline decoding may rank by total signal evidence first.  In live use that
    can be the wrong trade-off: two candidates often cover the same keying, but
    the slightly higher-evidence one has a visibly worse timing/text score.  This
    arbiter is intentionally a separate layer so the signal detector, carrier
    tracker, Morse candidate generators, and UI renderer do not need to know
    about each other's compromises.
    """

    def choose(self, decoded: NextgenSession, session: _LiveSessionLike | None = None) -> NextgenSession:
        if not decoded.candidates:
            return decoded
        committed_text = session.committed_text if session is not None else ""
        previous_text = session.last_candidate_text if session is not None else ""
        candidates = tuple(
            sorted(
                decoded.candidates,
                key=lambda candidate: live_candidate_selection_score(
                    candidate,
                    committed_text=committed_text,
                    previous_text=previous_text,
                ),
                reverse=True,
            )
        )
        best = candidates[0]
        if best == decoded.best and candidates == decoded.candidates:
            return decoded
        return replace(decoded, text=best.text, confidence=best.confidence, best=best, candidates=candidates)


def live_candidate_selection_score(
    candidate: NextgenCandidate,
    *,
    committed_text: str = "",
    previous_text: str = "",
) -> float:
    text = candidate.text or ""
    compact = "".join(char for char in text if not char.isspace())
    known_chars = sum(1 for char in compact if char != "?")
    unknowns = compact.count("?")
    punctuation = sum(1 for char in compact if not char.isalnum() and char != "?")
    quality = 18.0 if candidate.quality_score is None else float(candidate.quality_score)

    # Evidence matters, but live text should not sacrifice a much cleaner timing
    # interpretation for a negligible evidence increase.  This is intentionally
    # content-neutral: no CQ/callsign dictionary is encoded here.
    score = (
        candidate.evidence_score * 0.70
        + known_chars * 0.65
        + candidate.confidence * 6.0
        - quality * 0.85
        - unknowns * 2.0
        - punctuation * 0.75
    )

    if committed_text:
        stitched = _stitch_rolling_text(committed_text, text, min_overlap_chars=2)
        compatible = _texts_are_compatible(committed_text, text) or stitched is not None
        if compatible:
            score += 6.0
        elif _compact_common_prefix_len(committed_text, text) >= 3:
            score += 2.0
        else:
            score -= 4.0
    if previous_text:
        if _texts_are_compatible(previous_text, text):
            score += 2.0
        elif _compact_common_prefix_len(previous_text, text) >= 4:
            score += 1.0
    return score


def _texts_are_compatible(left: str, right: str) -> bool:
    return bool(left and right and (left.startswith(right) or right.startswith(left)))


def _compact_common_prefix_len(left: str, right: str) -> int:
    left_compact = "".join(char for char in left if not char.isspace())
    right_compact = "".join(char for char in right if not char.isspace())
    limit = min(len(left_compact), len(right_compact))
    index = 0
    while index < limit and left_compact[index] == right_compact[index]:
        index += 1
    return index


def _compact_with_index_map(text: str) -> tuple[str, list[int]]:
    compact_chars: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        indexes.append(index)
    return "".join(compact_chars), indexes


def _stitch_rolling_text(previous: str, current: str, *, min_overlap_chars: int) -> str | None:
    if not previous or not current:
        return None
    if current.startswith(previous):
        return current
    if previous.startswith(current):
        return None

    previous_compact, _previous_map = _compact_with_index_map(previous)
    current_compact, current_map = _compact_with_index_map(current)
    max_overlap = min(len(previous_compact), len(current_compact))
    for overlap in range(max_overlap, min_overlap_chars - 1, -1):
        if previous_compact[-overlap:] != current_compact[:overlap]:
            continue
        append_start = current_map[overlap - 1] + 1
        suffix = current[append_start:].lstrip()
        if not suffix:
            return None
        separator = "" if previous.endswith((" ", "/")) or suffix.startswith((" ", "/")) else " "
        return f"{previous}{separator}{suffix}".strip()
    return None
