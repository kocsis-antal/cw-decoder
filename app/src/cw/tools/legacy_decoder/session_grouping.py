from __future__ import annotations

from cw.tools.legacy_decoder.models import DecodeCandidate, DecodedSession

def _group_candidates_into_sessions(
    candidates: list[DecodeCandidate],
    *,
    carrier_hz: float,
    max_candidates_per_session: int,
    min_session_evidence_score: float,
) -> tuple[DecodedSession, ...]:
    if not candidates:
        return ()
    candidates = sorted(candidates, key=lambda candidate: (candidate.start_s, candidate.end_s, -candidate.evidence_score))
    groups: list[list[DecodeCandidate]] = []
    for candidate in candidates:
        placed = False
        for group in groups:
            if _candidate_overlaps_group(candidate, group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])
    sessions: list[DecodedSession] = []
    for group in groups:
        group.sort(key=lambda candidate: (-candidate.evidence_score, candidate.quality_score or 1e9, -candidate.confidence))
        kept = tuple(group[: max(1, max_candidates_per_session)])
        best = kept[0]
        if best.evidence_score < min_session_evidence_score:
            continue
        sessions.append(
            DecodedSession(
                carrier_hz=round(float(carrier_hz), 3),
                session_id=len(sessions) + 1,
                start_s=round(min(candidate.start_s for candidate in group), 6),
                end_s=round(max(candidate.end_s for candidate in group), 6),
                text=best.text,
                confidence=best.confidence,
                best=best,
                candidates=kept,
            )
        )
    sessions.sort(key=lambda session: (session.start_s, session.end_s))
    # Re-number after sort in case a late-overlapping candidate appended to an older group.
    return tuple(
        DecodedSession(
            carrier_hz=session.carrier_hz,
            session_id=index,
            start_s=session.start_s,
            end_s=session.end_s,
            text=session.text,
            confidence=session.confidence,
            best=session.best,
            candidates=session.candidates,
        )
        for index, session in enumerate(sessions, start=1)
    )

def _candidate_overlaps_group(candidate: DecodeCandidate, group: list[DecodeCandidate]) -> bool:
    group_start = min(item.start_s for item in group)
    group_end = max(item.end_s for item in group)
    overlap = min(candidate.end_s, group_end) - max(candidate.start_s, group_start)
    if overlap <= 0:
        # Thresholds can move segment edges a little.  Close starts likely refer to the same keying burst.
        return min(abs(candidate.start_s - item.start_s) for item in group) <= 0.35
    shorter = max(1e-6, min(candidate.end_s - candidate.start_s, group_end - group_start))
    return overlap / shorter >= 0.25

def _weighted_session_confidence(sessions: tuple[DecodedSession, ...]) -> float:
    weighted = 0.0
    total = 0.0
    for session in sessions:
        duration = max(0.05, session.end_s - session.start_s)
        weighted += session.confidence * duration
        total += duration
    return weighted / total if total > 0 else 0.0

def _unique_candidates(candidates: list[DecodeCandidate]) -> list[DecodeCandidate]:
    best_by_key: dict[tuple[str, tuple[str, ...], int, int], DecodeCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.text,
            candidate.tokens,
            int(round(candidate.start_s * 10)),
            int(round(candidate.end_s * 10)),
        )
        existing = best_by_key.get(key)
        if existing is None or candidate.evidence_score > existing.evidence_score:
            best_by_key[key] = candidate
    return list(best_by_key.values())
