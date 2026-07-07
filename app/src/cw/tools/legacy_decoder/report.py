from __future__ import annotations

import json
from dataclasses import asdict
from cw.tools.legacy_decoder.models import DecodeCandidate, DecodeReport

def report_to_json(report: DecodeReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)

def format_decode_report(report: DecodeReport) -> str:
    lines: list[str] = []
    lines.append(
        f"raw={report.path} sample_rate={report.sample_rate} format={report.sample_format} "
        f"channels={report.channels} start_s={report.start_s:.3f} duration_s={report.duration_s:.3f}"
    )
    if report.detected_carriers:
        lines.append("detected carriers:")
        for candidate in report.detected_carriers:
            lines.append(f"  {candidate.carrier_hz:8.1f} Hz rel={candidate.relative_power:5.3f}")
    lines.append("decoded carriers:")
    for carrier in report.carriers:
        lines.append(
            f"  {carrier.carrier_hz:8.1f} Hz conf={carrier.confidence:5.2f} text={carrier.text or '<none>'}"
        )
        if carrier.sessions:
            lines.append("      sessions:")
            for session in carrier.sessions:
                lines.append(
                    f"        s{session.session_id:<2} {session.start_s:8.3f}-{session.end_s:8.3f} "
                    f"conf={session.confidence:4.2f} text={session.text or '<none>'}"
                )
                if not session.candidates:
                    continue
                lines.append("             rank det thr unit_ms wpm score conf evidence text")
                for index, candidate in enumerate(session.candidates, start=1):
                    lines.append("             " + _format_candidate_row(index, candidate))
        elif carrier.candidates:
            lines.append("      rank det thr unit_ms wpm score conf evidence text")
            for index, candidate in enumerate(carrier.candidates, start=1):
                lines.append("      " + _format_candidate_row(index, candidate))
    return "\n".join(lines)

def _format_candidate_row(index: int, candidate: DecodeCandidate) -> str:
    unit_ms = "-" if candidate.unit_s is None else f"{candidate.unit_s * 1000:7.1f}"
    wpm = "-" if candidate.wpm is None else f"{candidate.wpm:5.1f}"
    score = "-" if candidate.quality_score is None else f"{candidate.quality_score:5.1f}"
    return (
        f"{index:>4} {_detector_label(candidate.detector):>4} {candidate.threshold_ratio:>4.2f} {unit_ms:>7} {wpm:>5} "
        f"{score:>5} {candidate.confidence:>4.2f} {candidate.evidence_score:>8.2f} "
        f"{candidate.text or '<none>'}"
    )

def _detector_label(detector: str) -> str:
    if detector == "threshold":
        return "thr"
    if detector == "viterbi":
        return "vit"
    if detector in {"symbol-hmm", "char-hmm"}:
        return "hmm"
    if detector.endswith("-lattice"):
        return "lat"
    return detector[:4]
