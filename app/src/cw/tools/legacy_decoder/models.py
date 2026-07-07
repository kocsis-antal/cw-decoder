from __future__ import annotations

from dataclasses import dataclass
from cw.tools.legacy_decoder.base import ClassifiedRun
from cw.tools.legacy_decoder.carrier_detection import CarrierCandidate

@dataclass(frozen=True)
class SignalRun:
    kind: str
    start_s: float
    duration_s: float
    confidence: float
    units: float | None = None
    symbol: str = ""

@dataclass(frozen=True)
class DecodeCandidate:
    carrier_hz: float
    detector: str
    threshold_ratio: float
    threshold: float
    noise_floor: float
    signal_floor: float
    duty_cycle: float
    unit_s: float | None
    wpm: float | None
    text: str
    tokens: tuple[str, ...]
    quality_score: float | None
    confidence: float
    evidence_score: float
    start_s: float
    end_s: float
    runs: tuple[SignalRun, ...]

@dataclass(frozen=True)
class DecodedSession:
    carrier_hz: float
    session_id: int
    start_s: float
    end_s: float
    text: str
    confidence: float
    best: DecodeCandidate | None
    candidates: tuple[DecodeCandidate, ...]

@dataclass(frozen=True)
class CarrierDecodeResult:
    carrier_hz: float
    text: str
    confidence: float
    best: DecodeCandidate | None
    candidates: tuple[DecodeCandidate, ...]
    sessions: tuple[DecodedSession, ...] = ()

@dataclass(frozen=True)
class DecodeReport:
    path: str
    sample_rate: int
    sample_format: str
    channels: int
    start_s: float
    duration_s: float
    detected_carriers: tuple[CarrierCandidate, ...]
    carriers: tuple[CarrierDecodeResult, ...]

@dataclass(frozen=True)
class _SymbolOption:
    symbol: str
    penalty: float

@dataclass(frozen=True)
class _CharHmmState:
    position: int
    tokens: tuple[str, ...]
    classified_runs: tuple[ClassifiedRun, ...]
    cost: float

@dataclass(frozen=True)
class _SymbolHmmState:
    position: int
    tokens: tuple[str, ...]
    current_token: str
    classified_runs: tuple[ClassifiedRun, ...]
    cost: float

@dataclass(frozen=True)
class _LatticeState:
    tokens: tuple[str, ...]
    current_token: str
    classified_runs: tuple[ClassifiedRun, ...]
    penalty: float
