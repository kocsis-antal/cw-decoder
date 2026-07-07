from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionPathDebug:
    analyzer: str
    decoder: str
    unresolved_tokens: int


@dataclass(frozen=True)
class SelectionGroupDebug:
    text: str
    unresolved_tokens: int
    support_count: int
    family_count: int
    neighbor_stability: int
    selected: bool = False
    kept_previous: bool = False
    eligible: bool = True
    rejection_reason: str = ""
    paths: tuple[SelectionPathDebug, ...] = ()


@dataclass(frozen=True)
class ChannelSelectionDebug:
    channel_id: int
    selected_text: str = ""
    kept_previous: bool = False
    available_family_count: int = 0
    groups: tuple[SelectionGroupDebug, ...] = ()


@dataclass(frozen=True)
class SelectionDebugChunk:
    time_s: float
    channels: tuple[ChannelSelectionDebug, ...] = ()


__all__ = [
    "SelectionPathDebug",
    "SelectionGroupDebug",
    "ChannelSelectionDebug",
    "SelectionDebugChunk",
]
