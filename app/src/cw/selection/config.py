from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    """Selection settings.

    Selection is stateless: it ranks only the current uncommitted decoded
    candidates that the application layer passes in.  Persistent text and
    tentative tails live in the channel transcript.
    """

    selection_max_unknown_ratio: float = 1.0
    selection_max_unresolved_tokens: int = 999
    selection_min_support_count: int = 1
    selection_min_family_count: int = 1


def validate_selection_config(config: SelectionConfig) -> None:
    if not 0 <= config.selection_max_unknown_ratio <= 1:
        raise ValueError("selection_max_unknown_ratio must be between 0 and 1")
    if config.selection_max_unresolved_tokens < 0:
        raise ValueError("selection_max_unresolved_tokens must not be negative")
    if config.selection_min_support_count < 1:
        raise ValueError("selection_min_support_count must be at least 1")
    if config.selection_min_family_count < 1:
        raise ValueError("selection_min_family_count must be at least 1")


__all__ = ["SelectionConfig", "validate_selection_config"]
