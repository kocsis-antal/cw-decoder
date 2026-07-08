from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    """Selection settings.

    Selection is stateless: it ranks only the current decoded candidates that
    the application layer passes in.  It may require independent support, but it
    must not repeat decoder/signal unknown-quality gates.  Stable/tentative
    marking and receiver audio trimming are app-level output policy.
    """

    selection_min_support_count: int = 1
    selection_min_family_count: int = 1
    # Only these analyzer families may produce the public selected output.
    # Other families still run and remain visible in debug output.  Use an
    # empty tuple to allow every family (useful for experiments/tests).
    selection_candidate_families: tuple[str, ...] = ("energy_distribution",)


def validate_selection_config(config: SelectionConfig) -> None:
    if config.selection_min_support_count < 1:
        raise ValueError("selection_min_support_count must be at least 1")
    if config.selection_min_family_count < 1:
        raise ValueError("selection_min_family_count must be at least 1")
    for family in config.selection_candidate_families:
        if not str(family).strip():
            raise ValueError("selection_candidate_families must not contain empty family names")


__all__ = ["SelectionConfig", "validate_selection_config"]
