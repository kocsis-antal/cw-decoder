from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    """Selection settings.

    Selection is stateless: it ranks only the current decoded candidates that
    the application layer passes in.  It must not repeat decoder/signal
    unknown-quality gates.  Stable/tentative marking and receiver audio trimming
    are app-level output policy.
    """

    selection_min_support_count: int = 1


def validate_selection_config(config: SelectionConfig) -> None:
    if config.selection_min_support_count < 1:
        raise ValueError("selection_min_support_count must be at least 1")


__all__ = ["SelectionConfig", "validate_selection_config"]
