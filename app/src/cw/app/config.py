from __future__ import annotations

from dataclasses import dataclass

from cw.decoder.config import DecoderConfig, validate_decoder_config
from cw.receiving.config import ReceivingConfig, validate_receiving_config
from cw.signal.config import SignalConfig, validate_signal_config
from cw.selection.config import SelectionConfig, validate_selection_config


@dataclass(frozen=True)
class ProcessingConfig(ReceivingConfig, SignalConfig, DecoderConfig, SelectionConfig):
    """Application composition config.

    Individual layers only depend on the fields they own.  The app can keep a
    convenient flat config because it is the composition root.

    Stable-prefix splitting is an app-level output policy: it marks the current
    selected tokens as stable/tentative and gives receiving a trim point for
    audio that no longer needs to be reprocessed.
    """

    stable_prefix_enabled: bool = True
    stable_prefix_hold_chars: int = 3
    stable_prefix_fallback_after_chars: int = 6
    stable_prefix_commit_unresolved: bool = False
    stable_audio_context_chars: int = 2


def validate_processing_config(config: ProcessingConfig) -> None:
    if config.stable_prefix_hold_chars < 0:
        raise ValueError("stable_prefix_hold_chars must not be negative")
    if config.stable_prefix_fallback_after_chars < 0:
        raise ValueError("stable_prefix_fallback_after_chars must not be negative")
    if config.stable_audio_context_chars < 0:
        raise ValueError("stable_audio_context_chars must not be negative")
    validate_receiving_config(config)
    validate_signal_config(config)
    validate_decoder_config(config)
    validate_selection_config(config)
