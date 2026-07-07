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

    Stable channel prefixes can be committed into per-channel memory.  The
    receiver then trims the audio that produced those committed characters and
    only reprocesses the uncertain tail.
    """

    incremental_commit_enabled: bool = True
    commit_hold_chars: int = 3
    commit_fallback_after_chars: int = 6
    commit_unresolved: bool = False
    commit_audio_context_chars: int = 2


def validate_processing_config(config: ProcessingConfig) -> None:
    if config.commit_hold_chars < 0:
        raise ValueError("commit_hold_chars must not be negative")
    if config.commit_fallback_after_chars < 0:
        raise ValueError("commit_fallback_after_chars must not be negative")
    if config.commit_audio_context_chars < 0:
        raise ValueError("commit_audio_context_chars must not be negative")
    validate_receiving_config(config)
    validate_signal_config(config)
    validate_decoder_config(config)
    validate_selection_config(config)
