import pytest

from cw.streaming import StreamingConfig, StreamingSTFT, StreamProcessor, simulate_stream
from cw.stream_models import validate_streaming_config


def test_streaming_module_keeps_backwards_compatible_public_imports() -> None:
    assert StreamingConfig().hop_ms == 5.0
    assert StreamingSTFT is not None
    assert callable(simulate_stream)
    assert StreamProcessor is not None


def test_streaming_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="hop_ms must be positive"):
        validate_streaming_config(StreamingConfig(hop_ms=0))


def test_streaming_config_validation_rejects_invalid_active_history_margin() -> None:
    with pytest.raises(ValueError, match="active_history_margin_s must not be negative"):
        validate_streaming_config(StreamingConfig(active_history_margin_s=-0.1))
