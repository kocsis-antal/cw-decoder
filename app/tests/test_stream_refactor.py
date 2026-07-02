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


def test_streaming_config_validation_rejects_invalid_history_limit() -> None:
    with pytest.raises(ValueError, match="max_history_s must be positive"):
        validate_streaming_config(StreamingConfig(max_history_s=0))

    with pytest.raises(ValueError, match="max_idle_history_s must be positive"):
        validate_streaming_config(StreamingConfig(max_idle_history_s=-1))


def test_stream_processor_can_hard_limit_retained_history() -> None:
    sample_rate = 8000
    signal = [0.0] * int(sample_rate * 3.0)
    processor = StreamProcessor(sample_rate, StreamingConfig(max_history_s=0.5))

    chunk = sample_rate // 10
    for start in range(0, len(signal), chunk):
        processor.push(signal[start : start + chunk])

    result = processor.finish(final_time_s=3.0)

    assert result.retained_frames < result.frames_processed
    assert result.retained_frames <= 120
    assert result.pruned_frames > 0
