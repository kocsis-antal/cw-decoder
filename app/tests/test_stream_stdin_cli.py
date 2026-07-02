import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from cw.generator import GeneratorConfig, build_events, render_wave


def test_stream_stdin_cli_accepts_raw_s16le_json_events() -> None:
    config = GeneratorConfig(sample_rate=8000, tone_hz=700.0, wpm=20.0, amplitude=0.7)
    signal = render_wave(build_events("CQ CQ", config), config)
    signal = np.concatenate([signal, np.zeros(int(config.sample_rate * 0.8), dtype=np.float32)])
    raw = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cw.cli",
            "stream-stdin",
            "--sample-rate",
            str(config.sample_rate),
            "--sample-format",
            "s16le",
            "--json-events",
            "--input-block-ms",
            "10",
        ],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=Path.cwd(),
        check=True,
    )

    events = [json.loads(line) for line in completed.stdout.decode().splitlines()]

    assert any(event["type"] == "CHANNEL_STARTED" for event in events)
    assert any(event["type"] == "TEXT_COMMITTED" for event in events)
    assert any(event["type"] == "SESSION_FINAL" and "CQ" in event["text"] for event in events)
