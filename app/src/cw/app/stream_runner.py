from __future__ import annotations

import math
import sys
from collections.abc import Callable

from cw.app.debug_output import ChannelDebugOutput, channel_debug_output_to_json
from cw.app.jsonl import channel_output_to_json
from cw.app.channel_output import ChannelOutput
from cw.app.pipeline import ProcessingPipeline, OutputChunk
from cw.io.models import AudioSource
from cw.app.config import ProcessingConfig
from cw.ui.dashboard import HumanDashboardRenderer


def run_stream_to_json(
    source: AudioSource,
    config: ProcessingConfig,
    *,
    stats_interval_s: float = 0.0,
    debug_sink: Callable[[ChannelDebugOutput], None] | None = None,
) -> OutputChunk:
    return _run_stream(
        source,
        config,
        output_sink=lambda output: print(channel_output_to_json(output), flush=True),
        debug_sink=debug_sink,
        stats_interval_s=stats_interval_s,
    )


def run_stream_to_dashboard(
    source: AudioSource,
    config: ProcessingConfig,
    *,
    stats_interval_s: float = 0.0,
    debug_sink: Callable[[ChannelDebugOutput], None] | None = None,
) -> OutputChunk:
    renderer = HumanDashboardRenderer(sys.stdout)
    renderer.start()
    try:
        return _run_stream(
            source,
            config,
            output_sink=renderer.emit,
            tick_sink=renderer.tick,
            debug_sink=debug_sink,
            stats_interval_s=stats_interval_s,
        )
    finally:
        renderer.close()


def debug_json_stderr_sink(output: ChannelDebugOutput) -> None:
    print(channel_debug_output_to_json(output), file=sys.stderr, flush=True)


def _run_stream(
    source: AudioSource,
    config: ProcessingConfig,
    *,
    output_sink: Callable[[ChannelOutput], None],
    tick_sink: Callable[[float], None] | None = None,
    debug_sink: Callable[[ChannelDebugOutput], None] | None = None,
    stats_interval_s: float = 0.0,
) -> OutputChunk:
    pipeline = ProcessingPipeline(source.sample_rate, config, debug=debug_sink is not None)
    try:
        all_outputs: list[ChannelOutput] = []
        all_debug_outputs: list[ChannelDebugOutput] = []
        last_stats_s = 0.0
        interrupted = False
        try:
            for block in source:
                if block.sample_rate != source.sample_rate:
                    raise ValueError("audio block sample_rate changed during stream")
                chunk = pipeline.push(block)
                public_outputs = list(chunk.outputs)
                all_outputs.extend(public_outputs)
                _emit_debug(chunk.debug_outputs, debug_sink, all_debug_outputs)
                for output in public_outputs:
                    try:
                        output_sink(output)
                    except BrokenPipeError:
                        return OutputChunk(
                            time_s=pipeline.processed_duration_s,
                            outputs=tuple(all_outputs),
                            debug_outputs=tuple(all_debug_outputs),
                            stats=chunk.stats,
                        )
                if tick_sink is not None:
                    tick_sink(pipeline.processed_duration_s)
                last_stats_s = _maybe_print_stream_stats(pipeline, len(all_outputs), last_stats_s, interval_s=stats_interval_s)
        except KeyboardInterrupt:
            interrupted = True
            print("stream interrupted", file=sys.stderr)

        final_time_s = (
            source.duration_s
            if not interrupted and source.duration_s is not None
            else pipeline.processed_duration_s
        )
        final_chunk = pipeline.finish(final_time_s=final_time_s)
        final_public_tail = list(final_chunk.outputs)
        all_outputs.extend(final_public_tail)
        _emit_debug(final_chunk.debug_outputs, debug_sink, all_debug_outputs)
        for output in final_public_tail:
            try:
                output_sink(output)
            except BrokenPipeError:
                break
        return OutputChunk(
            time_s=final_chunk.time_s,
            outputs=tuple(all_outputs),
            debug_outputs=tuple(all_debug_outputs),
            stats=final_chunk.stats,
        )
    finally:
        pipeline.close()


def _emit_debug(
    debug_outputs: tuple[ChannelDebugOutput, ...],
    debug_sink: Callable[[ChannelDebugOutput], None] | None,
    all_debug_outputs: list[ChannelDebugOutput],
) -> None:
    if debug_sink is None:
        return
    for debug_output in debug_outputs:
        all_debug_outputs.append(debug_output)
        debug_sink(debug_output)


def _maybe_print_stream_stats(pipeline, emitted_output_count: int, last_stats_s: float, *, interval_s: float) -> float:
    if interval_s <= 0:
        return last_stats_s
    now_s = pipeline.processed_duration_s
    if now_s - last_stats_s < interval_s:
        return last_stats_s
    print(
        "stream "
        f"duration_s={now_s:.1f} "
        f"outputs={emitted_output_count} "
        f"rms_dbfs={_dbfs(pipeline.last_input_rms):.1f} "
        f"peak_dbfs={_dbfs(pipeline.last_input_peak):.1f}",
        file=sys.stderr,
        flush=True,
    )
    return now_s


def _dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(min(value, 1.0))
