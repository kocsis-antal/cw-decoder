from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable

from cw.cli_format import _format_event_line
from cw.stream_models import StreamChunkResult, StreamEvent


def _print_stream_json_events(source, config, args) -> None:
    from cw.stream_events import stream_event_to_json

    def emit_json_event(event: StreamEvent) -> None:
        # Live JSONL is commonly piped through Docker/tee on Windows and Linux.
        # Print each event as soon as it is produced; otherwise stdout buffering
        # can make the decoder appear silent until Ctrl+C or EOF, while stderr
        # live stats continue to update.
        print(stream_event_to_json(event), flush=True)

    _run_nextgen_stream(source, config, args, json_events=True, event_sink=emit_json_event)


def _run_live_human_stream(source, config, args) -> StreamChunkResult | None:
    return _run_nextgen_stream(source, config, args, json_events=False)


def _run_nextgen_stream(
    source,
    config,
    args,
    *,
    json_events: bool,
    event_sink: Callable[[StreamEvent], None] | None = None,
) -> StreamChunkResult:
    from cw.nextgen_stream import NextgenStreamProcessor

    processor = NextgenStreamProcessor(source.sample_rate, config)
    all_events: list[StreamEvent] = []
    interrupted = False
    last_stats_s = 0.0
    try:
        for block in source:
            if block.sample_rate != source.sample_rate:
                raise ValueError("audio block sample_rate changed during stream")
            chunk = processor.push(block.samples)
            all_events.extend(chunk.events)
            if event_sink is not None:
                for event in chunk.events:
                    event_sink(event)
            elif not json_events and args.events:
                for event in chunk.events:
                    print(_format_event_line(event), flush=True)
            last_stats_s = _maybe_print_live_stats(args, processor, len(all_events), last_stats_s)
    except KeyboardInterrupt:
        interrupted = True
        print("stream interrupted", file=sys.stderr)

    if interrupted and getattr(args, "no_finalize_on_interrupt", False):
        print("stream interrupted; finalization skipped", file=sys.stderr)
        return StreamChunkResult(
            time_s=processor.processed_duration_s,
            events=all_events,
            frames_processed=processor.frames_processed,
            tracker_frames_processed=processor.tracker_frames_processed,
            retained_frames=processor.retained_frames,
            pruned_frames=processor.pruned_frames,
        )

    final_time_s = source.duration_s if source.duration_s is not None else processor.processed_duration_s
    try:
        final_result = processor.finish(final_time_s=final_time_s)
    except KeyboardInterrupt:
        print("stream finalization interrupted; exiting", file=sys.stderr)
        return StreamChunkResult(
            time_s=processor.processed_duration_s,
            events=all_events,
            frames_processed=processor.frames_processed,
            tracker_frames_processed=processor.tracker_frames_processed,
            retained_frames=processor.retained_frames,
            pruned_frames=processor.pruned_frames,
        )
    new_events = final_result.events[len(all_events) :]
    all_events.extend(new_events)
    if event_sink is not None:
        for event in new_events:
            event_sink(event)
    elif not json_events and args.events:
        for event in new_events:
            print(_format_event_line(event), flush=True)
    return StreamChunkResult(
        time_s=final_result.time_s,
        events=all_events,
        frames_processed=final_result.frames_processed,
        tracker_frames_processed=final_result.tracker_frames_processed,
        retained_frames=final_result.retained_frames,
        pruned_frames=final_result.pruned_frames,
    )


def _maybe_print_live_stats(args, processor, emitted_event_count: int, last_stats_s: float) -> float:
    interval_s = float(getattr(args, "live_stats_interval_s", 0.0) or 0.0)
    if interval_s <= 0:
        return last_stats_s

    now_s = processor.processed_duration_s
    if now_s - last_stats_s < interval_s:
        return last_stats_s

    print(
        "live "
        f"duration_s={now_s:.1f} "
        f"frames={processor.frames_processed} "
        f"tracker_frames={processor.tracker_frames_processed} "
        f"retained_frames={processor.retained_frames} "
        f"events={emitted_event_count} "
        f"pruned_frames={processor.pruned_frames} "
        f"rms_dbfs={_dbfs(processor.last_input_rms):.1f} "
        f"peak_dbfs={_dbfs(processor.last_input_peak):.1f}",
        file=sys.stderr,
        flush=True,
    )
    return now_s


def _dbfs(value: float) -> float:
    import math

    if value <= 0:
        return -120.0
    return 20.0 * math.log10(min(value, 1.0))


def _print_stream_summary(result: StreamChunkResult, args) -> None:
    print(f"duration_s={result.time_s:.3f}")
    print(
        f"frames_processed={result.frames_processed} "
        f"tracker_frames_processed={result.tracker_frames_processed} "
        f"retained_frames={result.retained_frames} "
        f"pruned_frames={result.pruned_frames}"
    )
    print(f"events={len(result.events)}")
    finals = [event for event in result.events if event.kind == "SESSION_FINAL" and event.text]
    print(f"final_sessions={len(finals)}")
    print("final:")
    print("channel session carrier_hz score reason text")
    for event in finals:
        score = "-" if event.score is None else f"{event.score:.1f}"
        print(
            f"{event.channel_id:>7} "
            f"{event.session_id or 0:>7} "
            f"{event.carrier_hz:>10.1f} "
            f"{score:>5} "
            f"{event.reason:<13} "
            f"{event.text}"
        )
    if args.events:
        by_kind: dict[str, int] = defaultdict(int)
        for event in result.events:
            by_kind[event.kind] += 1
        print("event_counts:")
        for kind in sorted(by_kind):
            print(f"  {kind}={by_kind[kind]}")
