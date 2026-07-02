from __future__ import annotations

import sys

from cw.cli_format import (
    _display_text,
    _display_track_text,
    _format_event_line,
    _format_update_line,
    _has_multiple_sessions,
)


def _print_stream_json_events(source, config, args) -> None:
    from cw.stream_events import stream_event_to_json
    from cw.streaming import StreamProcessor

    processor = StreamProcessor(source.sample_rate, config)
    emitted_event_count = 0
    interrupted = False
    last_stats_s = 0.0
    try:
        for block in source:
            if block.sample_rate != source.sample_rate:
                raise ValueError("audio block sample_rate changed during stream")
            chunk = processor.push(block.samples)
            for event in chunk.events:
                print(stream_event_to_json(event), flush=True)
                emitted_event_count += 1
            last_stats_s = _maybe_print_live_stats(args, processor, emitted_event_count, last_stats_s)
    except KeyboardInterrupt:
        interrupted = True
        print("stream interrupted", file=sys.stderr)

    if interrupted and getattr(args, "no_finalize_on_interrupt", False):
        print("stream interrupted; finalization skipped", file=sys.stderr)
        return

    final_time_s = source.duration_s if source.duration_s is not None else processor.processed_duration_s
    try:
        result = processor.finish(final_time_s=final_time_s)
    except KeyboardInterrupt:
        print("stream finalization interrupted; exiting", file=sys.stderr)
        return
    for event in result.events[emitted_event_count:]:
        print(stream_event_to_json(event), flush=True)


def _run_live_human_stream(source, config, args):
    from cw.streaming import StreamProcessor

    processor = StreamProcessor(source.sample_rate, config)
    interrupted = False
    last_stats_s = 0.0
    emitted_event_count = 0
    try:
        for block in source:
            if block.sample_rate != source.sample_rate:
                raise ValueError("audio block sample_rate changed during stream")
            chunk = processor.push(block.samples)
            for update in chunk.updates:
                print(_format_update_line(update), flush=True)
            if args.events:
                for event in chunk.events:
                    print(_format_event_line(event), flush=True)
                    emitted_event_count += 1
            else:
                emitted_event_count += len(chunk.events)
            last_stats_s = _maybe_print_live_stats(args, processor, emitted_event_count, last_stats_s)
    except KeyboardInterrupt:
        interrupted = True
        print("stream interrupted", file=sys.stderr)

    if interrupted and getattr(args, "no_finalize_on_interrupt", False):
        print("stream interrupted; finalization skipped", file=sys.stderr)
        return None

    final_time_s = source.duration_s if source.duration_s is not None else processor.processed_duration_s
    try:
        return processor.finish(final_time_s=final_time_s)
    except KeyboardInterrupt:
        print("stream finalization interrupted; exiting", file=sys.stderr)
        return None


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

def _print_stream_summary(result, args) -> None:
    print(f"duration_s={result.duration_s:.3f}")
    print(
        f"frames_processed={result.frames_processed} "
        f"tracker_frames_processed={result.tracker_frames_processed} "
        f"retained_frames={result.retained_frames} "
        f"pruned_frames={result.pruned_frames} "
        f"active_pruned_frames={result.active_pruned_frames} "
        f"finalized_pruned_frames={result.finalized_pruned_frames}"
    )
    print(f"updates={len(result.updates)}")
    for update in result.updates[: args.updates]:
        print(_format_update_line(update))
    if len(result.updates) > args.updates:
        print(f"... {len(result.updates) - args.updates} more updates")
    if args.events:
        print("events:")
        for event in result.events:
            print(_format_event_line(event))
    print("final:")
    print("track carrier_hz first_seen last_seen hits score unit text")
    for track in result.tracks:
        print(
            f"{track.track_id:>5} "
            f"{track.carrier_hz:>10.1f} "
            f"{track.first_seen_s:>10.3f} "
            f"{track.last_seen_s:>9.3f} "
            f"{track.hits:>4} "
            f"{track.quality.score:>5.1f} "
            f"{track.decoded.unit_s:>4.3f} "
            f"{_display_track_text(track)}"
        )
    if _has_multiple_sessions(result.tracks) or args.events:
        print("sessions:")
        print("channel session first last final reason score unit text")
        for track in result.tracks:
            for session in track.sessions:
                print(
                    f"{track.track_id:>7} "
                    f"{session.session_id:>7} "
                    f"{session.first_seen_s:>5.3f} "
                    f"{session.last_seen_s:>5.3f} "
                    f"{session.final_time_s:>5.3f} "
                    f"{session.final_reason:<13} "
                    f"{session.quality.score:>5.1f} "
                    f"{session.decoded.unit_s:>4.3f} "
                    f"{_display_text(session.decoded.text)}"
                )

