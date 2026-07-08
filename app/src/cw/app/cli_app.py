from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cw.app.debug_output import channel_debug_output_to_json
from cw.app.stream_runner import debug_json_stderr_sink, run_stream_to_dashboard, run_stream_to_json
from cw.ui.dashboard import print_channel_output_view, print_channel_output_view_file
from cw.ui.debug_view import print_debug_output_view, print_debug_output_view_file
from cw.app.config import ProcessingConfig, validate_processing_config
from cw.io.pcm import supported_pcm_formats
from cw.io.raw_stream_source import RawPcmStreamSource
from cw.io.wav_source import WavFileSource


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cw",
        description="CW streaming receiver: audio in, channel JSONL out, optional dashboard view.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stdin_parser = subparsers.add_parser("stream-stdin", help="Read raw PCM from stdin and decode CW")
    _add_raw_input_options(stdin_parser, sample_rate_required=True)
    _add_receiver_options(stdin_parser)
    _add_output_options(stdin_parser)

    raw_parser = subparsers.add_parser("stream-raw-file", help="Read a raw PCM file and decode CW")
    raw_parser.add_argument("path", type=Path)
    _add_raw_input_options(raw_parser, sample_rate_required=True)
    _add_receiver_options(raw_parser)
    _add_output_options(raw_parser)

    wav_parser = subparsers.add_parser("stream-wav", help="Read a WAV/audio file and decode CW")
    wav_parser.add_argument("path", type=Path)
    _add_receiver_options(wav_parser)
    _add_output_options(wav_parser)

    view_parser = subparsers.add_parser("view-output", help="Render cw channel JSONL as the dashboard")
    view_parser.add_argument("path", nargs="?", type=Path, default=None, help="JSONL file; omit or use '-' for stdin")

    debug_view_parser = subparsers.add_parser("view-debug-output", help="Render cw debug JSONL in a readable form")
    debug_view_parser.add_argument("path", nargs="?", type=Path, default=None, help="debug JSONL file; omit or use '-' for stdin")

    args = parser.parse_args()

    if args.command == "view-output":
        if args.path is None or str(args.path) == "-":
            print_channel_output_view(sys.stdin, sys.stdout)
        else:
            print_channel_output_view_file(args.path, sys.stdout)
        return

    if args.command == "view-debug-output":
        if args.path is None or str(args.path) == "-":
            print_debug_output_view(sys.stdin, sys.stdout)
        else:
            print_debug_output_view_file(args.path, sys.stdout)
        return

    config = _build_streaming_config(args)
    validate_processing_config(config)
    source = _build_source(args, config)
    debug_sink, debug_file = _build_debug_sink(args)
    try:
        if args.json_output:
            run_stream_to_json(source, config, stats_interval_s=args.stats_interval_s, debug_sink=debug_sink)
        else:
            run_stream_to_dashboard(source, config, stats_interval_s=args.stats_interval_s, debug_sink=debug_sink)
    finally:
        if debug_file is not None:
            debug_file.close()


def _add_raw_input_options(parser: argparse.ArgumentParser, *, sample_rate_required: bool) -> None:
    parser.add_argument("--sample-rate", type=int, required=sample_rate_required)
    parser.add_argument("--sample-format", choices=supported_pcm_formats(), default="s16le")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--capture-raw", type=Path, default=None)


def _add_receiver_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-block-ms", type=float, default=10.0)
    parser.add_argument("--emit-interval-s", type=float, default=0.50)
    parser.add_argument("--stable-prefix-hold-chars", "--commit-hold-chars", dest="stable_prefix_hold_chars", type=int, default=3, help="keep this many trailing decoded characters tentative while a channel is active")
    parser.add_argument("--stable-prefix-fallback-after-chars", "--commit-fallback-after-chars", dest="stable_prefix_fallback_after_chars", type=int, default=6, help="if no word boundary appears, mark part of a no-space run stable; 0 disables")
    parser.add_argument("--stable-audio-context-chars", "--commit-audio-context-chars", dest="stable_audio_context_chars", type=int, default=2, help="keep this many stable characters in the audio tail as timing context")
    parser.add_argument("--stable-prefix-commit-unresolved", "--commit-unresolved", dest="stable_prefix_commit_unresolved", action="store_true", help="allow decode-error markers to become stable; default keeps them tentative")
    parser.add_argument("--no-stable-prefix", "--no-incremental-commit", dest="no_stable_prefix", action="store_true", help="disable stable-prefix marking and receiving audio trimming")
    parser.add_argument("--min-tone-hz", type=float, default=200.0)
    parser.add_argument("--max-tone-hz", type=float, default=3000.0)
    parser.add_argument("--max-tracks", type=int, default=5)
    parser.add_argument("--channel-match-hz", "--bandwidth-hz", dest="channel_match_hz", type=float, default=40.0, help="normal same-channel tracking tolerance in Hz")
    parser.add_argument("--carrier-window-s", type=float, default=2.0)
    parser.add_argument("--channel-window-s", type=float, default=8.0)
    parser.add_argument("--max-history-s", type=float, default=12.0)
    parser.add_argument("--peak-relative-threshold", type=float, default=0.05)
    parser.add_argument("--carrier-min-snr-db", type=float, default=14.0, help="minimum per-carrier spectral SNR for opening a channel")
    parser.add_argument("--carrier-peak-separation-hz", "--min-separation-hz", dest="carrier_peak_separation_hz", type=float, default=80.0, help="minimum spacing between simultaneous FFT carrier peaks")
    parser.add_argument("--peak-min-separation-hz", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--channel-reacquire-hz", type=float, default=80.0, help="frequency distance used to reacquire the same channel")
    parser.add_argument("--no-alias-suppression", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--signal-threshold-ratios", default="0.25,0.30,0.35,0.42", help="comma-separated signal threshold ratios")
    parser.add_argument("--signal-uncertainty-ratio", type=float, default=0.08)
    parser.add_argument("--signal-distribution-probabilities", default="0.70,0.80,0.90", help="comma-separated posterior acceptance probabilities for distribution-based signal tracks")
    parser.add_argument("--dot-dash-boundary-units", type=float, default=2.0)
    parser.add_argument("--no-adaptive-tone-thresholds", action="store_true", help="disable data-driven ti/ta boundary estimation")
    parser.add_argument("--element-letter-gap-units", type=float, default=2.6)
    parser.add_argument("--no-adaptive-element-letter-gap", action="store_true", help="disable data-driven element/letter gap boundary estimation")
    parser.add_argument("--min-element-letter-gap-units", type=float, default=1.4)
    parser.add_argument("--max-element-letter-gap-units", type=float, default=2.8)
    parser.add_argument("--session-gap-units", type=float, default=14.0, help="gap length, in ti units, decoded as a session_gap token")
    parser.add_argument("--signal-max-cpm", type=float, default=200.0, help="maximum plausible CW speed in characters/minute; marks faster than this are treated as glitches, 0 disables")
    parser.add_argument("--signal-min-keying-separation", type=float, default=1.25, help="minimum low/high envelope separability for accepting a channel as keyed CW")
    parser.add_argument("--signal-max-unknown-ratio", type=float, default=1.0)
    parser.add_argument("--decoder-max-unknown-ratio", type=float, default=0.20)
    parser.add_argument("--decoder-max-unknown-branches", type=int, default=256, help="maximum MARK/SPACE branches generated from UNKNOWN runs in one decoder track")
    parser.add_argument("--selection-min-support-count", type=int, default=1)
    parser.add_argument("--selection-min-family-count", type=int, default=1)
    parser.add_argument(
        "--selection-candidate-families",
        default="energy_distribution",
        help="comma-separated analyzer families allowed to produce selected output; empty allows all families",
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json-output", action="store_true", help="Write channel JSONL output instead of the dashboard")
    parser.add_argument("--debug-json-output", nargs="?", const="-", default=None, help="Write debug JSONL; omit the path or use '-' for stderr")
    parser.add_argument("--stats-interval-s", type=float, default=0.0)


def _build_debug_sink(args: argparse.Namespace):
    destination = getattr(args, "debug_json_output", None)
    if not destination:
        return None, None
    if str(destination) == "-":
        return debug_json_stderr_sink, None
    debug_file = Path(destination).open("w", encoding="utf-8")

    def _write_debug(output) -> None:
        print(channel_debug_output_to_json(output), file=debug_file, flush=True)

    return _write_debug, debug_file


def _build_streaming_config(args: argparse.Namespace) -> ProcessingConfig:
    carrier_peak_separation = args.peak_min_separation_hz if args.peak_min_separation_hz > 0 else args.carrier_peak_separation_hz
    max_history_s = None if args.max_history_s <= 0 else args.max_history_s
    return ProcessingConfig(
        input_block_ms=args.input_block_ms,
        emit_interval_s=args.emit_interval_s,
        stable_prefix_enabled=not args.no_stable_prefix,
        stable_prefix_hold_chars=args.stable_prefix_hold_chars,
        stable_prefix_fallback_after_chars=args.stable_prefix_fallback_after_chars,
        stable_prefix_commit_unresolved=args.stable_prefix_commit_unresolved,
        stable_audio_context_chars=args.stable_audio_context_chars,
        min_tone_hz=args.min_tone_hz,
        max_tone_hz=args.max_tone_hz,
        max_tracks=args.max_tracks,
        channel_match_hz=args.channel_match_hz,
        carrier_window_s=args.carrier_window_s,
        channel_window_s=args.channel_window_s,
        max_history_s=max_history_s,
        peak_relative_threshold=args.peak_relative_threshold,
        carrier_min_snr_db=args.carrier_min_snr_db,
        carrier_peak_separation_hz=carrier_peak_separation,
        alias_suppression=not args.no_alias_suppression,
        channel_reacquire_hz=args.channel_reacquire_hz,
        decoder_max_unknown_ratio=args.decoder_max_unknown_ratio,
        decoder_max_unknown_branches=args.decoder_max_unknown_branches,
        signal_threshold_ratios=_parse_float_csv(args.signal_threshold_ratios),
        signal_uncertainty_ratio=args.signal_uncertainty_ratio,
        signal_distribution_acceptance_probabilities=_parse_float_csv(args.signal_distribution_probabilities),
        dot_dash_boundary_units=args.dot_dash_boundary_units,
        adaptive_tone_thresholds=not args.no_adaptive_tone_thresholds,
        element_letter_gap_units=args.element_letter_gap_units,
        adaptive_element_letter_gap=not args.no_adaptive_element_letter_gap,
        min_element_letter_gap_units=args.min_element_letter_gap_units,
        max_element_letter_gap_units=args.max_element_letter_gap_units,
        session_gap_units=args.session_gap_units,
        signal_max_cpm=args.signal_max_cpm,
        signal_min_keying_separation=args.signal_min_keying_separation,
        signal_max_unknown_ratio=args.signal_max_unknown_ratio,
        selection_min_support_count=args.selection_min_support_count,
        selection_min_family_count=args.selection_min_family_count,
        selection_candidate_families=_parse_str_csv(args.selection_candidate_families),
    )


def _build_source(args: argparse.Namespace, config: ProcessingConfig):
    if args.command == "stream-stdin":
        return RawPcmStreamSource(
            sys.stdin.buffer,
            sample_rate=args.sample_rate,
            sample_format=args.sample_format,
            channels=args.channels,
            block_ms=config.input_block_ms,
            duration_s=args.duration_s,
            capture_raw_path=args.capture_raw,
        )
    if args.command == "stream-raw-file":
        return RawPcmStreamSource(
            args.path.open("rb"),
            sample_rate=args.sample_rate,
            sample_format=args.sample_format,
            channels=args.channels,
            block_ms=config.input_block_ms,
            duration_s=args.duration_s,
            capture_raw_path=args.capture_raw,
        )
    if args.command == "stream-wav":
        return WavFileSource(args.path, block_ms=config.input_block_ms)
    raise ValueError(f"unsupported command: {args.command}")


def _parse_float_csv(value: str) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return ()
    return tuple(float(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_str_csv(value: str) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return ()
    return tuple(part.strip().lower() for part in str(value).split(",") if part.strip())
