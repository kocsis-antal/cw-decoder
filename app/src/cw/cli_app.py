from __future__ import annotations

import argparse
from pathlib import Path

from cw.cli_commands import run_cli_command
from cw.cli_options import _add_carrier_detection_options, _add_decoder_options, _add_streaming_options


def main() -> None:
    parser = argparse.ArgumentParser(prog="cw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("text")

    decode_parser = subparsers.add_parser("decode-tokens")
    decode_parser.add_argument("tokens", nargs="+")

    decode_wav_parser = subparsers.add_parser("decode-wav")
    decode_wav_parser.add_argument("path", type=Path)
    _add_decoder_options(decode_wav_parser)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("path", type=Path)
    _add_decoder_options(inspect_parser)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("wav_path", type=Path)
    evaluate_parser.add_argument("labels_path", type=Path)
    _add_decoder_options(evaluate_parser)

    contest_parser = subparsers.add_parser("contest")
    contest_parser.add_argument("wav_path", type=Path)
    contest_parser.add_argument("labels_path", type=Path)
    contest_parser.add_argument("--frame-ms", default="10,20,30")
    contest_parser.add_argument("--hop-ms", default="5,10")
    contest_parser.add_argument("--bandwidth-hz", default="20,40,80")
    contest_parser.add_argument("--threshold-ratio", default="0.25,0.35,0.45")
    contest_parser.add_argument("--min-tone-hz", type=float, default=200.0)
    contest_parser.add_argument("--max-tone-hz", type=float, default=2000.0)
    contest_parser.add_argument("--top", type=int, default=10)

    contest_live_parser = subparsers.add_parser("contest-live")
    contest_live_parser.add_argument("wav_path", type=Path)
    contest_live_parser.add_argument("--frame-ms", default="10,20,30")
    contest_live_parser.add_argument("--hop-ms", default="5,10")
    contest_live_parser.add_argument("--bandwidth-hz", default="20,40,80")
    contest_live_parser.add_argument("--threshold-ratio", default="0.25,0.35,0.45")
    contest_live_parser.add_argument("--min-tone-hz", type=float, default=200.0)
    contest_live_parser.add_argument("--max-tone-hz", type=float, default=2000.0)
    contest_live_parser.add_argument("--top", type=int, default=10)
    contest_live_parser.add_argument("--consensus-top", type=int, default=5)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("text")
    benchmark_parser.add_argument("--out-dir", type=Path, default=Path("samples/benchmark"))
    benchmark_parser.add_argument("--presets", default="clean,jitter,drift,noise,straight,field,hard,ugly,brutal")
    benchmark_parser.add_argument("--seeds", default="123,999")
    benchmark_parser.add_argument("--frame-ms", default="10,20,30")
    benchmark_parser.add_argument("--hop-ms", default="5,10")
    benchmark_parser.add_argument("--bandwidth-hz", default="20,40,80")
    benchmark_parser.add_argument("--threshold-ratio", default="0.25,0.35,0.45")
    benchmark_parser.add_argument("--min-tone-hz", type=float, default=200.0)
    benchmark_parser.add_argument("--max-tone-hz", type=float, default=2000.0)
    benchmark_parser.add_argument("--expect", action="store_true")
    benchmark_parser.add_argument("--expected-pass-presets", default="clean,jitter,drift,noise,straight,field,hard,ugly")
    benchmark_parser.add_argument("--allowed-fail-presets", default="brutal")

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("text")
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument(
        "--preset",
        choices=["clean", "jitter", "drift", "noise", "straight", "field", "hard", "ugly", "brutal"],
        default="clean",
    )
    generate_parser.add_argument("--sample-rate", type=int, default=None)
    generate_parser.add_argument("--tone-hz", type=float, default=None)
    generate_parser.add_argument("--wpm", type=float, default=None)
    generate_parser.add_argument("--amplitude", type=float, default=None)
    generate_parser.add_argument("--timing-jitter", type=float, default=None)
    generate_parser.add_argument("--dot-jitter", type=float, default=None)
    generate_parser.add_argument("--dash-jitter", type=float, default=None)
    generate_parser.add_argument("--element-gap-jitter", type=float, default=None)
    generate_parser.add_argument("--letter-gap-jitter", type=float, default=None)
    generate_parser.add_argument("--word-gap-jitter", type=float, default=None)
    generate_parser.add_argument("--dash-ratio", type=float, default=None)
    generate_parser.add_argument("--speed-wobble", type=float, default=None)
    generate_parser.add_argument("--speed-wobble-hz", type=float, default=None)
    generate_parser.add_argument("--frequency-drift-hz", type=float, default=None)
    generate_parser.add_argument("--frequency-wobble-hz", type=float, default=None)
    generate_parser.add_argument("--frequency-wobble-rate-hz", type=float, default=None)
    generate_parser.add_argument("--amplitude-fade", type=float, default=None)
    generate_parser.add_argument("--amplitude-fade-hz", type=float, default=None)
    generate_parser.add_argument("--noise-snr-db", type=float, default=None)
    generate_parser.add_argument("--seed", type=int, default=None)

    generate_multi_parser = subparsers.add_parser("generate-multi")
    generate_multi_parser.add_argument("--out", type=Path, required=True)
    generate_multi_parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Semicolon separated source spec, e.g. id=me;freq=700;preset=field;text=CQ CQ DE YU7NKA",
    )
    generate_multi_parser.add_argument("--sample-rate", type=int, default=8000)
    generate_multi_parser.add_argument("--seed", type=int, default=None)
    generate_multi_parser.add_argument("--normalize-peak", type=float, default=0.95)
    generate_multi_parser.add_argument("--mix-noise-snr-db", type=float, default=None)

    qso_parser = subparsers.add_parser("generate-qso")
    qso_parser.add_argument("--out", type=Path, required=True)
    qso_parser.add_argument("--caller", default="YU7NKA")
    qso_parser.add_argument("--responder", default="YT7MK")
    qso_parser.add_argument("--freq", type=float, default=700.0)
    qso_parser.add_argument("--responder-offset-hz", type=float, default=6.0)
    qso_parser.add_argument("--start", type=float, default=0.0)
    qso_parser.add_argument("--turn-gap-s", type=float, default=1.6)
    qso_parser.add_argument("--caller-preset", default="straight")
    qso_parser.add_argument("--responder-preset", default="straight")
    qso_parser.add_argument("--caller-wpm", type=float, default=20.0)
    qso_parser.add_argument("--responder-wpm", type=float, default=18.0)
    qso_parser.add_argument("--caller-amplitude", type=float, default=0.60)
    qso_parser.add_argument("--responder-amplitude", type=float, default=0.50)
    qso_parser.add_argument("--sample-rate", type=int, default=8000)
    qso_parser.add_argument("--seed", type=int, default=123)
    qso_parser.add_argument("--normalize-peak", type=float, default=0.95)
    qso_parser.add_argument("--mix-noise-snr-db", type=float, default=None)

    detect_carriers_parser = subparsers.add_parser("detect-carriers")
    detect_carriers_parser.add_argument("wav_path", type=Path)
    _add_carrier_detection_options(detect_carriers_parser)

    contest_live_multi_parser = subparsers.add_parser("contest-live-multi")
    contest_live_multi_parser.add_argument("wav_path", type=Path)
    contest_live_multi_parser.add_argument("--frame-ms", default="10,20,30")
    contest_live_multi_parser.add_argument("--hop-ms", default="5,10")
    contest_live_multi_parser.add_argument("--bandwidth-hz", default="20,40,80")
    contest_live_multi_parser.add_argument("--threshold-ratio", default="0.25,0.35,0.45")
    contest_live_multi_parser.add_argument("--top", type=int, default=5)
    contest_live_multi_parser.add_argument("--consensus-top", type=int, default=3)
    _add_carrier_detection_options(contest_live_multi_parser)

    analyze_raw_parser = subparsers.add_parser("analyze-raw")
    analyze_raw_parser.add_argument("raw_path", type=Path)
    analyze_raw_parser.add_argument("--sample-rate", type=int, required=True)
    analyze_raw_parser.add_argument(
        "--sample-format",
        choices=["s16le", "s16be", "s32le", "s32be", "f32le", "f32be", "u8"],
        default="s16le",
    )
    analyze_raw_parser.add_argument("--channels", type=int, default=1)
    analyze_raw_parser.add_argument("--start-s", type=float, default=0.0)
    analyze_raw_parser.add_argument("--duration-s", type=float, default=None)
    analyze_raw_parser.add_argument("--carrier", action="append", type=float, default=[])
    analyze_raw_parser.add_argument("--detect-carriers", type=int, default=5)
    analyze_raw_parser.add_argument("--min-tone-hz", type=float, default=200.0)
    analyze_raw_parser.add_argument("--max-tone-hz", type=float, default=3000.0)
    analyze_raw_parser.add_argument("--min-separation-hz", type=float, default=80.0)
    analyze_raw_parser.add_argument("--peak-relative-threshold", type=float, default=0.10)
    analyze_raw_parser.add_argument("--frame-ms", type=float, default=30.0)
    analyze_raw_parser.add_argument("--hop-ms", type=float, default=5.0)
    analyze_raw_parser.add_argument("--bandwidth-hz", type=float, default=40.0)
    analyze_raw_parser.add_argument("--threshold-ratios", default="0.20,0.25,0.30,0.35,0.40,0.45")
    analyze_raw_parser.add_argument("--adaptive-gap-thresholds", action=argparse.BooleanOptionalAction, default=True)
    analyze_raw_parser.add_argument("--element-letter-gap-units", type=float, default=2.0)
    analyze_raw_parser.add_argument("--default-word-gap-units", type=float, default=7.0)
    analyze_raw_parser.add_argument("--gap-cluster-min-ratio", type=float, default=1.45)
    analyze_raw_parser.add_argument("--gap-cluster-min-delta-units", type=float, default=1.0)
    analyze_raw_parser.add_argument("--gap-cluster-min-lower-count", type=int, default=2)
    analyze_raw_parser.add_argument("--merge-short-gaps-ms", type=float, default=25.0)
    analyze_raw_parser.add_argument("--drop-short-tones-ms", type=float, default=12.0)
    analyze_raw_parser.add_argument("--unit-candidate-spread", type=float, default=0.0)
    analyze_raw_parser.add_argument("--unit-candidate-steps", type=int, default=1)
    analyze_raw_parser.add_argument("--punctuation-penalty", type=float, default=0.0)
    analyze_raw_parser.add_argument("--preview-runs", type=int, default=24)
    analyze_raw_parser.add_argument("--json", action="store_true", help="Print one machine-readable JSON report instead of the human diagnostic table")

    stream_sim_parser = subparsers.add_parser("stream-sim")
    stream_sim_parser.add_argument("wav_path", type=Path)
    stream_sim_parser.add_argument("--input-block-ms", type=float, default=10.0)
    stream_sim_parser.add_argument("--frame-ms", type=float, default=30.0)
    stream_sim_parser.add_argument("--hop-ms", type=float, default=5.0)
    stream_sim_parser.add_argument("--tracker-frame-ms", type=float, default=None)
    stream_sim_parser.add_argument("--tracker-hop-ms", type=float, default=None)
    stream_sim_parser.add_argument("--max-history-s", type=float, default=None)
    stream_sim_parser.add_argument("--max-idle-history-s", type=float, default=None)
    stream_sim_parser.add_argument("--min-tone-hz", type=float, default=200.0)
    stream_sim_parser.add_argument("--max-tone-hz", type=float, default=2000.0)
    stream_sim_parser.add_argument("--bandwidth-hz", type=float, default=40.0)
    stream_sim_parser.add_argument("--threshold-ratio", type=float, default=0.35)
    stream_sim_parser.add_argument("--threshold-ratios", default="")
    stream_sim_parser.add_argument("--adaptive-gap-thresholds", action=argparse.BooleanOptionalAction, default=True)
    stream_sim_parser.add_argument("--element-letter-gap-units", type=float, default=2.0)
    stream_sim_parser.add_argument("--default-word-gap-units", type=float, default=7.0)
    stream_sim_parser.add_argument("--gap-cluster-min-ratio", type=float, default=1.45)
    stream_sim_parser.add_argument("--gap-cluster-min-delta-units", type=float, default=1.0)
    stream_sim_parser.add_argument("--gap-cluster-min-lower-count", type=int, default=2)
    stream_sim_parser.add_argument("--peak-relative-threshold", type=float, default=0.25)
    stream_sim_parser.add_argument("--track-relative-threshold", type=float, default=0.10)
    stream_sim_parser.add_argument("--min-peak-snr-db", type=float, default=0.0)
    stream_sim_parser.add_argument("--min-keying-tone-runs", type=int, default=0)
    stream_sim_parser.add_argument("--min-keying-chars", type=int, default=0)
    stream_sim_parser.add_argument("--min-keying-known-chars", type=int, default=0)
    stream_sim_parser.add_argument("--min-keying-active-duration-s", type=float, default=0.0)
    stream_sim_parser.add_argument("--min-keying-duty-cycle", type=float, default=None)
    stream_sim_parser.add_argument("--max-keying-duty-cycle", type=float, default=None)
    stream_sim_parser.add_argument("--min-keying-unit-s", type=float, default=0.0)
    stream_sim_parser.add_argument("--max-keying-unit-s", type=float, default=None)
    stream_sim_parser.add_argument("--max-keying-score", type=float, default=None)
    stream_sim_parser.add_argument("--reject-et-only-sessions", action=argparse.BooleanOptionalAction, default=False)
    stream_sim_parser.add_argument("--et-only-min-chars", type=int, default=3)
    stream_sim_parser.add_argument("--merge-short-gaps-ms", type=float, default=0.0)
    stream_sim_parser.add_argument("--drop-short-tones-ms", type=float, default=0.0)
    stream_sim_parser.add_argument("--unit-candidate-spread", type=float, default=0.0)
    stream_sim_parser.add_argument("--unit-candidate-steps", type=int, default=1)
    stream_sim_parser.add_argument("--punctuation-penalty", type=float, default=0.0)
    stream_sim_parser.add_argument("--min-separation-hz", type=float, default=80.0)
    stream_sim_parser.add_argument("--peak-min-separation-hz", type=float, default=None)
    stream_sim_parser.add_argument("--track-match-hz", type=float, default=None)
    stream_sim_parser.add_argument("--channel-merge-hz", type=float, default=None)
    stream_sim_parser.add_argument("--max-tracks", type=int, default=5)
    stream_sim_parser.add_argument("--max-track-gap-s", type=float, default=2.0)
    stream_sim_parser.add_argument("--carrier-smoothing", type=float, default=0.20)
    stream_sim_parser.add_argument("--min-track-hits", type=int, default=2)
    stream_sim_parser.add_argument("--emit-interval-s", type=float, default=0.50)
    stream_sim_parser.add_argument("--min-update-score", type=float, default=25.0)
    stream_sim_parser.add_argument("--final-text-regression-margin", type=float, default=10.0)
    stream_sim_parser.add_argument("--max-final-score", type=float, default=30.0)
    stream_sim_parser.add_argument("--disable-final-quality-filter", action="store_true")
    stream_sim_parser.add_argument("--shadow-suppression-hz", type=float, default=None)
    stream_sim_parser.add_argument("--shadow-score-margin", type=float, default=15.0)
    stream_sim_parser.add_argument("--session-gap-units", type=float, default=20.0)
    stream_sim_parser.add_argument("--min-session-gap-s", type=float, default=1.20)
    stream_sim_parser.add_argument("--history-margin-s", type=float, default=0.25)
    stream_sim_parser.add_argument("--active-history-margin-s", type=float, default=None)
    stream_sim_parser.add_argument("--no-prune-finalized-sessions", action="store_true")
    stream_sim_parser.add_argument("--prune-committed-active-sessions", action="store_true")
    stream_sim_parser.add_argument("--raw-updates", action="store_true")
    stream_sim_parser.add_argument("--updates", type=int, default=20)
    stream_sim_parser.add_argument("--events", action="store_true", help="Print channel/session lifecycle events")
    stream_sim_parser.add_argument("--json-events", action="store_true", help="Print channel/session lifecycle events as JSON Lines and suppress human tables")

    stream_stdin_parser = subparsers.add_parser("stream-stdin")
    stream_stdin_parser.add_argument("--sample-rate", type=int, required=True)
    stream_stdin_parser.add_argument(
        "--sample-format",
        choices=["s16le", "s16be", "s32le", "s32be", "f32le", "f32be", "u8"],
        default="s16le",
    )
    stream_stdin_parser.add_argument("--channels", type=int, default=1)
    stream_stdin_parser.add_argument("--duration-s", type=float, default=None)
    stream_stdin_parser.add_argument(
        "--capture-raw",
        type=Path,
        default=None,
        help="Write the exact raw PCM stdin stream to this file while decoding, for reproducible replay",
    )
    stream_stdin_parser.add_argument(
        "--live-stats-interval-s",
        type=float,
        default=0.0,
        help="Print live input progress to stderr every N seconds of processed audio",
    )
    stream_stdin_parser.add_argument(
        "--no-finalize-on-interrupt",
        action="store_true",
        help="Exit immediately on Ctrl+C instead of flushing final stream events",
    )
    _add_streaming_options(stream_stdin_parser)
    stream_stdin_parser.set_defaults(
        max_tone_hz=3000.0,
        min_peak_snr_db=14.0,
        min_keying_tone_runs=3,
        min_keying_chars=2,
        min_keying_known_chars=2,
        min_keying_active_duration_s=0.12,
        min_keying_duty_cycle=0.03,
        max_keying_duty_cycle=0.92,
        min_keying_unit_s=0.03,
        max_keying_score=120.0,
        threshold_ratios="0.20,0.25,0.30,0.35,0.40,0.45",
        reject_et_only_sessions=True,
        merge_short_gaps_ms=25.0,
        drop_short_tones_ms=12.0,
        tracker_frame_ms=80.0,
        tracker_hop_ms=10.0,
        max_history_s=60.0,
        max_idle_history_s=20.0,
        finalization_delay_s=1.0,
    )

    stream_raw_file_parser = subparsers.add_parser("stream-raw-file")
    stream_raw_file_parser.add_argument("raw_path", type=Path)
    stream_raw_file_parser.add_argument("--sample-rate", type=int, required=True)
    stream_raw_file_parser.add_argument(
        "--sample-format",
        choices=["s16le", "s16be", "s32le", "s32be", "f32le", "f32be", "u8"],
        default="s16le",
    )
    stream_raw_file_parser.add_argument("--channels", type=int, default=1)
    stream_raw_file_parser.add_argument("--duration-s", type=float, default=None)
    stream_raw_file_parser.add_argument(
        "--live-stats-interval-s",
        type=float,
        default=0.0,
        help="Print replay progress to stderr every N seconds of processed audio",
    )
    _add_streaming_options(stream_raw_file_parser)
    stream_raw_file_parser.set_defaults(
        max_tone_hz=3000.0,
        min_peak_snr_db=14.0,
        min_keying_tone_runs=3,
        min_keying_chars=2,
        min_keying_known_chars=2,
        min_keying_active_duration_s=0.12,
        min_keying_duty_cycle=0.03,
        max_keying_duty_cycle=0.92,
        min_keying_unit_s=0.03,
        max_keying_score=120.0,
        threshold_ratios="0.20,0.25,0.30,0.35,0.40,0.45",
        reject_et_only_sessions=True,
        merge_short_gaps_ms=25.0,
        drop_short_tones_ms=12.0,
        tracker_frame_ms=80.0,
        tracker_hop_ms=10.0,
        max_history_s=60.0,
        max_idle_history_s=20.0,
        finalization_delay_s=1.0,
    )

    spacing_parser = subparsers.add_parser("spacing-benchmark")
    spacing_parser.add_argument("--text-a", default="CQ CQ DE YU7NKA")
    spacing_parser.add_argument("--text-b", default="CQ CQ DE YT7MK")
    spacing_parser.add_argument("--out-dir", type=Path, default=Path("samples/spacing"))
    spacing_parser.add_argument("--base-freq", type=float, default=700.0)
    spacing_parser.add_argument("--deltas", default="40,60,80,100,120,150")
    spacing_parser.add_argument("--merge-below-hz", type=float, default=60.0)
    spacing_parser.add_argument("--split-from-hz", type=float, default=100.0)
    spacing_parser.add_argument("--preset-a", default="field")
    spacing_parser.add_argument("--preset-b", default="straight")
    spacing_parser.add_argument("--wpm-a", type=float, default=20.0)
    spacing_parser.add_argument("--wpm-b", type=float, default=18.0)
    spacing_parser.add_argument("--amplitude-a", type=float, default=0.60)
    spacing_parser.add_argument("--amplitude-b", type=float, default=0.45)
    spacing_parser.add_argument("--start-b", type=float, default=0.40)
    spacing_parser.add_argument("--sample-rate", type=int, default=8000)
    spacing_parser.add_argument("--seed", type=int, default=123)
    spacing_parser.add_argument("--normalize-peak", type=float, default=0.95)
    spacing_parser.add_argument("--mix-noise-snr-db", type=float, default=None)
    spacing_parser.add_argument("--stream-frame-ms", type=float, default=30.0)
    spacing_parser.add_argument("--stream-hop-ms", type=float, default=5.0)
    spacing_parser.add_argument("--tracker-frame-ms", type=float, default=80.0)
    spacing_parser.add_argument("--tracker-hop-ms", type=float, default=10.0)
    spacing_parser.add_argument("--stream-bandwidth-hz", type=float, default=40.0)
    spacing_parser.add_argument("--stream-threshold-ratio", type=float, default=0.35)
    spacing_parser.add_argument("--peak-relative-threshold", type=float, default=0.25)
    spacing_parser.add_argument("--track-relative-threshold", type=float, default=0.10)
    spacing_parser.add_argument("--min-peak-snr-db", type=float, default=0.0)
    spacing_parser.add_argument("--max-final-score", type=float, default=30.0)
    spacing_parser.add_argument("--disable-final-quality-filter", action="store_true")
    spacing_parser.add_argument("--shadow-suppression-hz", type=float, default=None)
    spacing_parser.add_argument("--shadow-score-margin", type=float, default=15.0)
    spacing_parser.add_argument("--min-separation-hz", type=float, default=80.0)
    spacing_parser.add_argument("--peak-min-separation-hz", type=float, default=None)
    spacing_parser.add_argument("--track-match-hz", type=float, default=None)
    spacing_parser.add_argument("--channel-merge-hz", type=float, default=None)
    spacing_parser.add_argument("--max-tracks", type=int, default=5)
    spacing_parser.add_argument("--expect", action="store_true")

    args = parser.parse_args()
    run_cli_command(args)
