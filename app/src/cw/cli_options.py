from __future__ import annotations

import argparse


def _parse_float_csv(value: str | None) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _add_streaming_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-block-ms", type=float, default=10.0)
    parser.add_argument("--frame-ms", type=float, default=30.0)
    parser.add_argument("--hop-ms", type=float, default=5.0)
    parser.add_argument("--tracker-frame-ms", type=float, default=None)
    parser.add_argument("--tracker-hop-ms", type=float, default=None)
    parser.add_argument("--max-history-s", type=float, default=None, help="Hard cap for retained decode frame history; prevents unbounded live memory/CPU growth")
    parser.add_argument("--max-idle-history-s", type=float, default=None, help="Shorter retained-history cap before any channel/session has been recognized")
    parser.add_argument("--min-tone-hz", type=float, default=200.0)
    parser.add_argument("--max-tone-hz", type=float, default=2000.0)
    parser.add_argument("--bandwidth-hz", type=float, default=40.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.35)
    parser.add_argument(
        "--threshold-ratios",
        default="",
        help="Comma separated dynamic threshold candidates. Empty means use only --threshold-ratio.",
    )
    parser.add_argument("--adaptive-gap-thresholds", action=argparse.BooleanOptionalAction, default=True, help="Estimate letter/word gap boundary from session timing instead of using a fixed 5-unit split")
    parser.add_argument("--element-letter-gap-units", type=float, default=2.0, help="Boundary between intra-character and inter-character gaps, in estimated CW units")
    parser.add_argument("--default-word-gap-units", type=float, default=7.0, help="Fallback word-gap boundary when no distinct word-gap cluster is visible")
    parser.add_argument("--gap-cluster-min-ratio", type=float, default=1.45, help="Minimum multiplicative separation required to split letter and word gap clusters")
    parser.add_argument("--gap-cluster-min-delta-units", type=float, default=1.0, help="Minimum absolute gap separation in CW units required for a letter/word split")
    parser.add_argument("--gap-cluster-min-lower-count", type=int, default=2, help="Require at least this many shorter inter-letter candidates before creating a word-gap cluster")
    parser.add_argument("--peak-relative-threshold", type=float, default=0.25)
    parser.add_argument("--track-relative-threshold", type=float, default=0.10)
    parser.add_argument("--min-peak-snr-db", type=float, default=0.0, help="Require each carrier peak to rise this many dB above the per-frame spectral floor")
    parser.add_argument("--min-keying-tone-runs", type=int, default=0, help="Require this many tone runs before a carrier/session is published")
    parser.add_argument("--min-keying-chars", type=int, default=0, help="Require this many decoded non-space characters before a carrier/session is published")
    parser.add_argument("--min-keying-known-chars", type=int, default=0, help="Require this many decoded non-? characters before a carrier/session is published")
    parser.add_argument("--min-keying-active-duration-s", type=float, default=0.0, help="Require this much total keyed tone duration before publishing")
    parser.add_argument("--min-keying-duty-cycle", type=float, default=None, help="Reject sessions with lower keyed duty cycle when set")
    parser.add_argument("--max-keying-duty-cycle", type=float, default=None, help="Reject sessions with higher keyed duty cycle when set")
    parser.add_argument("--min-keying-unit-s", type=float, default=0.0, help="Reject unrealistically fast keying unit estimates below this value")
    parser.add_argument("--max-keying-unit-s", type=float, default=None, help="Reject unrealistically slow keying unit estimates above this value when set")
    parser.add_argument("--max-keying-score", type=float, default=None, help="Reject sessions with worse quality score before publishing when set")
    parser.add_argument("--reject-et-only-sessions", action=argparse.BooleanOptionalAction, default=False, help="Reject sessions made only of repeated E/T characters")
    parser.add_argument("--et-only-min-chars", type=int, default=3, help="Minimum length for the repeated E/T-only rejection")
    parser.add_argument("--merge-short-gaps-ms", type=float, default=0.0, help="Merge short off-gaps inside a tone; useful for live dropout repair")
    parser.add_argument("--drop-short-tones-ms", type=float, default=0.0, help="Drop very short tone spikes before decoding")
    parser.add_argument("--unit-candidate-spread", type=float, default=0.0, help="Try neighbouring unit estimates around the initial WPM guess")
    parser.add_argument("--unit-candidate-steps", type=int, default=1, help="Number of unit hypotheses to score when unit-candidate-spread is set")
    parser.add_argument("--punctuation-penalty", type=float, default=0.0, help="Softly prefer alphanumeric text over punctuation when choosing a unit hypothesis")
    parser.add_argument("--min-separation-hz", type=float, default=80.0)
    parser.add_argument("--peak-min-separation-hz", type=float, default=None)
    parser.add_argument("--track-match-hz", type=float, default=None)
    parser.add_argument("--channel-merge-hz", type=float, default=None)
    parser.add_argument("--max-tracks", type=int, default=5)
    parser.add_argument("--max-track-gap-s", type=float, default=2.0)
    parser.add_argument("--carrier-smoothing", type=float, default=0.20)
    parser.add_argument("--min-track-hits", type=int, default=2)
    parser.add_argument("--emit-interval-s", type=float, default=0.50)
    parser.add_argument("--min-update-score", type=float, default=25.0)
    parser.add_argument("--final-text-regression-margin", type=float, default=10.0, help="Use the last stable live text when final re-decode is this many score points worse")
    parser.add_argument("--max-final-score", type=float, default=30.0)
    parser.add_argument("--disable-final-quality-filter", action="store_true")
    parser.add_argument("--shadow-suppression-hz", type=float, default=None)
    parser.add_argument("--shadow-score-margin", type=float, default=15.0)
    parser.add_argument("--session-gap-units", type=float, default=20.0)
    parser.add_argument("--min-session-gap-s", type=float, default=1.20)
    parser.add_argument("--finalization-delay-s", type=float, default=0.0, help="Wait this many seconds after a detected session gap before publishing SESSION_FINAL; helps live threshold candidates settle")
    parser.add_argument("--history-margin-s", type=float, default=0.25)
    parser.add_argument("--active-history-margin-s", type=float, default=None)
    parser.add_argument("--no-prune-finalized-sessions", action="store_true")
    parser.add_argument("--prune-committed-active-sessions", action="store_true")
    parser.add_argument("--raw-updates", action="store_true")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--events", action="store_true", help="Print channel/session lifecycle events")
    parser.add_argument(
        "--json-events",
        action="store_true",
        help="Print channel/session lifecycle events as JSON Lines and suppress human tables",
    )


def _add_carrier_detection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detect-frame-ms", type=float, default=100.0)
    parser.add_argument("--detect-hop-ms", type=float, default=20.0)
    parser.add_argument("--detect-min-tone-hz", type=float, default=200.0)
    parser.add_argument("--detect-max-tone-hz", type=float, default=2000.0)
    parser.add_argument("--max-carriers", type=int, default=5)
    parser.add_argument("--min-separation-hz", type=float, default=80.0)
    parser.add_argument("--relative-threshold", type=float, default=0.15)


def _carrier_detection_config(args: argparse.Namespace):
    from cw.multi_decoder import CarrierDetectionConfig

    return CarrierDetectionConfig(
        frame_ms=args.detect_frame_ms,
        hop_ms=args.detect_hop_ms,
        min_tone_hz=args.detect_min_tone_hz,
        max_tone_hz=args.detect_max_tone_hz,
        max_carriers=args.max_carriers,
        min_separation_hz=args.min_separation_hz,
        relative_threshold=args.relative_threshold,
    )


def _add_decoder_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--min-tone-hz", type=float, default=200.0)
    parser.add_argument("--max-tone-hz", type=float, default=2000.0)
    parser.add_argument("--bandwidth-hz", type=float, default=40.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.35)


def _decoder_config(args: argparse.Namespace):
    from cw.decoder import DecoderConfig

    return DecoderConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        min_tone_hz=args.min_tone_hz,
        max_tone_hz=args.max_tone_hz,
        bandwidth_hz=args.bandwidth_hz,
        threshold_ratio=args.threshold_ratio,
    )


def _streaming_config(args: argparse.Namespace):
    from cw.streaming import StreamingConfig

    return StreamingConfig(
        input_block_ms=args.input_block_ms,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        tracker_frame_ms=args.tracker_frame_ms,
        tracker_hop_ms=args.tracker_hop_ms,
        max_history_s=args.max_history_s,
        max_idle_history_s=args.max_idle_history_s,
        min_tone_hz=args.min_tone_hz,
        max_tone_hz=args.max_tone_hz,
        bandwidth_hz=args.bandwidth_hz,
        threshold_ratio=args.threshold_ratio,
        threshold_ratios=_parse_float_csv(args.threshold_ratios),
        adaptive_gap_thresholds=args.adaptive_gap_thresholds,
        element_letter_gap_units=args.element_letter_gap_units,
        default_word_gap_units=args.default_word_gap_units,
        gap_cluster_min_ratio=args.gap_cluster_min_ratio,
        gap_cluster_min_delta_units=args.gap_cluster_min_delta_units,
        gap_cluster_min_lower_count=args.gap_cluster_min_lower_count,
        peak_relative_threshold=args.peak_relative_threshold,
        track_relative_threshold=args.track_relative_threshold,
        min_peak_snr_db=args.min_peak_snr_db,
        min_keying_tone_runs=args.min_keying_tone_runs,
        min_keying_chars=args.min_keying_chars,
        min_keying_known_chars=args.min_keying_known_chars,
        min_keying_active_duration_s=args.min_keying_active_duration_s,
        min_keying_duty_cycle=args.min_keying_duty_cycle,
        max_keying_duty_cycle=args.max_keying_duty_cycle,
        min_keying_unit_s=args.min_keying_unit_s,
        max_keying_unit_s=args.max_keying_unit_s,
        max_keying_score=args.max_keying_score,
        reject_et_only_sessions=args.reject_et_only_sessions,
        et_only_min_chars=args.et_only_min_chars,
        merge_short_gaps_ms=args.merge_short_gaps_ms,
        drop_short_tones_ms=args.drop_short_tones_ms,
        unit_candidate_spread=args.unit_candidate_spread,
        unit_candidate_steps=args.unit_candidate_steps,
        punctuation_penalty=args.punctuation_penalty,
        min_separation_hz=args.min_separation_hz,
        peak_min_separation_hz=args.peak_min_separation_hz,
        track_match_hz=args.track_match_hz,
        channel_merge_hz=args.channel_merge_hz,
        max_tracks=args.max_tracks,
        max_track_gap_s=args.max_track_gap_s,
        carrier_smoothing=args.carrier_smoothing,
        min_track_hits=args.min_track_hits,
        emit_interval_s=args.emit_interval_s,
        stable_updates=not args.raw_updates,
        min_update_score=args.min_update_score,
        final_text_regression_margin=args.final_text_regression_margin,
        max_final_score=None if args.disable_final_quality_filter else args.max_final_score,
        shadow_suppression_hz=args.shadow_suppression_hz,
        shadow_score_margin=args.shadow_score_margin,
        session_gap_units=args.session_gap_units,
        min_session_gap_s=args.min_session_gap_s,
        finalization_delay_s=getattr(args, "finalization_delay_s", 0.0),
        prune_finalized_sessions=not args.no_prune_finalized_sessions,
        prune_committed_active_sessions=args.prune_committed_active_sessions,
        history_margin_s=args.history_margin_s,
        active_history_margin_s=args.active_history_margin_s,
    )

