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
    parser.add_argument("--max-history-s", type=float, default=12.0, help="Hard cap for retained live audio history; defaults to 12 s so streaming CPU/memory do not grow unbounded")
    parser.add_argument("--max-idle-history-s", type=float, default=None, help="Shorter retained-history cap before any channel/session has been recognized")
    parser.add_argument("--live-carrier-window-s", type=float, default=2.0, help="Short recent audio window used only for live carrier detection/tracking")
    parser.add_argument("--live-decode-window-s", type=float, default=8.0, help="Recent per-carrier audio window decoded for live text; carrier tracking remains short-windowed")
    parser.add_argument("--min-tone-hz", type=float, default=200.0)
    parser.add_argument("--max-tone-hz", type=float, default=3000.0)
    parser.add_argument("--bandwidth-hz", type=float, default=40.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.35)
    parser.add_argument(
        "--threshold-ratios",
        default="",
        help="Comma separated dynamic threshold candidates. Empty means use only --threshold-ratio.",
    )
    parser.add_argument("--soft-activity", action=argparse.BooleanOptionalAction, default=True, help="Use nextgen Viterbi probability tone/gap candidates in addition to hard thresholds")
    parser.add_argument("--soft-tone-on-probability", type=float, default=0.56, help="Probability level where the soft tone gate turns on")
    parser.add_argument("--soft-tone-off-probability", type=float, default=0.28, help="Probability level where the soft tone gate turns off")
    parser.add_argument("--soft-bridge-min-probability", type=float, default=0.18, help="Minimum mean probability required to bridge a short fade gap inside a tone")
    parser.add_argument("--soft-bridge-max-gap-ms", type=float, default=90.0, help="Maximum absolute gap length the soft gate may bridge inside a tone")
    parser.add_argument("--soft-bridge-gap-units", type=float, default=1.6, help="Maximum gap length in estimated keying units the soft gate may bridge inside a tone")
    parser.add_argument("--viterbi-transition-penalty", type=float, default=1.15, help="Penalty for tone/gap state changes in the probability Viterbi activity decoder")
    parser.add_argument("--symbol-hmm-decoding", action=argparse.BooleanOptionalAction, default=True, help="Decode directly from carrier activity probabilities with a duration-HMM/beam symbol model")
    parser.add_argument("--symbol-hmm-beam-width", type=int, default=16, help="Maximum states retained by the direct symbol-HMM decoder")
    parser.add_argument("--symbol-hmm-max-candidates", type=int, default=3, help="Maximum direct symbol-HMM alternatives generated per session range")
    parser.add_argument("--symbol-hmm-unit-spread", type=float, default=0.18, help="Relative unit-time spread searched by the direct symbol-HMM decoder")
    parser.add_argument("--symbol-hmm-unit-steps", type=int, default=3, help="Number of unit-time hypotheses tried by the direct symbol-HMM decoder")
    parser.add_argument("--symbol-hmm-transition-penalty", type=float, default=0.18, help="Penalty for adding each tone/gap duration transition in the direct symbol-HMM decoder")
    parser.add_argument("--symbol-hmm-min-unit-s", type=float, default=0.025, help="Minimum dit unit accepted by the direct symbol-HMM decoder")
    parser.add_argument("--symbol-hmm-max-unit-s", type=float, default=0.250, help="Maximum dit unit accepted by the direct symbol-HMM decoder")
    parser.add_argument("--symbol-hmm-live-interval-s", type=float, default=2.0, help="Run the expensive direct symbol-HMM live path at most this often per carrier; 0 means every decode tick")
    parser.add_argument("--lattice-decoding", action=argparse.BooleanOptionalAction, default=True, help="Keep alternate dot/dash and gap interpretations alive near timing boundaries")
    parser.add_argument("--lattice-beam-width", type=int, default=12, help="Maximum number of timing interpretation states retained by the lattice decoder")
    parser.add_argument("--lattice-max-candidates", type=int, default=3, help="Maximum lattice alternatives generated per unit hypothesis")
    parser.add_argument("--lattice-tone-margin-units", type=float, default=0.45, help="Allow both dot and dash when tone length is this close to the 2-unit boundary")
    parser.add_argument("--lattice-gap-margin-units", type=float, default=0.60, help="Allow neighbouring gap classes when gap length is this close to a boundary")
    parser.add_argument("--adaptive-gap-thresholds", action=argparse.BooleanOptionalAction, default=True, help="Estimate letter/word gap boundary from session timing instead of using a fixed 5-unit split")
    parser.add_argument("--element-letter-gap-units", type=float, default=2.0, help="Boundary between intra-character and inter-character gaps, in estimated CW units")
    parser.add_argument("--default-word-gap-units", type=float, default=7.0, help="Fallback word-gap boundary when no distinct word-gap cluster is visible")
    parser.add_argument("--gap-cluster-min-ratio", type=float, default=1.45, help="Minimum multiplicative separation required to split letter and word gap clusters")
    parser.add_argument("--gap-cluster-min-delta-units", type=float, default=1.0, help="Minimum absolute gap separation in CW units required for a letter/word split")
    parser.add_argument("--gap-cluster-min-lower-count", type=int, default=2, help="Require at least this many shorter inter-letter candidates before creating a word-gap cluster")
    parser.add_argument("--peak-relative-threshold", type=float, default=0.05)
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
    parser.add_argument("--channel-reacquire-hz", type=float, default=None, help="Reuse a recent channel within this wider frequency tolerance instead of opening a new one")
    parser.add_argument("--channel-reacquire-s", type=float, default=15.0, help="Reuse recent/dormant channels for this many seconds after last seen")
    parser.add_argument("--max-tracks", type=int, default=5)
    parser.add_argument("--max-track-gap-s", type=float, default=2.0)
    parser.add_argument("--carrier-smoothing", type=float, default=0.20)
    parser.add_argument("--min-track-hits", type=int, default=2)
    parser.add_argument("--emit-interval-s", type=float, default=0.50)
    parser.add_argument("--min-update-score", type=float, default=25.0)
    parser.add_argument("--min-live-commit-chars", type=int, default=2, help="Minimum stable non-space characters before emitting a live TEXT_COMMITTED update")
    parser.add_argument("--preview-updates", action=argparse.BooleanOptionalAction, default=True, help="Emit best-effort TEXT_PREVIEW events for active CW before the text is stable enough to commit")
    parser.add_argument("--preview-interval-s", type=float, default=0.75, help="Minimum seconds between TEXT_PREVIEW events for one live session")
    parser.add_argument("--preview-min-chars", type=int, default=1, help="Minimum non-space characters before emitting a TEXT_PREVIEW update")
    parser.add_argument("--preview-max-score", type=float, default=80.0, help="Reject TEXT_PREVIEW candidates with a worse quality score; use a negative value to disable")
    parser.add_argument("--signal-activity-interval-s", type=float, default=2.0, help="Emit SIGNAL_ACTIVE heartbeat events for confirmed carriers that have no decodable text yet")
    parser.add_argument("--live-progress-interval-s", type=float, default=1.25, help="Emit a best-effort active-session progress update after this many seconds without a stable prefix commit")
    parser.add_argument("--live-progress-min-overlap-chars", type=int, default=3, help="Minimum compact text overlap required when stitching rolling-window live progress updates")
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
    parser.add_argument(
        "--human-view",
        choices=["dashboard", "compact", "events", "off"],
        default="dashboard",
        help="Human live output mode used when --json-events is not set. dashboard is the default; compact is kept as an alias; events is the old verbose lifecycle log; off prints only the final decoded summary.",
    )
    parser.add_argument(
        "--events",
        "--human-events",
        dest="human_view",
        action="store_const",
        const="events",
        help="Print the old verbose human-readable channel/session lifecycle events",
    )
    parser.add_argument(
        "--no-human-events",
        dest="human_view",
        action="store_const",
        const="off",
        help="Suppress live human events and print only the final summary",
    )
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
        live_carrier_window_s=args.live_carrier_window_s,
        live_decode_window_s=args.live_decode_window_s,
        min_tone_hz=args.min_tone_hz,
        max_tone_hz=args.max_tone_hz,
        bandwidth_hz=args.bandwidth_hz,
        threshold_ratio=args.threshold_ratio,
        threshold_ratios=_parse_float_csv(args.threshold_ratios),
        soft_activity=args.soft_activity,
        soft_tone_on_probability=args.soft_tone_on_probability,
        soft_tone_off_probability=args.soft_tone_off_probability,
        soft_bridge_min_probability=args.soft_bridge_min_probability,
        soft_bridge_max_gap_ms=args.soft_bridge_max_gap_ms,
        soft_bridge_gap_units=args.soft_bridge_gap_units,
        viterbi_transition_penalty=args.viterbi_transition_penalty,
        symbol_hmm_decoding=args.symbol_hmm_decoding,
        symbol_hmm_beam_width=args.symbol_hmm_beam_width,
        symbol_hmm_max_candidates=args.symbol_hmm_max_candidates,
        symbol_hmm_unit_spread=args.symbol_hmm_unit_spread,
        symbol_hmm_unit_steps=args.symbol_hmm_unit_steps,
        symbol_hmm_transition_penalty=args.symbol_hmm_transition_penalty,
        symbol_hmm_min_unit_s=args.symbol_hmm_min_unit_s,
        symbol_hmm_max_unit_s=args.symbol_hmm_max_unit_s,
        symbol_hmm_live_interval_s=args.symbol_hmm_live_interval_s,
        lattice_decoding=args.lattice_decoding,
        lattice_beam_width=args.lattice_beam_width,
        lattice_max_candidates=args.lattice_max_candidates,
        lattice_tone_margin_units=args.lattice_tone_margin_units,
        lattice_gap_margin_units=args.lattice_gap_margin_units,
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
        channel_reacquire_hz=args.channel_reacquire_hz,
        channel_reacquire_s=args.channel_reacquire_s,
        max_tracks=args.max_tracks,
        max_track_gap_s=args.max_track_gap_s,
        carrier_smoothing=args.carrier_smoothing,
        min_track_hits=args.min_track_hits,
        emit_interval_s=args.emit_interval_s,
        stable_updates=not args.raw_updates,
        min_update_score=args.min_update_score,
        min_live_commit_chars=args.min_live_commit_chars,
        preview_updates=args.preview_updates,
        preview_interval_s=args.preview_interval_s,
        preview_min_chars=args.preview_min_chars,
        preview_max_score=None if args.preview_max_score is not None and args.preview_max_score < 0 else args.preview_max_score,
        signal_activity_interval_s=args.signal_activity_interval_s,
        live_progress_interval_s=args.live_progress_interval_s,
        live_progress_min_overlap_chars=args.live_progress_min_overlap_chars,
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

