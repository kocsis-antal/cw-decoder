from __future__ import annotations

import sys

from cw.cli_format import (
    _display_text,
    _format_config,
)
from cw.cli_options import _carrier_detection_config, _decoder_config, _streaming_config
from cw.cli_stream import _print_stream_json_events, _print_stream_summary, _run_live_human_stream
from cw.morse_table import decode_tokens, encode_text


def run_cli_command(args) -> None:
    if args.command == "encode":
        print(" ".join(encode_text(args.text)))
    elif args.command == "decode-tokens":
        print(decode_tokens(args.tokens))
    elif args.command == "decode-wav":
        from cw.decoder import decode_wav

        result = decode_wav(args.path, _decoder_config(args))
        print(result.text)
        print(f"carrier_hz={result.carrier_hz:.1f}")
        print(f"unit_s={result.unit_s:.3f}")
        print(f"tokens={' '.join(result.tokens)}")
    elif args.command == "inspect":
        from cw.decoder import decode_wav

        result = decode_wav(args.path, _decoder_config(args))
        print(f"text={result.text}")
        print(f"carrier_hz={result.carrier_hz:.1f}")
        print(f"threshold={result.threshold:.6f}")
        print(f"unit_s={result.unit_s:.3f}")
        print(f"tokens={' '.join(result.tokens)}")
        print("runs:")
        for run in result.classified_runs:
            print(
                f"  {run.kind:<4} "
                f"start={run.start_s:7.3f}s "
                f"duration={run.duration_s:7.3f}s "
                f"units={run.units:5.2f} "
                f"symbol={run.symbol}"
            )
    elif args.command == "evaluate":
        from cw.evaluation import evaluate_wav

        result = evaluate_wav(args.wav_path, args.labels_path, _decoder_config(args))
        print(f"expected_text={result.expected_text}")
        print(f"decoded_text={result.decoded_text}")
        print(f"text_ok={result.text_ok}")
        print(f"token_accuracy={result.token_accuracy:.1%}")
        print(f"expected_carrier_hz={result.expected_carrier_hz:.1f}")
        print(f"detected_carrier_hz={result.detected_carrier_hz:.1f}")
        print(f"carrier_error_hz={result.carrier_error_hz:+.1f}")
        print(f"expected_unit_s={result.expected_unit_s:.3f}")
        print(f"detected_unit_s={result.detected_unit_s:.3f}")
        print(f"unit_error_ms={result.unit_error_ms:+.1f}")
        print(f"events_compared={result.timing.compared_count}")
        print(f"event_count_delta={result.timing.count_delta:+d}")
        print(f"event_symbol_accuracy={result.timing.symbol_accuracy:.1%}")
        print(f"avg_start_error_ms={result.timing.avg_start_error_ms:.1f}")
        print(f"max_start_error_ms={result.timing.max_start_error_ms:.1f}")
        print(f"avg_duration_error_ms={result.timing.avg_duration_error_ms:.1f}")
        print(f"max_duration_error_ms={result.timing.max_duration_error_ms:.1f}")
    elif args.command == "contest":
        from cw.contest import ContestGrid, parse_float_list, run_contest

        grid = ContestGrid(
            frame_ms=parse_float_list(args.frame_ms),
            hop_ms=parse_float_list(args.hop_ms),
            bandwidth_hz=parse_float_list(args.bandwidth_hz),
            threshold_ratio=parse_float_list(args.threshold_ratio),
        )
        results = run_contest(
            args.wav_path,
            args.labels_path,
            grid,
            min_tone_hz=args.min_tone_hz,
            max_tone_hz=args.max_tone_hz,
        )
        print(f"tested={len(results)}")
        print("rank score text_ok token_acc symbol_acc frame hop bandwidth threshold unit_err_ms avg_start_ms avg_duration_ms text")
        for result in results[: args.top]:
            evaluation = result.evaluation
            config = result.config
            print(
                f"{result.rank:>4} "
                f"{result.score:>7.1f} "
                f"{str(evaluation.text_ok):>7} "
                f"{evaluation.token_accuracy:>9.1%} "
                f"{evaluation.timing.symbol_accuracy:>10.1%} "
                f"{config.frame_ms:>5.1f} "
                f"{config.hop_ms:>3.1f} "
                f"{config.bandwidth_hz:>9.1f} "
                f"{config.threshold_ratio:>9.2f} "
                f"{evaluation.unit_error_ms:>11.1f} "
                f"{evaluation.timing.avg_start_error_ms:>12.1f} "
                f"{evaluation.timing.avg_duration_error_ms:>15.1f} "
                f"{evaluation.decoded_text}"
            )
    elif args.command == "contest-live":
        from cw.contest import ContestGrid, parse_float_list, run_live_contest, summarize_live_consensus

        grid = ContestGrid(
            frame_ms=parse_float_list(args.frame_ms),
            hop_ms=parse_float_list(args.hop_ms),
            bandwidth_hz=parse_float_list(args.bandwidth_hz),
            threshold_ratio=parse_float_list(args.threshold_ratio),
        )
        results = run_live_contest(
            args.wav_path,
            grid,
            min_tone_hz=args.min_tone_hz,
            max_tone_hz=args.max_tone_hz,
        )
        print(f"tested={len(results)}")
        print("rank score unknown tokens dots dashes ratio_err gap_err unit_cv frame hop bandwidth threshold carrier unit text")
        for result in results[: args.top]:
            quality = result.quality
            config = result.config
            decoded = result.decoded
            print(
                f"{result.rank:>4} "
                f"{quality.score:>7.1f} "
                f"{quality.unknown_count:>7} "
                f"{quality.token_count:>6} "
                f"{quality.dot_count:>4} "
                f"{quality.dash_count:>6} "
                f"{quality.tone_ratio_error:>9.3f} "
                f"{quality.gap_min_error:>7.3f} "
                f"{quality.unit_cv:>7.3f} "
                f"{config.frame_ms:>5.1f} "
                f"{config.hop_ms:>3.1f} "
                f"{config.bandwidth_hz:>9.1f} "
                f"{config.threshold_ratio:>9.2f} "
                f"{decoded.carrier_hz:>7.1f} "
                f"{decoded.unit_s:>4.3f} "
                f"{_display_text(decoded.text)}"
            )
        consensus = summarize_live_consensus(results)
        print("consensus:")
        print("rank count share best_score best_rank frame hop bandwidth threshold carrier unit text")
        for result in consensus[: args.consensus_top]:
            config = result.best_config
            decoded = result.best_decoded
            print(
                f"{result.rank:>4} "
                f"{result.count:>5} "
                f"{result.share:>6.1%} "
                f"{result.best_score:>10.1f} "
                f"{result.best_rank:>9} "
                f"{config.frame_ms:>5.1f} "
                f"{config.hop_ms:>3.1f} "
                f"{config.bandwidth_hz:>9.1f} "
                f"{config.threshold_ratio:>9.2f} "
                f"{decoded.carrier_hz:>7.1f} "
                f"{decoded.unit_s:>4.3f} "
                f"{_display_text(result.text)}"
            )
    elif args.command == "benchmark":
        from cw.benchmark import (
            check_benchmark_expectations,
            parse_int_list,
            parse_string_list,
            run_benchmark,
        )
        from cw.contest import ContestGrid, parse_float_list

        grid = ContestGrid(
            frame_ms=parse_float_list(args.frame_ms),
            hop_ms=parse_float_list(args.hop_ms),
            bandwidth_hz=parse_float_list(args.bandwidth_hz),
            threshold_ratio=parse_float_list(args.threshold_ratio),
        )
        results = run_benchmark(
            args.text,
            args.out_dir,
            parse_string_list(args.presets),
            parse_int_list(args.seeds),
            grid,
            min_tone_hz=args.min_tone_hz,
            max_tone_hz=args.max_tone_hz,
        )
        print(f"cases={len(results)}")
        print(
            "preset seed known_ok known_score known_cfg live_ok live_score "
            "live_known_rank live_cfg live_text"
        )
        for result in results:
            known_config = result.known_best.config
            live_config = result.live_best.config
            print(
                f"{result.preset:<6} "
                f"{result.seed:>4} "
                f"{str(result.known_best.evaluation.text_ok):>8} "
                f"{result.known_best.score:>11.1f} "
                f"{_format_config(known_config):>17} "
                f"{str(result.live_evaluation.text_ok):>7} "
                f"{result.live_best.quality.score:>10.1f} "
                f"{str(result.live_rank_in_known):>15} "
                f"{_format_config(live_config):>17} "
                f"{result.live_best.decoded.text}"
            )
        if args.expect:
            expectation = check_benchmark_expectations(
                results,
                parse_string_list(args.expected_pass_presets),
                parse_string_list(args.allowed_fail_presets),
            )
            print(f"expectation_passed={expectation.passed}")
            print(f"expected_pass_presets={','.join(expectation.expected_pass_presets)}")
            print(f"allowed_fail_presets={','.join(expectation.allowed_fail_presets)}")
            for failure in expectation.failures:
                print(f"expectation_failure={failure}")
            if not expectation.passed:
                sys.exit(1)
    elif args.command == "generate":
        from cw.generator import generator_config_from_preset, override_generator_config, write_sample

        config = override_generator_config(
            generator_config_from_preset(args.preset),
            sample_rate=args.sample_rate,
            tone_hz=args.tone_hz,
            wpm=args.wpm,
            amplitude=args.amplitude,
            timing_jitter=args.timing_jitter,
            dot_jitter=args.dot_jitter,
            dash_jitter=args.dash_jitter,
            element_gap_jitter=args.element_gap_jitter,
            letter_gap_jitter=args.letter_gap_jitter,
            word_gap_jitter=args.word_gap_jitter,
            dash_ratio=args.dash_ratio,
            speed_wobble=args.speed_wobble,
            speed_wobble_hz=args.speed_wobble_hz,
            frequency_drift_hz=args.frequency_drift_hz,
            frequency_wobble_hz=args.frequency_wobble_hz,
            frequency_wobble_rate_hz=args.frequency_wobble_rate_hz,
            amplitude_fade=args.amplitude_fade,
            amplitude_fade_hz=args.amplitude_fade_hz,
            noise_snr_db=args.noise_snr_db,
            seed=args.seed,
        )
        label_path = write_sample(args.text, args.out, config)
        print(f"Wrote {args.out}")
        print(f"Wrote {label_path}")
    elif args.command == "generate-multi":
        from cw.multi_generator import parse_source_spec, write_multi_sample

        sources = [
            parse_source_spec(
                source_spec,
                index=index,
                sample_rate=args.sample_rate,
                seed=args.seed,
            )
            for index, source_spec in enumerate(args.source)
        ]
        result = write_multi_sample(
            sources,
            args.out,
            sample_rate=args.sample_rate,
            normalize_peak=args.normalize_peak,
            noise_snr_db=args.mix_noise_snr_db,
            seed=args.seed,
        )
        print(f"Wrote {result.wav_path}")
        print(f"Wrote {result.label_path}")
        print(f"sources={result.source_count}")
        print(f"duration_s={result.duration_s:.3f}")
        print(f"normalized_gain={result.normalized_gain:.3f}")
    elif args.command == "generate-qso":
        from cw.qso_generator import ContestQsoConfig, write_contest_qso_sample

        config = ContestQsoConfig(
            caller_call=args.caller,
            responder_call=args.responder,
            base_frequency_hz=args.freq,
            responder_offset_hz=args.responder_offset_hz,
            start_s=args.start,
            turn_gap_s=args.turn_gap_s,
            caller_preset=args.caller_preset,
            responder_preset=args.responder_preset,
            caller_wpm=args.caller_wpm,
            responder_wpm=args.responder_wpm,
            caller_amplitude=args.caller_amplitude,
            responder_amplitude=args.responder_amplitude,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
        result = write_contest_qso_sample(
            args.out,
            config,
            normalize_peak=args.normalize_peak,
            mix_noise_snr_db=args.mix_noise_snr_db,
        )
        print(f"Wrote {result.wav_path}")
        print(f"Wrote {result.label_path}")
        print(f"sources={result.source_count}")
        print(f"duration_s={result.duration_s:.3f}")
        print(f"normalized_gain={result.normalized_gain:.3f}")
    elif args.command == "detect-carriers":
        from cw.multi_decoder import detect_carriers

        carriers = detect_carriers(args.wav_path, _carrier_detection_config(args))
        print(f"detected={len(carriers)}")
        print("rank frequency_hz relative_power power")
        for carrier in carriers:
            print(
                f"{carrier.rank:>4} "
                f"{carrier.frequency_hz:>12.1f} "
                f"{carrier.relative_power:>14.3f} "
                f"{carrier.power:.6g}"
            )
    elif args.command == "contest-live-multi":
        from cw.contest import ContestGrid, parse_float_list
        from cw.multi_decoder import run_multi_live_contest

        grid = ContestGrid(
            frame_ms=parse_float_list(args.frame_ms),
            hop_ms=parse_float_list(args.hop_ms),
            bandwidth_hz=parse_float_list(args.bandwidth_hz),
            threshold_ratio=parse_float_list(args.threshold_ratio),
        )
        results = run_multi_live_contest(args.wav_path, grid, _carrier_detection_config(args))
        print(f"carriers={len(results)}")
        for result in results:
            best_consensus = result.best_consensus
            config = best_consensus.best_config
            decoded = best_consensus.best_decoded
            print()
            print(
                f"source={result.rank} "
                f"carrier_hz={result.carrier.frequency_hz:.1f} "
                f"rel_power={result.carrier.relative_power:.3f} "
                f"best_score={best_consensus.best_score:.1f} "
                f"consensus_share={best_consensus.share:.1%}"
            )
            print(
                f"best_config=frame:{config.frame_ms:g}ms "
                f"hop:{config.hop_ms:g}ms "
                f"bandwidth:{config.bandwidth_hz:g}Hz "
                f"threshold:{config.threshold_ratio:g} "
                f"unit:{decoded.unit_s:.3f}s"
            )
            print(f"text={_display_text(best_consensus.text)}")
            if args.top > 0:
                print("top:")
                for live_result in result.live_results[: args.top]:
                    quality = live_result.quality
                    cfg = live_result.config
                    dec = live_result.decoded
                    print(
                        f"  {live_result.rank:>4} "
                        f"score={quality.score:>6.1f} "
                        f"f={cfg.frame_ms:g} h={cfg.hop_ms:g} b={cfg.bandwidth_hz:g} t={cfg.threshold_ratio:g} "
                        f"unit={dec.unit_s:.3f} text={_display_text(dec.text)}"
                    )
            if args.consensus_top > 0:
                print("consensus:")
                for consensus in result.consensus[: args.consensus_top]:
                    print(
                        f"  {consensus.rank:>4} "
                        f"count={consensus.count:>3} "
                        f"share={consensus.share:>6.1%} "
                        f"score={consensus.best_score:>6.1f} "
                        f"text={_display_text(consensus.text)}"
                    )
    elif args.command == "spacing-benchmark":
        from cw.spacing_benchmark import (
            SpacingBenchmarkConfig,
            check_spacing_expectations,
            parse_float_list as parse_spacing_float_list,
            run_spacing_benchmark,
        )

        spacing_config = SpacingBenchmarkConfig(
            base_frequency_hz=args.base_freq,
            deltas_hz=parse_spacing_float_list(args.deltas),
            merge_below_hz=args.merge_below_hz,
            split_from_hz=args.split_from_hz,
            source_a_preset=args.preset_a,
            source_b_preset=args.preset_b,
            source_a_wpm=args.wpm_a,
            source_b_wpm=args.wpm_b,
            source_a_amplitude=args.amplitude_a,
            source_b_amplitude=args.amplitude_b,
            source_b_start_s=args.start_b,
            sample_rate=args.sample_rate,
            seed=args.seed,
            normalize_peak=args.normalize_peak,
            mix_noise_snr_db=args.mix_noise_snr_db,
            stream_frame_ms=args.stream_frame_ms,
            stream_hop_ms=args.stream_hop_ms,
            tracker_frame_ms=args.tracker_frame_ms,
            tracker_hop_ms=args.tracker_hop_ms,
            stream_bandwidth_hz=args.stream_bandwidth_hz,
            stream_threshold_ratio=args.stream_threshold_ratio,
            peak_relative_threshold=args.peak_relative_threshold,
            track_relative_threshold=args.track_relative_threshold,
            min_peak_snr_db=args.min_peak_snr_db,
            max_final_score=None if args.disable_final_quality_filter else args.max_final_score,
            shadow_suppression_hz=args.shadow_suppression_hz,
            shadow_score_margin=args.shadow_score_margin,
            min_separation_hz=args.min_separation_hz,
            peak_min_separation_hz=args.peak_min_separation_hz,
            track_match_hz=args.track_match_hz,
            channel_merge_hz=args.channel_merge_hz,
            max_tracks=args.max_tracks,
        )
        results = run_spacing_benchmark(args.text_a, args.text_b, args.out_dir, spacing_config)
        print(f"cases={len(results)}")
        print(
            "delta_hz expected result channels carriers source_a_ok source_b_ok texts"
        )
        for result in results:
            carrier_text = ",".join(f"{carrier:.1f}" for carrier in result.carriers_hz) or "-"
            decoded_text = " || ".join(_display_text(text) for text in result.decoded_texts) or "<none>"
            print(
                f"{result.delta_hz:>8.1f} "
                f"{result.expected:<9} "
                f"{result.result_label:<6} "
                f"{result.detected_channels:>8} "
                f"{carrier_text:<16} "
                f"{str(result.source_a_ok):>11} "
                f"{str(result.source_b_ok):>11} "
                f"{decoded_text}"
            )
        if args.expect:
            expectation = check_spacing_expectations(results)
            print(f"expectation_passed={expectation.passed}")
            for failure in expectation.failures:
                print(f"expectation_failure={failure}")
            if not expectation.passed:
                sys.exit(1)

    elif args.command == "decode-raw":
        from cw.nextgen import decode_raw_file_nextgen, format_decode_report, report_to_json
        from cw.prob_analysis import parse_float_csv

        report = decode_raw_file_nextgen(
            args.raw_path,
            sample_rate=args.sample_rate,
            sample_format=args.sample_format,
            channels=args.channels,
            start_s=args.start_s,
            duration_s=args.duration_s,
            carriers=tuple(args.carrier or ()),
            detect_carriers=args.detect_carriers,
            min_tone_hz=args.min_tone_hz,
            max_tone_hz=args.max_tone_hz,
            min_separation_hz=args.min_separation_hz,
            peak_relative_threshold=args.peak_relative_threshold,
            detect_frame_ms=args.detect_frame_ms,
            detect_hop_ms=args.detect_hop_ms,
            lowpass_ms=args.lowpass_ms,
            envelope_hop_ms=args.envelope_hop_ms,
            threshold_ratios=parse_float_csv(args.threshold_ratios),
            merge_short_gaps_ms=args.merge_short_gaps_ms,
            drop_short_tones_ms=args.drop_short_tones_ms,
            unit_candidate_spread=args.unit_candidate_spread,
            unit_candidate_steps=args.unit_candidate_steps,
            adaptive_gap_thresholds=args.adaptive_gap_thresholds,
            element_letter_gap_units=args.element_letter_gap_units,
            default_word_gap_units=args.default_word_gap_units,
            gap_cluster_min_ratio=args.gap_cluster_min_ratio,
            gap_cluster_min_delta_units=args.gap_cluster_min_delta_units,
            gap_cluster_min_lower_count=args.gap_cluster_min_lower_count,
            session_gap_s=args.session_gap_s,
            min_session_evidence_score=args.min_session_evidence_score,
            max_candidates_per_carrier=args.max_candidates_per_carrier,
            max_candidates_per_session=args.max_candidates_per_session,
        )
        print(report_to_json(report) if args.json else format_decode_report(report))

    elif args.command == "analyze-raw":
        from cw.prob_analysis import analyze_raw_file, format_human_report, parse_float_csv, report_to_json

        report = analyze_raw_file(
            args.raw_path,
            sample_rate=args.sample_rate,
            sample_format=args.sample_format,
            channels=args.channels,
            start_s=args.start_s,
            duration_s=args.duration_s,
            carriers=tuple(args.carrier or ()),
            detect_carriers=args.detect_carriers,
            min_tone_hz=args.min_tone_hz,
            max_tone_hz=args.max_tone_hz,
            min_separation_hz=args.min_separation_hz,
            peak_relative_threshold=args.peak_relative_threshold,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            bandwidth_hz=args.bandwidth_hz,
            threshold_ratios=parse_float_csv(args.threshold_ratios),
            adaptive_gap_thresholds=args.adaptive_gap_thresholds,
            element_letter_gap_units=args.element_letter_gap_units,
            default_word_gap_units=args.default_word_gap_units,
            gap_cluster_min_ratio=args.gap_cluster_min_ratio,
            gap_cluster_min_delta_units=args.gap_cluster_min_delta_units,
            gap_cluster_min_lower_count=args.gap_cluster_min_lower_count,
            merge_short_gaps_ms=args.merge_short_gaps_ms,
            drop_short_tones_ms=args.drop_short_tones_ms,
            unit_candidate_spread=args.unit_candidate_spread,
            unit_candidate_steps=args.unit_candidate_steps,
            punctuation_penalty=args.punctuation_penalty,
            preview_runs=args.preview_runs,
        )
        print(report_to_json(report) if args.json else format_human_report(report))

    elif args.command == "stream-sim":
        from cw.streaming import WavFileSource

        config = _streaming_config(args)
        source = WavFileSource(args.wav_path, config.input_block_ms)
        if args.json_events:
            _print_stream_json_events(source, config, args)
            return

        result = _run_live_human_stream(source, config, args)
        if result is None:
            return
        _print_stream_summary(result, args)
    elif args.command == "stream-stdin":
        from cw.streaming import RawPcmStreamSource

        config = _streaming_config(args)
        source = RawPcmStreamSource(
            sys.stdin.buffer,
            sample_rate=args.sample_rate,
            sample_format=args.sample_format,
            channels=args.channels,
            block_ms=config.input_block_ms,
            duration_s=args.duration_s,
            capture_raw_path=args.capture_raw,
        )
        if args.json_events:
            _print_stream_json_events(source, config, args)
            return

        result = _run_live_human_stream(source, config, args)
        if result is None:
            return
        _print_stream_summary(result, args)
    elif args.command == "stream-raw-file":
        from cw.streaming import RawPcmStreamSource

        config = _streaming_config(args)
        with args.raw_path.open("rb") as raw_file:
            source = RawPcmStreamSource(
                raw_file,
                sample_rate=args.sample_rate,
                sample_format=args.sample_format,
                channels=args.channels,
                block_ms=config.input_block_ms,
                duration_s=args.duration_s,
            )
            if args.json_events:
                _print_stream_json_events(source, config, args)
                return

            result = _run_live_human_stream(source, config, args)
        if result is None:
            return
        _print_stream_summary(result, args)


