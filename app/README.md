
### Final track validation

`stream-sim` applies a final publishability gate before printing completed
tracks. This keeps the tracker sensitive while preventing weak Morse-like shadow
interpretations from being shown as separate real channels. The default final
quality limit is `--max-final-score 30`; use `--disable-final-quality-filter` to
see every candidate during debugging. Close-by shadow tracks are also suppressed
when they are much worse than a neighbouring track. Tune that with
`--shadow-suppression-hz` and `--shadow-score-margin`.

For example, a near-carrier side interpretation such as `CQ CQ DE YUHEE` with a
much worse score than the real `CQ CQ DE YU7NKA` track is dropped from the final
result, but two close real carriers with similar quality are still kept.

## Raw carrier analysis

`analyze-raw` is an offline diagnostic entry point for live captures.  It does
not change the streaming decoder output.  It inspects a reproducible raw PCM
slice and prints the carrier candidates, per-carrier threshold candidates, tone
and gap run distributions, unit estimate, decoded text, and the first classified
runs.  Use it when a live session looks wrong and you want to see whether the
error came from carrier selection, thresholding, unit estimation, or gap
classification.

Example:

```bash
python -m cw.cli analyze-raw samples/live/20260702-184936.s16le \
  --sample-rate 8000 \
  --start-s 18 \
  --duration-s 18 \
  --carrier 600 \
  --carrier 1600 \
  --threshold-ratios 0.12,0.16,0.20,0.25,0.30,0.35
```

If `--carrier` is omitted, the command reports the strongest carriers in the
configured tone range and analyzes those.  Add `--json` to get a single
machine-readable report for later regression tests or notebooks.

## Next-generation raw decoder

`decode-raw` is the first carrier-centric decoder path. It demodulates each
carrier separately with baseband mixing and a low-pass envelope, then keeps
confidence-bearing tone/gap runs and chooses between threshold/unit hypotheses
without QSO-specific text bias.

```bash
python -m cw.cli decode-raw samples/live/20260702-184936.s16le
```

For a focused replay slice:

```bash
python -m cw.cli decode-raw samples/live/20260702-184936.s16le \
  --start-s 18 \
  --duration-s 18 \
  --carrier 600 \
  --carrier 1600
```

The report is session-oriented. One carrier may contain several timed
transmissions, and each session keeps a short candidate ranking so regressions
can compare both the chosen text and the alternatives. Defaults are tuned for
the current raw live captures: 8 kHz `s16le`, carrier search up to 3 kHz,
dynamic threshold candidates, short dropout repair, direct Symbol-HMM duration
candidates with second-pass unit refinement, `--session-gap-s 1.2`, and
`--min-session-evidence-score 0.0`.

## Next-generation live stream

`stream-stdin` and `stream-raw-file` now use the carrier-centric nextgen stream
path. Without `--json-events` they render a live human dashboard by default.
The dashboard is just a viewer on top of the same stream event model: it keeps
an in-place table of carrier frequency, current state, and a rolling
per-carrier transcript. Session/gap fragments remain visible beside the currently
forming text, separated only by extra whitespace, and decoded rows stay visible
for a while even after the carrier goes dormant. Internal channel/session ids stay hidden. The dashboard uses terminal width when available; for `docker compose run -T` you can pass `COLUMNS` or set `CW_VIEW_COLUMNS` explicitly. With `--json-events` the command keeps the
stable JSONL wire schema for tools, servers, future web UI adapters, and deep
debugging. The text decisions are made by vivőnkénti baseband-envelope decoding
instead of the older STFT text path.

The live layer is intentionally low-latency. Earlier builds repeatedly re-decoded
a long rolling batch window; that made strong/old carriers dominate live updates.
The default live window is now short (`--live-decode-window-s`, default 3 s), while
committed text is carried forward by the stream state machine. The expensive
Symbol-HMM remains available for offline decoding and can be opted into for live
experiments, but the live default uses the faster envelope/lattice path so the
dashboard keeps moving. The streamer assigns stable channel/session ids and emits
the usual lifecycle events:

```text
CHANNEL_STARTED -> SESSION_STARTED -> TEXT_COMMITTED -> SESSION_FINAL -> CHANNEL_DORMANT
```

Defaults are intended to work without extra parameters for the current capture
workflow: 8 kHz raw PCM, tone search up to 3 kHz, a short live decode window, dynamic thresholds, adaptive
gap model, and short dropout repair.  Fine-tuning
parameters remain available, but they are normal decoder controls rather than
separate old/new code paths.

Linux/Pulse example:

```bash
ffmpeg -hide_banner -loglevel error \
  -f pulse -i "$(pactl get-default-sink).monitor" \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --prune-committed-active-sessions \
    --live-stats-interval-s 5 \
    --no-finalize-on-interrupt \
    --capture-raw "samples/live/${ts}.s16le"
```


For a separate viewer pipeline, keep the decoder in JSONL mode and pipe it into
the dashboard renderer:

```bash
python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --json-events \
| python -m cw.cli view-events
```

Use `--json-events` directly when another program consumes the stream. Use
`--human-events` only for the old verbose lifecycle log.

Windows/VB-Cable CMD example:

```cmd
ffmpeg -hide_banner -loglevel error -f dshow -i audio="CABLE Output (VB-Audio Virtual Cable)" -f s16le -ac 1 -ar 8000 - | docker compose -f infra\compose.yml run --rm -T cw python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --prune-committed-active-sessions --live-stats-interval-s 5 --no-finalize-on-interrupt --capture-raw "samples/live/%ts%.s16le"
```


## Live receiver note

Live monitoring uses the dashboard human view by default.  The live receiver now uses two separate windows: a short carrier-detection window, and a longer per-carrier text-decode window.  This keeps new/second carriers responsive without starving common phrases such as `CQ CQ DE ...` of enough timing context.
