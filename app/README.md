
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
  --threshold-ratios 0.20,0.25,0.30,0.35
```

If `--carrier` is omitted, the command reports the strongest carriers in the
configured tone range and analyzes those.  Add `--json` to get a single
machine-readable report for later regression tests or notebooks.
