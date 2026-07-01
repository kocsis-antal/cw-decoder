
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
