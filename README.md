# CW Morse

Experimental CW Morse signal generator and decoder.

The runtime files live under `infra/`. The Python source lives under `app/`.
The container only mounts `app/` to `/app`, so it cannot see its own Compose file.

## Commands

Build the development image:

```bash
docker compose -f infra/compose.yml build
```

Open a shell:

```bash
docker compose -f infra/compose.yml run --rm cw
```

Run tests:

```bash
docker compose -f infra/compose.yml run --rm cw pytest
```

Try the first CLI:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli encode "CQ CQ DE HA5ABC"
```

Generate the first WAV sample and label file:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq.wav
```

Generate a harder sample with timing jitter, carrier drift and noise:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_harder.wav --timing-jitter 0.15 --frequency-drift-hz 25 --noise-snr-db 30 --seed 123
```

Generate the same kind of harder sample through a preset:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_hard.wav --preset hard
```

Generate a straight-key style human timing sample:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_straight.wav --preset straight
```

Generate a field-style sample with straight-key timing, drift, wobble, fading and noise:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_field.wav --preset field
```

Generate an ugly sample:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_ugly.wav --preset ugly
```

Generate a brutal stress-test sample:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_brutal.wav --preset brutal
```

Generate a custom fading/wobbling sample:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate "CQ CQ DE HA5ABC" --out samples/generated/cq_wobble.wav --timing-jitter 0.2 --speed-wobble 0.2 --speed-wobble-hz 0.12 --frequency-drift-hz 60 --frequency-wobble-hz 20 --frequency-wobble-rate-hz 0.25 --amplitude-fade 0.5 --amplitude-fade-hz 0.35 --noise-snr-db 15 --seed 123
```

Decode the generated WAV:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli decode-wav samples/generated/cq.wav
```

Inspect the detected runs:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli inspect samples/generated/cq.wav
```

Evaluate the decoder output against the generated labels:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli evaluate samples/generated/cq.wav samples/generated/cq.labels.json
```

Run a parameter contest:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli contest samples/generated/cq.wav samples/generated/cq.labels.json
```

Run a live-style contest without labels:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli contest-live samples/generated/cq.wav
```

The live contest also prints a `consensus` table. It groups all decoder
configs by decoded text, so a result is stronger when many different configs
agree on the same message.

Run a benchmark over presets and seeds:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli benchmark "CQ CQ DE HA5ABC"
```

Run the benchmark as a regression expectation:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli benchmark "CQ CQ DE HA5ABC" --expect
```

## Multi-source experiments

Generate a WAV with multiple CW sources at different audio tones:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate-multi \
  --out samples/generated/two_sources.wav \
  --source "id=me;freq=700;preset=field;text=CQ CQ DE YU7NKA" \
  --source "id=apu;freq=1000;preset=straight;text=CQ CQ DE YT7MK;start=0.4;amplitude=0.45"
```

Detect carrier tones in the mixed WAV:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli detect-carriers samples/generated/two_sources.wav
```

Run a live-style multi-source decode. This first detects carrier tones, then
runs a separate targeted decoder contest for each detected carrier:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli contest-live-multi samples/generated/two_sources.wav
```

A `--source` value is a semicolon-separated list of `key=value` pairs. Required
keys are `freq` and `text`. Useful optional keys are `id`, `preset`, `start`,
`amplitude`, `wpm`, `seed`, and the same distortion keys supported by the
single-source generator, for example `frequency_drift_hz` or `amplitude_fade`.

## Streaming simulation

`stream-sim` does not split the WAV into separate files. It reads the audio as a
continuous sample stream, feeds it into an internal ring-buffer-like STFT, and
produces overlapping FFT frames. Carrier detection and decoding are updated from
the accumulated frame history, which is the first step toward replacing the WAV
input with a microphone input later.

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/two_sources.wav
```

The output contains incremental `updates` and a final track table. Each update
includes the current session id, so a QSO turn change is visible while the text
is still growing. By default, updates are stabilized: a partial text is printed
only after it appears as a stable prefix in consecutive decoding snapshots and
its quality score is below `--min-update-score`. This avoids most early
half-letter guesses while still showing the text grow on the fly.

For debugging the raw, unstable candidates, use:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/two_sources.wav --raw-updates
```

Print channel/session lifecycle events:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/two_sources.wav --events
```

The event stream separates a durable carrier channel from a concrete transmission
session. A channel can become dormant and still remain useful for a GUI card,
while each session has its own final text and its own decoded unit/tempo. Long
silence inside one carrier can split the final result into multiple sessions.
The split threshold is controlled by `--session-gap-units` and
`--min-session-gap-s`. When a track has more than one session, the final track
summary uses `|` separators and a `sessions:` table is printed so separate QSO
overs do not collapse into one unreadable line.

By default, finalized session frame history is pruned from the rolling stream
state. The channel keeps the finalized session text and metadata, but the old
FFT frames no longer have to be decoded again on every update. The `stream-sim`
header prints `frames_processed`, `retained_frames`, and `pruned_frames` so this
can be checked during experiments. Use `--no-prune-finalized-sessions` when you
want full-history debug behaviour. `--history-margin-s` controls the small
safety overlap kept before the active session.

## Contest-QSO scenario

Generate a short contest-style exchange on one durable carrier channel. The two
operators use slightly different audio tones and different speeds, but the tones
are close enough that the streaming tracker should keep them on one channel and
split the overs into separate sessions after long silence:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli generate-qso \
  --out samples/generated/qso.wav \
  --caller YU7NKA \
  --responder YT7MK
```

Then inspect the live/session behaviour:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/qso.wav --events
```

The expected final sessions are roughly:

```text
CQ TEST YU7NKA
YU7NKA YT7MK
YT7MK 599 001
TU 599 002
TU
```

You can make it harsher with the same streaming path by changing presets, adding
mix noise, or running two generated QSOs in parallel with `generate-multi` or the
`qso_generator` helpers used by the tests.
