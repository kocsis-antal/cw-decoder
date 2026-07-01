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
