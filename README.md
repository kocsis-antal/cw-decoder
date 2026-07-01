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

## Carrier spacing benchmark

The spacing benchmark generates two overlapping CW sources at controlled audio
frequency offsets and runs the streaming tracker on each case. It is meant to
answer questions such as: below what spacing should two tones be treated as one
crowded channel, and from what spacing can two parallel transmissions be decoded
separately?

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli spacing-benchmark --expect
```

Default expectations are intentionally conservative:

```text
40 Hz          merge / not reliably separable
60-80 Hz       ambiguous observation zone
100 Hz and up  should split and decode both sources
```

The benchmark now uses two STFT paths by default: a shorter decode STFT
(`--stream-frame-ms 30`, `--stream-hop-ms 5`) for Morse timing, and a longer
tracker STFT (`--tracker-frame-ms 80`, `--tracker-hop-ms 10`) for carrier
spacing. This keeps the frequency resolution needed for close carriers without
forcing the Morse edge detector to use the same long frame. Useful knobs are
`--deltas`, `--merge-below-hz`, `--split-from-hz`, `--stream-frame-ms`,
`--stream-hop-ms`, `--tracker-frame-ms`, and `--tracker-hop-ms`. Tracker spacing
can also be tuned with separate thresholds: `--peak-min-separation-hz` controls
how close two FFT peaks may be before one suppresses the other, `--track-match-hz`
controls how far a peak may move frame-to-frame and still match an existing
carrier track, and `--channel-merge-hz` controls how close final carrier
candidates may be before they are treated as the same channel.
`--min-separation-hz` remains the legacy default used when these more specific
values are omitted.

## Streaming simulation

`stream-sim` does not split the WAV into separate files. It reads the audio as a
continuous sample stream, feeds it into an internal ring-buffer-like STFT, and
produces overlapping FFT frames. Carrier detection is now frame-by-frame and can use a separate, longer FFT path
from the Morse decoder: each tracker FFT frame yields spectral peaks, a
lightweight tracker links those peaks into smoothed carrier tracks, and each
active carrier track drives its own channel/session decode using the shorter
decode frames. The retained decode-frame history is still used to decode the
current session, which keeps the WAV replay deterministic while moving the
internals toward microphone input.

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


### Streaming core layout

The streaming code is split into a few focused modules:

```text
stream_models.py  public stream dataclasses and configuration validation
stream_stft.py    incremental overlapping STFT over a continuous sample stream
stream_decode.py  carrier/session decoding helpers from retained FFT frames
stream_tracker.py frame-by-frame spectral peak tracking into carrier tracks
stream_state.py   mutable ChannelState/SessionState lifecycle and commit bookkeeping
streaming.py      stream orchestration and pruning
```


The tracker keeps smoothed carrier estimates, marks tracks dormant after `--max-track-gap-s`, and filters long-term weak sidebands/noise through `--track-relative-threshold`. Peak separation, frame-to-frame track matching, and channel-level merging are separate configuration knobs, so experiments can distinguish "two spectral peaks", "same drifting carrier track", and "same GUI/logical channel". The mutable live state is explicit: `ChannelState` is the long-lived carrier/GUI anchor, while `SessionState` owns the current transmission's committed prefix and start/final bookkeeping. Decoded tempo/unit values still come from each `StreamSessionResult`, so a later reply on the same channel starts with a clean session-level timing estimate instead of inheriting the previous operator's tempo.

`cw.streaming` still re-exports the public stream API, so existing imports such
as `from cw.streaming import StreamingConfig, StreamingSTFT` continue to work.

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
