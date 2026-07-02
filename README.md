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

Print channel/session lifecycle events for humans:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/two_sources.wav --events
```

Print the same lifecycle events as JSON Lines for tools, GUI prototypes, or
future websocket adapters:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/two_sources.wav --json-events
```

Each JSON line uses the stable schema `cw.stream.event.v1` and contains fields
such as `type`, `time_s`, `channel_id`, `session_id`, `carrier_hz`, `text`,
`score`, and `reason`. The external JSON field is named `type`; the internal
Python dataclass still uses `kind`.

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
header prints `frames_processed`, `retained_frames`, `pruned_frames`,
`active_pruned_frames`, and `finalized_pruned_frames` so this can be checked
during experiments. Use `--no-prune-finalized-sessions` when you want
full-history debug behaviour for completed sessions. `--history-margin-s`
controls the small safety overlap kept before the active session.

There is also an opt-in active-session pruning mode:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/qso.wav \
  --prune-committed-active-sessions
```

This keeps committed text in `SessionState`, trims the decode tail after the
last safely committed word, and can discard older frames even while the current
transmission is still active. It is intentionally not the default yet: WAV replay
regressions stay conservative, while long-running experiments can turn it on and
watch `active_pruned_frames`. `--active-history-margin-s` can override the
safety margin used for this active pruning path.

For code that wants to feed audio progressively, use `StreamProcessor` directly:

```python
from cw.streaming import StreamProcessor, StreamingConfig

processor = StreamProcessor(sample_rate=8000, config=StreamingConfig())
for block in audio_blocks:
    chunk = processor.push(block)
    for event in chunk.events:
        print(event.kind, event.text)

result = processor.finish()
```

There is also a block-source layer for replaying audio through the same shape a
future microphone or SDR adapter will use:

```python
from cw.streaming import StreamingConfig, WavFileSource, process_audio_source

config = StreamingConfig()
source = WavFileSource("samples/generated/two_sources.wav", config.input_block_ms)
result = process_audio_source(
    source,
    config,
    on_chunk=lambda chunk: [print(event.kind, event.text) for event in chunk.events],
)
```

`stream-sim` now reads WAV input through `WavFileSource` instead of loading the
whole signal before processing. With `--json-events`, events are printed as
chunks are processed, and only final end-of-stream events are flushed at EOF.
The same `AudioSource` interface is intended for later microphone, SDR, or
network audio adapters.

### Raw PCM stdin / virtual microphone input

`stream-stdin` reads raw mono or interleaved PCM from standard input and feeds it
into the same `StreamProcessor`. This is the first live-input path: the decoder
does not care whether the bytes come from a real microphone, a virtual audio
cable, a browser WebSDR monitor, or `ffmpeg`.

A deterministic replay smoke test can be made from any WAV with `ffmpeg`:

```bash
ffmpeg -hide_banner -loglevel error -i samples/generated/qso.wav -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --json-events \
    --prune-committed-active-sessions
```

On Linux/PulseAudio/PipeWire, a browser SDR can be captured through a virtual
sink and its monitor source:

```bash
pactl load-module module-null-sink sink_name=cw_sdr sink_properties=device.description=CW_SDR
# Route the browser/WebSDR tab to CW_SDR in pavucontrol or the system mixer.
parec -d cw_sdr.monitor --format=s16le --rate=8000 --channels=1 \
| docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --json-events \
    --prune-committed-active-sessions
```

Supported raw sample formats are `s16le`, `s16be`, `s32le`, `s32be`, `f32le`,
`f32be`, and `u8`. Use `--channels 2` for stereo input; the source averages
channels to mono before decoding. `--duration-s` is useful for bounded live
experiments or tests; without it, the command runs until stdin reaches EOF or it
is interrupted.

Live input has two squelch layers by default. First, `stream-stdin` requires
spectral peaks to stand at least `--min-peak-snr-db 14` dB above the per-frame
spectral floor before they can become carrier tracks. Second, a CW keying gate
keeps channels/sessions private until the decoded envelope looks like real keyed
CW rather than a steady carrier or short `E`/`T` noise. The stdin defaults are:

```text
--min-keying-tone-runs 3
--min-keying-chars 2
--min-keying-known-chars 2
--min-keying-active-duration-s 0.12
--min-keying-unit-s 0.03
--min-keying-duty-cycle 0.03
--max-keying-duty-cycle 0.92
--max-keying-score 120
--reject-et-only-sessions
--merge-short-gaps-ms 25
--drop-short-tones-ms 12
```

The last two options are a live deglitch layer. It repairs short dropouts inside
a dash before unit estimation and drops tiny tone spikes. This helps cases where
a clean `CQ` would otherwise become punctuation-like text such as `C=` because a
dash was briefly split into several dots. Experimental WPM/unit hypothesis
search is also available through `--unit-candidate-spread`,
`--unit-candidate-steps`, and `--punctuation-penalty`, but it is not enabled by
default yet because it still needs more real-signal tuning.

For very weak signals, lower the spectral gate, for example
`--min-peak-snr-db 10`; for a noisy empty band, raise it, for example
`--min-peak-snr-db 18`. If the first characters of a real station are being held
back too long, lower the keying gate, for example `--min-keying-chars 1` or
`--min-keying-tone-runs 2`. `stream-sim` keeps the older permissive lab defaults
unless you pass these options explicitly.

On Windows, run raw PCM pipes from `cmd.exe`, not PowerShell. PowerShell can
corrupt binary stdout/stderr pipelines. When host-side `ffmpeg` reads generated
project files from the repository root, use the host path under `app`, for
example `app\samples\generated\qso.wav`; inside the container the same file is
`/app/samples/generated/qso.wav`. A VB-CABLE/WebSDR command looks like this:

```cmd
ffmpeg -hide_banner -loglevel error -f dshow -i audio="CABLE Output (VB-Audio Virtual Cable)" -f s16le -ac 1 -ar 8000 - | docker compose -f infra\compose.yml run --rm -T cw python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --json-events --prune-committed-active-sessions
```


### Streaming core layout

The streaming code is split into a few focused modules:

```text
stream_models.py  public stream dataclasses and configuration validation
stream_stft.py    incremental overlapping STFT over a continuous sample stream
stream_decode.py  carrier/session decoding helpers from retained FFT frames
stream_tracker.py frame-by-frame spectral peak tracking, SNR squelch, carrier tracks
stream_keying.py CW-like keying gate before public channel/session events
stream_state.py   mutable ChannelState/SessionState lifecycle, commit bookkeeping, active prune anchors
stream_events.py  stable JSON/JSONL serialization for stream events
stream_sources.py block-based WAV/array/raw-PCM audio sources for replay and live input
stream_processor.py stateful push/finish live processor and pruning orchestration
streaming.py      compatibility/public API re-exports
```


The tracker keeps smoothed carrier estimates, marks tracks dormant after `--max-track-gap-s`, rejects per-frame peaks below `--min-peak-snr-db`, and filters long-term weak sidebands/noise through `--track-relative-threshold`. Peak separation, frame-to-frame track matching, and channel-level merging are separate configuration knobs, so experiments can distinguish "two spectral peaks", "same drifting carrier track", and "same GUI/logical channel". The mutable live state is explicit: `ChannelState` is the long-lived carrier/GUI anchor, while `SessionState` owns the current transmission's committed prefix, active prune boundary, and start/final bookkeeping. Decoded tempo/unit values still come from each `StreamSessionResult`, so a later reply on the same channel starts with a clean session-level timing estimate instead of inheriting the previous operator's tempo.

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
