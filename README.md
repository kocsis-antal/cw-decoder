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

By default, non-JSON streaming commands render the live human dashboard: one
row per carrier frequency, current state, and a rolling transcript of what
has been decoded on that carrier so far. Session/gap fragments are separated with extra whitespace, not with a
transmitted-looking CW character; the currently unstable preview suffix is
shown in square brackets, for example `CQ EV81OB EV81OB K   CQ [EV8]`. Decoded transcript rows stay
visible for a while even after the carrier goes dormant. The transcript is bounded for long monitoring sessions, but the dashboard now uses terminal width when available and a generous fallback otherwise; set `CW_VIEW_COLUMNS` or pass `COLUMNS` through Docker if you want an explicit width. Internal channel/session ids stay in
the event model but are not shown. No extra flag is needed:

```bash
docker compose -f infra/compose.yml run --rm cw python -m cw.cli stream-sim samples/generated/two_sources.wav
```

For the old verbose lifecycle log, use `--human-view events` or the backwards
compatible `--events` / `--human-events` alias. `--human-view compact` is kept
as an alias for the dashboard. To suppress live human output and keep only the
final decoded summary, use `--human-view off` or `--no-human-events`.

Print lifecycle events as JSON Lines for tools, GUI prototypes, or future
websocket adapters:

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


### Live receiver architecture

The live path is deliberately layered.  `NextgenStreamProcessor` is now mainly
an orchestrator; the operational pieces live behind small independent layers:

```text
raw PCM blocks
  -> short-window carrier detector
  -> channel tracker / reacquisition
  -> per-carrier decode window
  -> multi-hypothesis Morse candidate generators
  -> live hypothesis arbiter
  -> JSON events / dashboard renderer
```

The carrier detector uses a short window so new or weaker parallel stations can
appear quickly.  The per-carrier text decoder uses a longer window, because
phrases such as `CQ CQ DE ...` need more context than a carrier detector does.
Those two windows are controlled separately with `--live-carrier-window-s` and
`--live-decode-window-s`.  The default retained audio history is capped at
12 seconds; transcript state carries the already committed text, not an
unbounded audio buffer.

Candidate ranking is also split from signal detection.  Offline reports still
keep all generated candidates, but live output passes each session through a
live hypothesis arbiter.  The arbiter compares the parallel threshold/Viterbi/HMM
candidates and prefers a cleaner timing/text hypothesis when the evidence is
essentially tied, instead of blindly accepting the highest raw evidence score.
This avoids cases where a visibly good `CQ CQ ...` candidate is replaced by a
slightly higher-evidence but worse-looking alternative.

### Raw PCM stdin / virtual microphone input

`stream-stdin` reads raw mono or interleaved PCM from standard input and feeds it
into the same `StreamProcessor`. This is the first live-input path: the decoder
does not care whether the bytes come from a real microphone, a virtual audio
cable, a browser WebSDR monitor, or `ffmpeg`.

A deterministic replay smoke test can be made from any WAV with `ffmpeg`.
When host-side `ffmpeg` is started from the repository root, generated samples
are under `app/samples/...`; inside the container the same file is
`samples/...`:

```bash
ffmpeg -hide_banner -loglevel error -i app/samples/generated/qso.wav -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --prune-committed-active-sessions
```

On Linux/PulseAudio/PipeWire, install the small PulseAudio-compatible command
line tools if `pactl`/`parec` are missing:

```bash
sudo apt install --no-install-recommends pulseaudio-utils pavucontrol
```

The quickest Linux test is to capture the default output monitor. This listens
to whatever currently goes to the default speakers/headphones:

```bash
pactl get-default-sink
pactl list short sources | grep monitor

ffmpeg -hide_banner -loglevel error \
  -f pulse -i "$(pactl get-default-sink).monitor" \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --prune-committed-active-sessions \
    --live-stats-interval-s 5
```

Without `--json-events`, live input renders the human dashboard: one row per
carrier frequency, the current state, and a rolling transcript of what has
been decoded on that carrier so far. Gap-separated transcript fragments use extra whitespace rather than a transmitted-looking separator; unstable preview text is bracketed, so Morse punctuation such as `/` and `?` remains unambiguous. Internal channel/session ids are kept in the
event model but hidden from this view. With
`--json-events`, stdout is pure JSONL for servers, logging, future web viewers,
or deep debugging. `--live-stats-interval-s` prints progress to stderr without
polluting JSONL; it also prints `rms_dbfs` and `peak_dbfs`, which are useful for
spotting muted or very low-level monitor audio. A very quiet stream, for example
below roughly `-45 dBFS`, is worth turning up before tuning decoder thresholds.

For general live SDR work, avoid tuning the decoder to one hard-coded audio
pitch range unless the capture is intentionally about one fixed station. Moving
the receiver shifts where stations land in the audio passband, so frequency
limits such as `--min-tone-hz` / `--max-tone-hz` are best treated as temporary
debug tools, not normal live defaults. For clean but misread CW, try a mild unit
hypothesis search before changing the core decoder:

```bash
--unit-candidate-spread 0.35 --unit-candidate-steps 7 --punctuation-penalty 4
```

For a cleaner browser/WebSDR setup, create a virtual sink and route only the SDR
tab to it in `pavucontrol` or the system mixer:

```bash
pactl load-module module-null-sink sink_name=cw_sdr sink_properties=device.description=CW_SDR
# Route the browser/WebSDR tab to CW_SDR in pavucontrol or the system mixer.
parec -d cw_sdr.monitor --format=s16le --rate=8000 --channels=1 \
| docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --prune-committed-active-sessions \
    --live-stats-interval-s 5
```

To hear the virtual sink while decoding it, loop it back to the real default
output:

```bash
pactl load-module module-loopback source=cw_sdr.monitor sink=@DEFAULT_SINK@ latency_msec=50
```

For troubleshooting, first record a short WAV and decode it as a normal replay.
This separates audio routing problems from decoder problems:

```bash
ffmpeg -hide_banner -loglevel info \
  -f pulse -i "$(pactl get-default-sink).monitor" \
  -t 10 -ac 1 -ar 8000 \
  app/samples/generated/linux_monitor_test.wav

docker compose -f infra/compose.yml run --rm cw \
  python -m cw.cli stream-sim samples/generated/linux_monitor_test.wav --events
```

If the recording is silent or contains the wrong application, the browser is
probably playing to a different sink than the one being monitored. Check
`pavucontrol`'s Playback tab while the WebSDR is playing.

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

Live final events now also have a small anti-regression guard. If the decoder
had already emitted stable live text for a session and the closing re-decode is
more than `--final-text-regression-margin` score points worse, the final event
keeps the stable text instead of replacing it with trailing-noise garbage. Live
`SESSION_FINAL` events also respect `--max-final-score`; low-quality closed
sessions are used internally for stale-history suppression and are closed with
`reason=quality_suppressed` and empty text, so lifecycle consumers do not keep
an open session while the noisy text stays out of the transcript.

For very weak signals, lower the spectral gate, for example
`--min-peak-snr-db 10`; for a noisy empty band, raise it, for example
`--min-peak-snr-db 18`. If the first characters of a real station are being held
back too long, lower the keying gate, for example `--min-keying-chars 1` or
`--min-keying-tone-runs 2`. `stream-sim` keeps the older permissive lab defaults
unless you pass these options explicitly.

Stopping live input with Ctrl+C normally flushes final sessions. On a noisy live
source that final flush can be slow because there may be many retained
candidates. Use `--no-finalize-on-interrupt` for quick live experiments where
you want Ctrl+C to exit immediately:

```bash
... | docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --no-finalize-on-interrupt
```

### Reproducible live captures

Live radio is not repeatable, so decoder changes should be driven by captured
raw audio samples rather than by a one-off console log. `stream-stdin` can write
the exact PCM byte stream it receives while it is decoding it:

```bash
mkdir -p app/samples/live
ts=$(date +%Y%m%d-%H%M%S)

ffmpeg -hide_banner -loglevel error \
  -f pulse -i "$(pactl get-default-sink).monitor" \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
    python -m cw.cli stream-stdin \
      --sample-rate 8000 \
      --sample-format s16le \
      --json-events \
      --prune-committed-active-sessions \
      --live-stats-interval-s 5 \
      --no-finalize-on-interrupt \
      --capture-raw "samples/live/${ts}.s16le" \
  2> "app/samples/live/${ts}.stats.log" \
| tee "app/samples/live/${ts}.events.jsonl"
```

The `--capture-raw` path is inside the container, so use `samples/live/...`. The
events and stats redirections are host-side, so use `app/samples/live/...` from
the repository root. The captured file can be replayed without any live audio
routing:

```bash
docker compose -f infra/compose.yml run --rm cw \
  python -m cw.cli stream-raw-file samples/live/${ts}.s16le \
    --sample-rate 8000 \
    --sample-format s16le \
    --prune-committed-active-sessions
```

For multi-station recordings, avoid using a narrow `--min-tone-hz` /
`--max-tone-hz` range unless the sample is intentionally about one fixed audio
pitch. Tuning the receiver changes where stations land in the audio passband.
Likewise, `--max-tracks 3` is a debug cap; it can hide weaker keyed stations
when a few strong carriers or audio artefacts are present.

On Windows, run raw PCM pipes from `cmd.exe`, not PowerShell. PowerShell can
corrupt binary stdout/stderr pipelines. When host-side `ffmpeg` reads generated
project files from the repository root, use the host path under `app`, for
example `app\samples\generated\qso.wav`; inside the container the same file is
`/app/samples/generated/qso.wav`. A VB-CABLE/WebSDR command looks like this:

```cmd
ffmpeg -hide_banner -loglevel error -f dshow -i audio="CABLE Output (VB-Audio Virtual Cable)" -f s16le -ac 1 -ar 8000 - | docker compose -f infra\compose.yml run --rm -T cw python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --prune-committed-active-sessions --live-stats-interval-s 5
```

PowerShell can still be used as a launcher by wrapping the whole binary pipe in
`cmd /d /c "..."`, but the pipe itself should stay inside `cmd.exe`.

For a human-readable console view, simply omit `--json-events`; that is the
default live dashboard. If another process needs the raw event stream, keep
JSONL and pipe it into the same dashboard viewer:

```bash
python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --json-events \
| python -m cw.cli view-events
```

Saved JSONL can be viewed the same way:

```bash
python -m cw.cli view-events samples/live/20260703-104548.events.jsonl
```

Use `--human-events` only for the old verbose lifecycle log with channel/session
debug details.


### Streaming core layout

The live path now has one main engine: a low-latency carrier-centric nextgen streamer.
The older STFT live processor and its dedicated tracker/state modules were
removed from the public runtime so replay, stdin, and tests exercise the same
logic. Live processing uses a short rolling decode window (`--live-decode-window-s`, default 3 s) and keeps the accumulated transcript in stream state, instead of repeatedly treating the last 12 seconds as a fresh batch decode.

```text
nextgen.py        carrier demodulation, lattice/Symbol-HMM candidates, session decoding
nextgen_stream.py low-latency live channel/session state machine around the nextgen decoder
stream_models.py  stream dataclasses and configuration validation
stream_events.py  stable JSON/JSONL serialization for stream events
stream_sources.py block-based WAV/array/raw-PCM audio sources for replay/live input
stream_decode.py  shared Morse timing helpers still used by diagnostics and nextgen candidates
streaming.py      small public facade around NextgenStreamProcessor and audio sources
event_view.py     human-readable rendering for JSONL stream events
```

The intended architecture is now explicit: carrier detection and text decoding
live in `nextgen.py`; live concerns such as short-window decoding, stable commits, previews, progress
updates, and channel dormancy live in `nextgen_stream.py`; source handling and
wire-format rendering are separate.

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

## Next-generation raw decoder

`decode-raw` is the first carrier-centric decoder path.  It is intended to
become the default decoder core after it has enough live-regression coverage.
Unlike the streaming STFT decoder, it demodulates each carrier separately:

```text
raw PCM -> carrier detection -> per-carrier baseband mix -> low-pass envelope
        -> symbol-HMM / Viterbi / threshold candidates -> session text
```

It is deliberately content-neutral.  It does not reward `CQ`, `DE`, callsigns,
Q-codes or contest exchanges.  Candidate selection uses timing quality,
envelope confidence, known-character evidence and uncertainty penalties.

Example with automatic carrier detection:

```bash
python -m cw.cli decode-raw samples/live/20260702-184936.s16le
```

Example with explicit carriers and a slice:

```bash
python -m cw.cli decode-raw samples/live/20260702-184936.s16le \
  --start-s 18 \
  --duration-s 18 \
  --carrier 600 \
  --carrier 1600
```

The output is session-oriented: one carrier may contain several timed
transmissions, and each session keeps its own ranked threshold/unit candidates.
This avoids treating a long capture as a single huge text blob.

Useful tuning knobs are intentionally generic: `--session-gap-s` controls how
large a carrier-local silence splits sessions, and `--min-session-evidence-score`
suppresses tiny low-evidence fragments. The raw defaults match the current live
capture workflow: `--sample-rate 8000`, `--sample-format s16le`,
`--max-tone-hz 3000`, dynamic threshold candidates and short dropout repair are
all enabled without extra arguments.

## Direct Symbol-HMM decoder

The nextgen path now includes a direct, duration-aware Symbol-HMM candidate path.
Unlike the threshold, activity-Viterbi, and timing-lattice layers, this decoder
does not start from a pre-cut tone/gap run list.  It searches the carrier
envelope probability frames directly for Morse duration states: dit tone, dah
tone, element gap, letter gap, and word gap.  A small beam keeps alternative
state paths alive until the session candidate is scored.

This path is used by the same `decode-raw`, `stream-stdin`, `stream-raw-file`,
and `stream-sim` code paths; it is not tied to a file/offline source.  The
direct dit/dah/gap Symbol-HMM now participates as a normal candidate source.
It also runs a second unit pass derived from its own best duration path, so a
rolling window or mixed-speed capture is not locked to one initial WPM guess.
When the direct candidates are still weak, an additional complete-character
Morse template beam (`char-hmm`, displayed as `det=hmm`) can try valid Morse
character duration templates directly against the probability frames.  It
remains content-neutral: no `CQ`, `DE`, callsign, Q-code, `5NN`, or `73` bonus
exists anywhere in the scoring.

Diagnostic switches:

```bash
--no-symbol-hmm-decoding
--symbol-hmm-beam-width 16
--symbol-hmm-max-candidates 3
--symbol-hmm-unit-spread 0.18
--symbol-hmm-unit-steps 3
--symbol-hmm-transition-penalty 0.18
```

In human `decode-raw` output, direct Symbol-HMM rows show as `det=hmm`.

## Nextgen Viterbi activity decoder

The carrier-centric nextgen decoder now has a default two-state Viterbi activity path next to the hard threshold candidates.  It treats each carrier envelope frame as probabilistic evidence for either `tone` or `gap`, then chooses the most likely whole-window state path with a transition penalty.  A brief fade inside a dash can stay tone if two extra tone/gap transitions would be less likely than the weak frames; a real silent Morse gap is still preserved when the accumulated silence evidence wins.

This replaces the previous one-way hysteresis decision as the default soft path.  It is still content-neutral: no `CQ`, `DE`, callsign, Q-code, `5NN`, or `73` bonus is used.

Useful tuning switches for live or replay experiments:

```bash
--no-soft-activity
--viterbi-transition-penalty 1.15
--soft-bridge-min-probability 0.18
--soft-bridge-max-gap-ms 90
--soft-bridge-gap-units 1.6
```

The Viterbi path is enabled by default but carries a small candidate penalty, so a clean hard-threshold decode with comparable evidence still wins.  Viterbi candidates are visible in `decode-raw` rank tables with `det=vit`.

## Nextgen timing lattice / beam decoder

After the threshold and Viterbi activity gates produce candidate tone/gap runs, the
nextgen decoder keeps a small timing lattice alive around Morse boundary
cases.  A run that sits close to the dot/dash boundary can be interpreted as
both `.` and `-`; a gap close to an element/letter or letter/word split can keep
both neighbouring interpretations until the whole token sequence is scored.

This is still content-neutral: there is no `CQ`, `DE`, callsign, Q-code, `5NN`,
or `73` bonus.  The beam score uses only timing distance, unknown/punctuation
penalties, envelope confidence, and amount of decoded signal evidence.  Word gaps
that survive the timing lattice get a small generic evidence reward because they
come from received spacing, not from QSO semantics.

It is enabled by default in `decode-raw`, `stream-stdin`, `stream-raw-file`, and
`stream-sim`.  Diagnostic switches:

```bash
--no-lattice-decoding
--lattice-beam-width 12
--lattice-max-candidates 3
--lattice-tone-margin-units 0.45
--lattice-gap-margin-units 0.60
```

In human `decode-raw` output, lattice-derived rows show as `det=lat`, hard threshold rows as `det=thr`, Viterbi activity rows as `det=vit`, and direct Symbol-HMM rows as `det=hmm`.
