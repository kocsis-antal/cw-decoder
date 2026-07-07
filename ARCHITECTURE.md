# CW streaming receiver architecture

The runtime is organized as a one-way processing chain. There is no standalone
`events` package and no legacy `nextgen` runtime layer. Layers exchange their
own DTOs; user-facing channel-output JSONL is produced only at the application output
boundary.

## Runtime flow

```text
external audio source
  -> io
    -> receiving
      -> signal
        -> decoder
          -> selection
            -> app output/json
              -> ui
```

In words:

```text
io          source bytes/files/arrays -> AudioBlock stream
receiving   AudioBlock -> tracked ChannelSignal state snapshots
signal      ChannelSignal -> one or more SignalTrack analyses
decoder     SignalTrack -> DecodeResult / decoded text answers
selection   SelectionInput(all decoded texts) -> per-channel ChannelWinner
app         wires layers together and serializes ChannelOutput JSONL; optional debug side-channel
ui          sample dashboard consuming channel-output snapshots; debug JSONL viewer
```

## Layer responsibilities

### `io/`

Owns audio input adapters and IO-owned audio DTOs.

```text
input:  stdin/raw PCM/WAV/array/file-like stream
output: AudioBlock
```

It does not detect carriers, segment keying, decode Morse, or produce events.

### `receiving/`

Owns receiving state: audio history, spectral carrier candidates, channel
creation, channel drift/reacquire logic, and channel lifecycle state.

```text
input:  AudioBlock
output: ReceiveChunk(
          channels=ChannelSignal[],
          stats=ReceivingStats
        )
```

Receiving is channel-centric. A channel has a stable `channel_id`; the frequency
may drift, but the identity stays the same. A carrier that is not confirmed yet
is represented as the same DTO with `state=ChannelState.CANDIDATE`. If it later
becomes active, the same `channel_id` is reported with
`state=ChannelState.ACTIVE`.

Receiving is not an event bus. It emits state snapshots, not events. Downstream
application output serializes the current channel snapshots.

The receiving layer does not emit text or transcript output.

### `signal/`

Owns signal activity analysis. This is the boundary between "there is a
tracked channel" and "this is the channel keying as a digital MARK/SPACE/UNKNOWN
run sequence".

```text
input:  active ChannelSignal
output: SignalTrack[]
```

A single channel can produce multiple tracks, one per analyzer/profile.  The
current runtime has two signal families:

```text
threshold_activity    percentile-based energy threshold, multiple ratios; the uncertainty band emits UNKNOWN
energy_distribution   two-component log-energy distribution model, multiple posterior acceptance probabilities
```

A track does not carry the original audio or duplicate channel metadata; the
caller knows which ChannelSignal was analyzed. Each track contains runs:
`MARK`, `SPACE`, or `UNKNOWN`, each with a duration. Its only signal-layer
metric is `unknown_ratio`, the time ratio that the analyzer could not classify
as clean MARK or SPACE.

### `decoder/`

Owns Morse/text decoding from signal tracks.

```text
input:  SignalTrack
output: DecodeResult(answers=DecodedText[])
```

The decoder consumes only signal-layer run sequences. It does not know channel
identity, carrier frequency, source time, audio thresholds, or receiving state.
It also does not publish ranking metrics. Public decoder output contains decoded
text answers plus one decoder-owned quality fact: `unresolved_tokens`, the number
of Morse tokens that could not be mapped to a valid character. Invalid tokens are
rendered in the text with `□`; a literal `?` remains a valid Morse character.

The current runtime decoder preserves `UNKNOWN` as a decoder input state. It
branches each UNKNOWN run locally into MARK/SPACE alternatives; it does not use a
global "all unknowns are MARK" or "all unknowns are SPACE" switch.

### `selection/`

Owns ranking and per-channel winner selection.

```text
input:  SelectionInput(
          ChannelDecodedTexts[]
            TrackDecodedTexts[]
              DecodeResult[]
        )
output: ChannelWinner[]
```

The app passes the whole current decoded-text batch to selection.  Selection
itself groups by channel and by decoded text, scores internally, applies
hysteresis, and chooses one current winner for each channel.  Its internal
strategy uses decoder-owned unresolved-token counts first, then identical-answer
support, signal-family diversity, and neighboring-parameter stability.  It does
not use text length, character rate, lexicographic order, dictionaries, language
models, or character-frequency preferences.  Selection scores stay inside the
selection layer and are not part of decoder output or the default JSON output.

### `app/`

Owns composition and output formatting. It wires the layers together and emits
compact channel-output JSONL. It should not contain DSP or Morse decoding
policy.

```text
input:  AudioSource
output: channel JSONL or dashboard snapshots
```

The default JSON output contains only channel identity/frequency/state plus the
selected text for that channel when one exists. Decoder details and selection
scores do not enter the default JSON output.

For tuning, the app can also emit an explicit debug side-channel. That debug
output is opt-in and separate from the public channel JSON. It may include signal
track names, `unknown_ratio`, compact MARK/SPACE/UNKNOWN runs, decoder answers,
and selection decision components.

### `ui/`

A sample human dashboard/viewer. The normal dashboard consumes application
channel-output snapshots only and does not interpret transcript/session state,
decoder internals, or selection scores. The debug viewer consumes only the
separate debug JSONL side-channel.

### `tools/`

Developer/offline diagnostics. Runtime layers must not import tools.

## Dependency direction

```text
io          -> no runtime layer
receiving   -> io DTOs only, receiving internals
signal      -> receiving public DTOs
(signal no decoder dependency)
decoder     -> signal public DTOs, decoder internals
selection   -> decoder public DTOs, selection internals
app         -> composition root: may wire io/receiving/signal/decoder/selection/ui
ui          -> app.channel_output only
tools       -> diagnostic/dev use only; runtime does not import it
```

The architecture tests enforce this shape.

## Public DTOs

```text
io/models.py             AudioBlock, AudioSource

receiving/models.py      CarrierObservation, ChannelSignal, ChannelState,
                         ReceiveChunk, ReceivingStats

signal/config.py         SignalConfig, validate_signal_config
signal/models.py         SignalState, SignalRun, SignalTrack
signal/api.py            signal-layer public exports

decoder/api.py           Decoder, DecodeResult, DecodedText

decoder/config.py        decoder-layer settings only
decoder/run_decoder.py   RunDecoder implementation for SignalTrack inputs
decoder/timing.py        internal Morse timing model; no channel/carrier/audio DTOs

selection/models.py      SelectionInput, ChannelDecodedTexts, TrackDecodedTexts,
                         ChannelWinner, SelectionChunk

app/config.py            ProcessingConfig, app-level validation
app/channel_output.py    compact channel-output snapshots
app/jsonl.py             ChannelOutput -> JSONL wire format
app/debug_output.py      opt-in debug side-channel serialization
app/pipeline.py          layer composition pipeline
```

## Runtime imports

```python
from cw.io.raw_stream_source import RawPcmStreamSource
from cw.receiving.processor import Receiver
from cw.signal.segmenters import SignalSegmenterBank
from cw.decoder.run_decoder import RunDecoder
from cw.selection.arbiter import ChannelResultSelector
from cw.app.jsonl import channel_output_to_json
```

## Legacy diagnostic decoder code

Older carrier-window/offline decoder experiments were moved out of the runtime
`decoder/` layer into `tools/legacy_decoder/`. They remain available for manual
comparison and diagnostics, but the runtime processing chain does not import
them.
