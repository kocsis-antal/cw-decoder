# CW live decoder

This build uses token-based channel output with a stateless stable-prefix split.

## Current architecture

Audio is captured continuously and processed into tracked CW channels:

```text
audio input
  -> receiving: carrier observations, stable channel ids, per-channel audio windows
  -> signal: MARK / SPACE / UNKNOWN activity tracks
  -> decoder: decoded token candidates
  -> selection: one current winner per channel
  -> app: stable/tentative split and JSON output
```

The transcript logic is not a per-channel memory object.  It receives the
currently selected token sequence and splits it into two parts:

- stable tokens: safe to mark stable in JSON;
- carried/tentative tokens: still part of the current uncertain tail.

The split also produces an optional absolute audio trim time.  The app feeds
that trim point back into receiving, so old stable audio does not need to be
reprocessed.  Receiving owns only the trim point; it does not know about text or
Morse tokens.

## Carrier/channel frequency settings

The receiving layer keeps separate controls for separate concepts:

- `carrier_peak_separation_hz`: minimum spacing between simultaneous FFT carrier peaks;
- `channel_match_hz`: normal same-channel tracking tolerance;
- `channel_reacquire_hz`: wider tolerance for reacquiring a recently missing channel.

These are intentionally not aliases of each other.

## Public output

JSON channel output is token-based. Tokens have `kind`, optional `value`, optional
`start_s` / `end_s`, and a `stable` flag.

Token kinds:

- `char`
- `unknown`
- `word_gap`
- `session_gap`

Text rendering, bracketed tentative display, and how a `session_gap` appears on
screen are UI responsibilities.

## Example

```json
{"channel_id":1,"carrier_hz":866.667,"state":"active","tokens":[{"kind":"char","value":"C","start_s":1.2,"end_s":1.5,"stable":true},{"kind":"char","value":"Q","start_s":1.7,"end_s":2.1,"stable":false}]}
```

## Selection scoring

Selection is stateless and ranks only the current decoded candidates.  It does
not use text semantics, gap-length rescoring, signal separability, or previous
winners.

The winner score is intentionally small:

- one vote per analyzer family, so multiple threshold variants from the same family do not outvote another independent family;
- a small penalty for unresolved/unknown tokens. Unknowns are not an automatic veto.

Raw `support_count` is still emitted in debug output, but it is diagnostic data;
it is not the primary ranking vote. Debug groups also include
`family_support_score`, `unknown_penalty_score`, and `final_score`.

## Tests

```bash
pytest -q
```
