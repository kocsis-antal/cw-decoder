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

- stable tokens: safe to commit in JSON/UI;
- carried/tentative tokens: retained audio context plus the current uncertain tail.

A token kept as audio context is deliberately not marked stable in public output.
It is present only so the next decode window has enough timing context; once a
stable word/session gap is available, context never reaches back across that gap.

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

## Receiving window trimming

Receiving does not use an independent per-channel rolling text/decode window.
The app-level stable-prefix split is the only normal source of channel audio
trimming: it can choose a Morse-token boundary and keep explicit context
characters.  A blind time window can cut into a retained character and make the
next decode worse than the previous one.

The only hard audio bound is `max_history_s`, the shared ring-buffer size.  It
is a memory/CPU safety limit, not a transcript or GUI retention setting.

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

The runtime signal path now has a single activity model: `energy_distribution`.
Its configured posterior acceptance probabilities create several variants, and
selection rewards agreement between those variants directly.

The winner score is intentionally small:

- `support_count`: how many energy-distribution variants produced the same best token stream;
- a small penalty for unresolved/unknown tokens. Unknowns are not an automatic veto;
- `neighbor_stability`: diagnostic tie-break help when adjacent probability variants agree.

Debug groups include `support_score`, `unknown_penalty_score`, and `final_score`.

## Tests

```bash
pytest -q
```
