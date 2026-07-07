# CW live decoder

This build uses a token-based live transcript model.

## Current architecture

Audio is captured continuously and processed into tracked CW channels. Each tracked channel owns its persistent transcript state:

- committed tokens: stable decoded material
- tentative tokens: current mutable tail
- audio trim point: where the receiver may drop old audio while keeping overlap context

The application pipeline wires the layers together, but it no longer keeps a separate string-prefix incremental transcript map. The old `cw.app.incremental` module was removed.

## Public output

JSON channel output is token-based. Tokens have `kind`, optional `value`, optional `start_s` / `end_s`, and a `stable` flag.

Token kinds:

- `char`
- `unknown`
- `word_gap`
- `session_gap`

Text rendering, bracketed tentative display, and how a `session_gap` appears on screen are UI responsibilities.

## Example

```json
{"channel_id":1,"carrier_hz":866.667,"state":"active","tokens":[{"kind":"char","value":"C","start_s":1.2,"end_s":1.5,"stable":true},{"kind":"char","value":"Q","start_s":1.7,"end_s":2.1,"stable":false}]}
```

## Tests

```bash
pytest -q
```
