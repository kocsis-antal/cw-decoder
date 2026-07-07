# CW Morse streaming receiver — layered core

A csomag célja egy CW/Morse vevő feldolgozó lánc, tiszta réteghatárokkal.
A korábbi labor/generátor/benchmark/contest mellékágak nincsenek a runtime főútban.

## Fő folyamat

```text
raw/WAV/audio source
  -> io                 # közös AudioBlock folyam
  -> receiving          # carrier/channel felismerés, text nélkül
  -> signal             # csatornajel MARK/SPACE/UNKNOWN futamok
  -> decoder            # dekódolt szövegválaszok
  -> selection          # csatornánként aktuális nyertes kiválasztása
  -> app JSONL output   # csatorna snapshot + nyertes szöveg
  -> HumanDashboardRenderer / view-output
```

A `receiving` réteg nem ad eseményeket és nem ad szöveget. Csatornaállapot-snapshotokat ad.
A `signal` réteg digitális jelfutamokat készít. Egy csatornából több track is készülhet: jelenleg több percentilis-küszöbös és több eloszlásalapú energia-modell fut rajta. A `decoder` ezekből dekódolt szövegválaszokat állít elő.
A `selection` a teljes aktuális dekódolt szöveghalmazból választ csatornánként aktuális nyertest. Az app ebből ír minimális JSONL csatorna-outputot.

## Fő parancsok

Raw PCM stdinből, dashboarddal:

```bash
ffmpeg -hide_banner -loglevel error \
  -f pulse -i "$(pactl get-default-sink).monitor" \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le \
    --capture-raw "samples/stream/example.s16le"
```

Gépi JSONL kimenettel:

```bash
python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --json-output
```

Mentett raw fájl visszajátszása JSON-ba:

```bash
python -m cw.cli stream-raw-file samples/stream/example.s16le \
  --sample-rate 8000 \
  --sample-format s16le \
  --json-output > channels.jsonl
```

JSONL megjelenítése dashboardként:

```bash
python -m cw.cli view-output channels.jsonl
```

Debug JSONL írása külön fájlba, a normál dashboard megtartása mellett:

```bash
python -m cw.cli stream-raw-file samples/stream/example.s16le \
  --sample-rate 8000 \
  --sample-format s16le \
  --debug-json-output debug.jsonl
```

Debug JSONL emberi olvasása:

```bash
python -m cw.cli view-debug-output debug.jsonl
```

## Publikus output JSON

A futás csatornánként aktuális snapshotokat ír. Az alap JSON szándékosan kicsi:

```json
{"channel_id":4,"carrier_hz":702.0,"state":"active","text":"CQ"}
```

Mezők:

- `channel_id`: stabil csatornaazonosító, nem a frekvencia.
- `carrier_hz`: a csatorna aktuálisan követett vivőfrekvenciája.
- `state`: `candidate`, `active`, `dormant` vagy `dropped`.
- `text`: az adott csatornához tartozó aktuális nyertes szöveg, ha van.

Nincs külön `schema`, `type`, `time_s`, `reason`, `relative_power`, score vagy decoder internals az alap JSON-ben. Ezek belső rétegadatok.

## Debug output

A debug kimenet külön side-channel: `--debug-json-output` kapcsolóval kérhető, és nem változtatja meg a publikus csatorna JSON-t. Ha a kapcsolóhoz nem adsz fájlnevet, vagy `-` értéket adsz, a debug JSONL stderr-re megy.

Egy debug sor csatornánként mutatja:

- a csatorna aktuális állapotát;
- az összes signal tracket: analyzer név, `unknown_ratio`, rövidített futamsor (`M80 S40 U20` milliszekundumban);
- trackenként a dekóder-válaszokat és `unresolved_tokens` értéket;
- a selection belső döntési indokait: azonos szöveg támogatása, család-diverzitás, szomszédos paraméter-stabilitás, hiszterézis.

Példa emberi debug nézet:

```text
DEBUG t=1.25s ch4 702.0Hz active selected="CQ"
  signals:
    energy_distribution:p=0.80 unknown=0.040 runs=M80 S90 M240
      run_decoder: "CQ"/bad=0
  selection:
   * "CQ" bad=0 support=3 families=2 neighbors=1
```

## Architektúra

Részletesen: `ARCHITECTURE.md`.

## Signal track családok

A signal réteg egy aktív csatornából track-halmazt készít. A track nem tud a dekóderről, csak `MARK` / `SPACE` / `UNKNOWN` futamokat ad és egy `unknown_ratio` időarányt.

Jelenlegi családok:

- `threshold_activity:threshold=...`: percentilis alapú energia-küszöbölés több aránnyal; a küszöb körüli bizonytalan sávot `UNKNOWN` futamként adja ki.
- `energy_distribution:p=...`: kétkomponensű log-energia eloszlásmodell. A `p` nem fix energiahatár, hanem posterior elfogadási valószínűség; ami ennél bizonytalanabb, `UNKNOWN`.

## Decoder

A runtime dekóder bemenete kizárólag `SignalTrack`: `MARK` / `SPACE` / `UNKNOWN`
futamok. Nem lát csatornaazonosítót, vivőfrekvenciát, audio-küszöböt vagy
receiving állapotot. Az `UNKNOWN` állapotot nem nyeli le gapként, és nem globális
kapcsolóval értelmezi: minden UNKNOWN futam külön, lokálisan ágazhat MARK vagy
SPACE irányba.

A publikus decoder output dekódolt szövegválaszokból áll. Egyetlen
dekóder-minőségi adat tartozik hozzájuk: `unresolved_tokens`, vagyis hány
morze-tokenből nem lett érvényes karakter. Ezek helyén a szövegben `□` látszik.
A `?` rendes, érvényes morze-karakter marad. A pontozás és
csatornanyertes-választás a `selection` réteg dolga.

## Selection

A selection réteg batch-alapon dolgozik: egy lépésben megkapja az összes aktuális csatornát, csatornánként az összes signal tracket és azok dekóder-válaszait. Ő csoportosít, pontoz és választ; a pontszám belső ügy.

A jelenlegi stratégia sorrendje:

- kevesebb `unresolved_tokens`;
- azonos szöveg több támogatással;
- több signal-modellcsalád támogatása;
- szomszédos paraméterbeállítások stabilitása;
- időbeli hiszterézis kis különbségnél.

Nem használ szótárat, nyelvi valószínűséget, karakterhosszt, karakter/másodperc mutatót, betűgyakoriságot vagy ABC szerinti döntést.

## Live incremental commit update

- `--channel-reacquire-hz` now defaults to `0`, which means the conservative channel-match/min-separation based reacquire range is used. It no longer defaults to a broad 220 Hz window that could merge two valid CW carriers.
- `--peak-min-separation-hz` now defaults to `0`, so the normal `--min-separation-hz` governs carrier separation unless explicitly overridden.
- Channel-level alias dropping is disabled by default with `--channel-alias-hz 0`; correlated carrier suppression in the observer remains available.
- Signal and selection hard gates are permissive by default. They are still available as explicit knobs, but the default path does not discard plausible weak candidates before later layers can inspect them.
- `--processing-workers` is kept only as a deprecated compatibility argument. Signal/decoder processing is single-threaded after input capture.
- Stable per-channel text prefixes are now committed with `--commit-hold-chars` trailing characters kept tentative while the channel is active.
- The receiver uses committed character end times as safe audio trim points, so it stops reprocessing old committed audio without cutting a Morse character in half.
- `--no-incremental-commit` disables this behavior for comparison/debugging.

## 2026-07 live readability and silence-gate update

- Active-channel output now renders the mutable tail in square brackets.  The unbracketed prefix is the channel's committed text; the bracketed suffix may still be revised by later audio.
- Incremental commit now cuts active audio primarily on decoded word-boundary gaps.  This avoids locking an early, still-changing character interpretation inside a callsign and preserves the separator when the next retained window starts with a word gap.
- Carrier observation has an additional spectral SNR gate: `--carrier-min-snr-db` (default `6.0`).  A relative peak threshold alone accepts arbitrary FFT maxima during silence, because every noise window has a strongest bin; the SNR gate prevents those noise peaks from becoming channels.
- The Python worker-thread experiment remains removed from the hot path; realtime performance comes from not reprocessing committed audio.
