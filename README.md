# CW Decoder

Valós idejű, többrétegű CW/Morse vevő és dekóder.

A program nyers hangfolyamból vagy hangfájlból egyszerre több CW vivőt képes követni, a jelet `MARK` / `SPACE` / `UNKNOWN` futamokra bontani, több dekódolási jelöltet előállítani, majd csatornánként kiválasztani az aktuális eredményt.

A normál használathoz Docker ajánlott. A host oldalon csak Docker és – élő hang becsatornázásához – `ffmpeg` szükséges.

---

## 1. Gyors indulás

A repository gyökeréből:

```bash
docker compose -f infra/compose.yml build
```

A fő CLI:

```text
python -m cw.cli <parancs>
```

Elérhető fő parancsok:

```text
stream-stdin         nyers PCM hangfolyam dekódolása stdin-ről
stream-raw-file      korábban mentett nyers PCM visszajátszása
stream-wav           WAV/audio fájl dekódolása
view-output          JSONL kimenet megjelenítése emberi dashboardként
view-debug-output    debug JSONL megjelenítése olvasható formában
```

Segítség:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli --help
```

Egy konkrét parancs kapcsolói:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-stdin --help
```

---

## 2. Élő vétel

A dekóder nem kezel közvetlenül hangkártya-API-kat.

A külső hangforrást – rádió, SDR-program, WebSDR, böngésző, virtuális audiokábel vagy fizikai hangbemenet – az `ffmpeg` alakítja át egységes nyers PCM folyamra:

```text
hangforrás
    -> ffmpeg
    -> mono / 8000 Hz / s16le PCM
    -> stdin
    -> cw.cli stream-stdin
```

A két oldal paramétereinek egyezniük kell:

```text
sample rate:   8000 Hz
channels:      1 (mono)
sample format: s16le
```

### Alap élő parancs

A bemenetet előállító `ffmpeg` parancs végét ehhez kell csövezni:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-stdin \
    --sample-rate 8000 \
    --sample-format s16le
```

A normál kimenet egy élő, ember számára olvasható dashboard.

---

# Windows hang becsatornázása

## 3. Windows + VB-Audio Virtual Cable

A tipikus felépítés:

```text
rádió / SDR / böngésző / lejátszó
        |
        v
CABLE Input (VB-Audio Virtual Cable)
        |
        | virtuális audiokábel
        v
CABLE Output (VB-Audio Virtual Cable)
        |
        v
ffmpeg
        |
        v
CW Decoder
```

A VB-CABLE elnevezése elsőre fordítottnak tűnhet:

- **CABLE Input**: ide küldi a hangot a lejátszó alkalmazás;
- **CABLE Output**: ezt olvassa felvevő bemenetként az `ffmpeg`.

### 3.1. A hangforrás átirányítása

A rádió-, SDR- vagy böngészőalkalmazás hangkimenetét állítsd erre:

```text
CABLE Input (VB-Audio Virtual Cable)
```

Windows alatt ez történhet:

- közvetlenül az alkalmazás saját audio output beállításában; vagy
- a Windows alkalmazásonkénti hangerő-/kimenetválasztójában.

### 3.2. FFmpeg audioeszközök listázása

```cmd
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

A listában keresd például ezt:

```text
CABLE Output (VB-Audio Virtual Cable)
```

### 3.3. Élő dekódolás VB-CABLE-ről

A repository gyökerében, Windows `cmd` vagy PowerShell alatt:

```cmd
ffmpeg -hide_banner -loglevel error -f dshow -i audio="CABLE Output (VB-Audio Virtual Cable)" -f s16le -ac 1 -ar 8000 - | docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le
```

### 3.4. Élő dekódolás és a teljes nyers hang mentése

```cmd
ffmpeg -hide_banner -loglevel error -f dshow -i audio="CABLE Output (VB-Audio Virtual Cable)" -f s16le -ac 1 -ar 8000 - | docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le --capture-raw samples/live/capture.s16le
```

A `samples/live/` könyvtár helyi munkaterület: az ott készülő felvételek és debug fájlok nincsenek verziókezelésre szánva.

### 3.5. Közvetlen fizikai hangbemenet Windows alatt

Először listázd az eszközöket:

```cmd
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

Majd a megfelelő bemenet nevét használd:

```cmd
ffmpeg -hide_banner -loglevel error -f dshow -i audio="A HANGBEMENET PONTOS NEVE" -f s16le -ac 1 -ar 8000 - | docker compose -f infra/compose.yml run --rm -T cw python -m cw.cli stream-stdin --sample-rate 8000 --sample-format s16le
```

---

# Linux hang becsatornázása

## 4. Linux + PipeWire / PulseAudio kompatibilis rendszer

A legtöbb modern Linux desktopon a PipeWire PulseAudio-kompatibilis felületén keresztül használható az `ffmpeg`, a `pactl` és a `pavucontrol`.

### 4.1. Elérhető hangforrások

```bash
pactl list short sources
```

Közvetlen bemenet vagy monitor hallgatása:

```bash
ffmpeg -hide_banner -loglevel error \
  -f pulse -i "FORRAS_NEVE" \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
    python -m cw.cli stream-stdin \
      --sample-rate 8000 \
      --sample-format s16le
```

## 4.2. A teljes alapértelmezett hangkimenet figyelése

Gyors megoldásként az alapértelmezett hangkimenet monitorja is használható:

```bash
ffmpeg -hide_banner -loglevel error \
  -f pulse -i "$(pactl get-default-sink).monitor" \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
    python -m cw.cli stream-stdin \
      --sample-rate 8000 \
      --sample-format s16le
```

Ez **minden**, az adott kimenetre küldött rendszerhangot tartalmazhatja. CW vételhez általában jobb külön virtuális audio útvonalat használni.

## 4.3. Virtuális audiokábel Linux alatt

Hozz létre egy külön virtuális sinket:

```bash
pactl load-module module-null-sink \
  sink_name=cw_sink \
  sink_properties=device.description=CW_Decoder
```

A parancs egy modulazonosítót ír ki. Ezzel később eltávolítható:

```bash
pactl unload-module MODULAZONOSITO
```

Ezután a `pavucontrol` **Playback** lapján irányítsd a rádió-, SDR- vagy böngészőalkalmazást a következő kimenetre:

```text
CW_Decoder
```

A virtuális sink monitorját az `ffmpeg` így olvassa:

```bash
ffmpeg -hide_banner -loglevel error \
  -f pulse -i cw_sink.monitor \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
    python -m cw.cli stream-stdin \
      --sample-rate 8000 \
      --sample-format s16le
```

Mentéssel együtt:

```bash
ffmpeg -hide_banner -loglevel error \
  -f pulse -i cw_sink.monitor \
  -f s16le -ac 1 -ar 8000 - \
| docker compose -f infra/compose.yml run --rm -T cw \
    python -m cw.cli stream-stdin \
      --sample-rate 8000 \
      --sample-format s16le \
      --capture-raw samples/live/capture.s16le
```

---

## 5. Mentett nyers felvétel visszajátszása

A `--capture-raw` által mentett `.s16le` fájl nyers PCM, ezért a visszajátszáskor meg kell adni ugyanazokat a paramétereket, amelyekkel készült.

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-raw-file samples/live/capture.s16le \
    --sample-rate 8000 \
    --sample-format s16le
```

JSONL kimenettel:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-raw-file samples/live/capture.s16le \
    --sample-rate 8000 \
    --sample-format s16le \
    --json-output \
  > channels.jsonl
```

---

## 6. WAV/audio fájl dekódolása

A fájlnak a Docker által becsatolt `app/` könyvtáron belül kell elérhetőnek lennie.

Például:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-wav samples/example.wav
```

---

## 7. Debug

A normál dashboard megtartása mellett külön debug JSONL írható:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli stream-raw-file samples/live/capture.s16le \
    --sample-rate 8000 \
    --sample-format s16le \
    --debug-json-output samples/live/capture-debug.jsonl
```

A debug fájl emberi olvasásra:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli view-debug-output samples/live/capture-debug.jsonl
```

Ha a `--debug-json-output` fájlnév nélkül vagy `-` értékkel szerepel, a debug JSONL a stderr-re kerül.

---

## 8. JSONL kimenet megjelenítése

Korábban mentett normál JSONL:

```bash
docker compose -f infra/compose.yml run --rm -T cw \
  python -m cw.cli view-output channels.jsonl
```

Pipe-ból:

```bash
cat channels.jsonl \
| docker compose -f infra/compose.yml run --rm -T cw \
    python -m cw.cli view-output -
```

---

# A kód felépítése

## 9. Feldolgozási lánc

A runtime egyirányú, réteges feldolgozási lánc:

```text
audio source
    |
    v
io
    |
    v
receiving
    |
    v
signal
    |
    v
decoder
    |
    v
selection
    |
    v
app
    |
    +--> JSONL
    |
    v
ui
```

Röviden:

```text
io          hangforrás -> AudioBlock
receiving   AudioBlock -> követett CW csatornák
signal      csatorna -> MARK / SPACE / UNKNOWN futamok
decoder     futamok -> Morse tokenek és dekódolási jelöltek
selection   jelöltek -> csatornánként aktuális nyertes
app         rétegek összekötése, stabil/tentatív kezelés, output
ui          emberi dashboard és debug nézet
```

A réteghatárok szándékosan szigorúak: egy alsóbb réteg nem ismeri a fölötte lévő réteg fogalmait.

---

## 10. Könyvtárstruktúra

```text
cw-decoder/
|
|-- README.md
|-- pyproject.toml
|
|-- infra/
|   |-- Dockerfile
|   |-- compose.yml
|   `-- requirements.txt
|
`-- app/
    |-- src/
    |   `-- cw/
    |       |-- cli.py
    |       |
    |       |-- io/
    |       |-- receiving/
    |       |-- signal/
    |       |-- decoder/
    |       |-- selection/
    |       |-- app/
    |       |-- ui/
    |       `-- tools/
    |
    |-- tests/
    `-- samples/
        `-- live/       # helyi capture/debug munkaterület, git által ignorálva
```

### `io/`

Hangforrás-adapterek és az egységes audio DTO-k.

Feladata például:

```text
stdin raw PCM
WAV/audio file
    ->
AudioBlock stream
```

Nem keres CW vivőt és nem dekódol Morse-jelet.

### `receiving/`

A vevőoldali állapot kezelése:

- audio history;
- spektrális vivőjelöltek;
- csatorna létrehozás;
- stabil `channel_id`;
- frekvenciakövetés;
- eltűnt csatorna újrafelismerése;
- csatorna-életciklus.

A frekvencia változhat, a `channel_id` a követett csatorna identitása.

Ez a réteg még **nem kezel szöveget vagy Morse-karaktereket**.

### `signal/`

A követett csatorna hangjából digitális jelállapotokat készít:

```text
MARK
SPACE
UNKNOWN
```

A jelenlegi normál runtime útvonal az `energy_distribution` aktivitásmodellt használja több posterior elfogadási valószínűséggel.

Egy csatornából ezért több `SignalTrack` készülhet különböző paraméterezéssel.

A signal réteg nem dekódol Morse-karaktereket.

### `decoder/`

A `SignalTrack` futamaiból Morse tokeneket és dekódolási jelölteket készít.

A decoder nem ismeri:

- a csatornaazonosítót;
- a vivőfrekvenciát;
- a hangkártyát;
- a receiving állapotát;
- a selection döntéseit.

Az `UNKNOWN` futam nem automatikusan SPACE és nem is automatikusan MARK: a dekóder helyileg vizsgálja a lehetséges értelmezéseket.

A nem feloldható Morse token megjelenítése:

```text
□
```

A `?` ezzel szemben érvényes Morse-karakter.

### `selection/`

Az aktuális dekódolási jelöltekből csatornánként kiválasztja az aktuális eredményt.

A selection szándékosan tartalomfüggetlen. Nem használ:

- szótárat;
- nyelvi modellt;
- karaktergyakoriságot;
- hívójel-adatbázist;
- ABC szerinti preferenciát.

A jelenlegi kiválasztás fő információi:

- hány signal-paraméterváltozat támogatja ugyanazt az eredményt;
- hány feloldatlan token maradt;
- a dekóder alternatíváin belüli rang;
- szomszédos signal-paraméterek stabilitása.

A selection stateless: nem ő tárolja a korábbi szöveget.

### `app/`

A composition root.

Összeköti:

```text
io
receiving
signal
decoder
selection
ui
```

Itt történik többek között:

- a teljes pipeline futtatása;
- a stabil és még módosuló tokenrész kezelése;
- a biztonságosan eldobható régi audio meghatározása;
- a publikus JSONL előállítása;
- az opcionális debug side-channel előállítása.

### `ui/`

Emberi megjelenítés:

- élő channel dashboard;
- korábban mentett channel JSONL nézet;
- debug JSONL olvasható megjelenítése.

A UI nem végez dekódolási vagy selection döntést.

### `tools/`

Fejlesztői és diagnosztikai segédeszközök.

A normál runtime rétegek nem függhetnek a `tools` csomagtól.

---

## 11. Stabil és tentatív szövegrész

Élő vételnél az aktuális dekódolás vége még változhat, ahogy újabb hang érkezik.

Ezért az app két logikai részre bontja az eredményt:

```text
stable / committed prefix
tentative tail
```

A stabil rész már biztonságosan megtartható.

A tentatív rész még újraértelmezhető későbbi audio alapján.

A dashboard a módosuló véget szögletes zárójelben jelenítheti meg:

```text
CQ CQ [DE HA]
```

A régi, már stabil audio eldobása csak biztonságos Morse-token határon történhet. A receiving réteg maga nem dönt szövegről; csak az app által megadott trim pontot alkalmazza.

A `max_history_s` végső memória-/CPU-védelmi korlát, nem szöveg- vagy GUI-retenciós beállítás.

---

## 12. Publikus output

A publikus output csatorna-snapshotokból áll.

A JSONL token-alapú. Egy token például:

```json
{
  "kind": "char",
  "value": "C",
  "start_s": 1.2,
  "end_s": 1.5,
  "stable": true
}
```

Fő token típusok:

```text
char
unknown
word_gap
session_gap
```

A publikus channel output tartalmazza többek között:

```text
channel_id
carrier_hz
state
tokens
layers
```

A `layers` rövid diagnosztikai állapotot ad a feldolgozási rétegekről.

A részletes signal/decoder/selection belső adatok a külön debug outputba kerülnek.

---

## 13. Tesztek

```bash
docker compose -f infra/compose.yml run --rm -T cw pytest -q
```

A `pyproject.toml` szerint a tesztek helye:

```text
app/tests
```

és a Python forrásgyökér:

```text
app/src
```

---

## 14. Natív Python futtatás Docker nélkül

Python 3.12 vagy újabb szükséges.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r infra/requirements.txt
export PYTHONPATH="$PWD/app/src"

python -m cw.cli --help
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r infra\requirements.txt
$env:PYTHONPATH="$PWD\app\src"

python -m cw.cli --help
```

Docker használatakor erre nincs szükség.

---

## 15. Fejlesztési alapelvek

A runtime rétegek felelőssége maradjon elkülönítve:

```text
receiving  != signal
signal     != decoder
decoder    != selection
selection  != transcript/history
ui         != döntési logika
```

Új minőségi mérőszám oda kerüljön, ahol az információ ténylegesen keletkezik.

A dekódolási döntések ne függjenek a várható szövegtartalomtól. A cél egy objektív CW jelfeldolgozó és dekódoló lánc, nem nyelvi utókorrekció.
