# Cleanup notes

A korábbi iterációkban a kód két irányban zajosodott el:

1. sok mellékparancs és laborsegéd maradt a runtime CLI körül;
2. a live kimenet több helyen próbált transcript-állapotot értelmezni.

Ebben a csomagban a cél nem újabb dekódoló heurisztika volt, hanem a fő út megtisztítása:

```text
audio source -> receiving -> signal -> decoder -> selection -> output JSONL/dashboard
```

## Rétegek

- `io/`: bemenet, PCM/WAV/raw blokkok.
- `receiving/`: vivő-megfigyelés és stabil csatornakövetés; nem dekódol, nem kezel transcriptet, nem ad publikus eventet.
- `signal/`: csatorna-audióból morze-jel futamok; egy csatornából több signal track készülhet, több analizátor/profil alapján.
- `decoder/`: morze-jel futamokból dekódolt szövegválaszok.
- `selection/`: a teljes aktuális dekódolt halmazból csatornánként egy szöveget választ; belső pontozása nem szivárog ki.
- `app/`: CLI, bekötés, JSONL output.
- `ui/`: `ChannelOutput` dashboard, csak megjelenít.
- `tools/`: offline/dev diagnosztika.

Nincs külön `events` réteg. A publikus live kimenet csatorna-snapshot, a belső diagnosztikai részletek nem szivárognak ki az alap JSON-be.

## Szándékos nem-cél

Nem próbáltam most minden vételi hibát decoder-oldali heurisztikával javítani. A signal réteg viszont már két külön családot tud futtatni ugyanarra a csatornára: küszöbös és eloszlásalapú aktivitásbontást.


## Decoder cleanup

A futó dekóder útvonal most már közvetlenül ezt csinálja:

```text
SignalTrack(MARK/SPACE/UNKNOWN runs) -> RunDecoder -> DecodeResult(DecodedText[])
```

Kikerült a runtime dekóder útból a régi vivő/session/report modell:
`CarrierDecodeResult`, `DecodedSession`, `DecodeReport`, `carrier_hz=0.0` adapterek.
Ezek a régi offline diagnosztikai kódokkal együtt a `tools/legacy_decoder/` alá kerültek.

Az `UNKNOWN` nem sima gap többé. A dekóder belső időzítési modellje megőrzi, és
minden UNKNOWN futamot lokálisan ágaztat MARK/SPACE irányba. Nincs globális
"minden UNKNOWN jel" / "minden UNKNOWN szünet" kapcsoló.

A publikus dekóder-válasz nem tartalmaz általános statisztikákat. Egyetlen
dekóder-minőségi adat maradt: `unresolved_tokens`, vagyis hány morze-tokenből
nem lett érvényes karakter. Ezek helyén a szövegben `□` látszik; a `?` továbbra
is rendes, érvényes morze-karakter.

## Selection cleanup

A selection réteg már nem egyenként előkészített `DecodeChoice` listán dolgozik, hanem teljes batch bemenetet kap: csatornák, azok signal trackjei és azok dekóder-válaszai. A selection végzi a csatorna szerinti csoportosítást, azonos szövegek támogatottságának számítását, modellcsalád-diverzitás és paraméter-szomszédsági stabilitás figyelembevételét, majd kis különbségnél időbeli hiszterézist alkalmaz.

A régi transcript/stable/unstable logika kikerült a selectionből. A kiválasztási pontszám belső részlet, nem publikus DTO és nem JSON-mező.

## Output/UI cleanup

Az utolsó réteg is snapshot-alapú lett. Az alap JSONL pontosan négy publikus mezőt tartalmaz:

```json
{"channel_id":4,"carrier_hz":702.0,"state":"active","text":"CQ"}
```

A dashboard ezt a `ChannelOutput` snapshotot jeleníti meg, és nem értelmez transcriptet, sessiont, stable/unstable részeket, dekóder-válaszokat vagy selection score-t. A `view-output` ugyanazt a JSONL formátumot olvassa vissza, amit a `--json-output` ír.
