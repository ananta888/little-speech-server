# little speech · Server

Lokale Sprach-zu-Text-Web-App: **Angular-Frontend + Python-Backend**. Kein
Fremd-Dienst, keine Audio-Uploads – Vosk und Whisper laufen auf dem eigenen
Rechner, die Zeichensetzung zieht das recasepunc-Modell nach.

## Möglichkeiten

- **Vosk** (CPU, sparsam): alle 76 Modelle des offiziellen Katalogs,
  Live-Erkennung per WebSocket (PCM-Strom,16 kHz) und Datei-Upload mit
  Wortzeiten, Konfidenzen und Abschnitten.
- **Whisper** (faster-whisper): tiny bis large-v3, Datei-Upload und Live in
  ~4-Sekunden-Abschnitten; Modelle kommen auf Abruf von Hugging Face.
- **recasepunc** (deutsch, ca. 1,1 G): Großschreibung und Satzzeichen für
  Transkripte – ein Klick im Abschnitt „Transkript“, mit „Wiederherstellen“
  zum Original. Ergebnis ist mit dem offiziellen Referenz-Output des Modells
  identisch (Test `test_recasepunc_matches_reference`).
- Export als **TXT** und **SRT** (Abschnittszeiten), editierbares Transkript,
  Wort-Chips mit Konfidenz (Werte unter 0,70 hervorgehoben).
- Modellverwaltung mit Download-Fortschritt im Browser; alles landet im
  Modellverzeichnis des Servers und bleibt erhalten.

## Voraussetzungen

- Python 3.12+, Node 20+ (nur für den Frontend-Build)
- `ffmpeg` im PATH (Pflicht für Datei-Uploads): `sudo apt install ffmpeg`
- Für GPU-Betrieb: CUDA-fähiges `torch` (Standard-Installation ist CPU;
  siehe „Optionen“ unten)

## Starten

```sh
# Backend-Abhängigkeiten
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt

# Frontend bauen (Output: frontend/dist/frontend/browser, wird vom Backend ausgeliefert)
cd frontend && npm ci && npm run build && cd ..

# Server starten (bindet API, WebSocket und Angular-Build ein)
cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Dann <http://127.0.0.1:8000> öffnen, in01 Engine und Modell wählen,
„Herunterladen“ klicken, fertig.

### Entwicklung (Frontend live)

```sh
# Terminal 1
cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000
# Terminal 2 (Proxy auf :8000, Port 4300, siehe proxy.conf.json)
cd frontend && npm start
```

## Abläufe

- **Datei-Upload** → `POST /api/transcribe` (Multipart). Vosk dekodiert
  serverseitig per ffmpeg auf16 kHz mono, Whisper reicht die Datei direkt.
- **Live-Mikrofon** → Browser erzeugt PCM16@16 kHz und sendet Binärframes an
  `WS /ws/stream`. Erste Nachricht ist
  `{"type":"config","engine":"vosk"|"whisper","modelId":"..."}`, dann
  `partial`/`final`-Antworten, zum Beenden `{"type":"stop"}` bis `end`.
- **Zeichensetzung** → `POST /api/postprocess`. Der Checkpoint wird einmalig
  lazy geladen (CPU-RAM danach ca. 0,5 G, GPU-Cache bei CUDA-Nutzung).

### REST-Übersicht

| Methode | Pfad | Zweck |
| --- | --- | --- |
| GET | `/api/health` | Gerät, ffmpeg, geladene Modelle |
| GET | `/api/models` | Katalog + Download-Status |
| POST | `/api/downloads/vosk/{id}` | Vosk-Archiv laden (entpackt sich selbst) |
| POST | `/api/downloads/whisper/{id}` | Whisper-Modell vorladen |
| POST | `/api/downloads/stop/{id}` | Download abbrechen |
| POST | `/api/transcribe` | Datei transkribieren |
| POST | `/api/postprocess` | recasepunc auf Text anwenden |
| GET | `/api/postprocess/status` | recasepunc vorhanden/geladen |

## Konfiguration (Umgebung, Präfix `LSS_`)

| Variable | Standard | Bedeutung |
| --- | --- | --- |
| `LSS_PORT`/`LSS_HOST` | 8000/127.0.0.1 | nur informativ (uvicorn-Argumente sind führend) |
| `LSS_MODELS_DIR` | `<Repo>/models` | Modellverzeichnis |
| `LSS_DEVICE` | `auto` | `auto`/`cpu`/`cuda` für Whisper und recasepunc |
| `LSS_CORS_ORIGINS` | `http://localhost:4300,...` | für den Dev-Server |
| `LSS_MAX_UPLOAD_MB` | 512 | Größenlimit Upload |

## Tests & Checks

```sh
cd backend
.venv/bin/ruff check app tests          # Stil/Imports (vendorter Code ist ausgenommen)
.venv/bin/python -m pytest tests -q     #10 Tests; große Modelle werden per skipif gemieden
cd ../frontend && npx ng build          # Typ- und Template-Prüfung ist der Build
```

Der Modellkatalog (Snapshot) lässt sich mit
`backend/scripts/fetch_vosk_catalog.py` neu einlesen.

## Speicher & Modelle

- recasepunc-Checkpoint1,3 G (der Adam-Anteil wird nach dem Laden freigegeben),
  Speicherbedarf danach ca. 0,5 G. Auf CUDA automatisch auf der GPU.
- Vosk-Großmodelle (1,9 G /4,4 G) brauchen Zeit zum Laden; die UI meldet das
  über den Status. Pro Engine wird genau ein Modell resident gehalten.
- Downloads liegen unter `models/` (gitignore) und werden wiederverwendet.

## Dritt-Lizenzen (Auszug)

- Vosk-Modelle und recasepunc-Archiv: Apache 2.0, alphacephei.com/vosk/models;
  recasepunc basiert auf benob/recasepunc mit `dbmdz/bert-base-german-uncased`
  (MIT) – der Checkpoint wird unverändert geladen, `vendor/recasepunc_official.py`
  enthält nur kommentierte Lade-Patches.
- faster-whisper (MIT), CTranslate2, onnxruntime (MIT), PyAV (BSD).
- Angular (MIT), RxJS (Apache-2.0).
- `tests/fixtures/speech.wav` stammt aus dem little-stt-Testbestand
  („1, 2, 3, 4, 5, this is a test …“).
