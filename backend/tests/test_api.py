"""Katalog- und API-Tests ohne große Modell-Downloads (Integration opt-in)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.catalog import RECASEPUNC_ID, vosk_catalog
from app.config import get_settings
from app.main import app

FIXTURE_WAV = Path(__file__).parent / "fixtures" / "speech.wav"
SMALL_DE = "vosk-model-small-de-0.15"


def test_catalog_snapshot():
    catalog = vosk_catalog()
    models = catalog["models"]
    assert len(models) >= 70
    speech = [model for model in models if model["kind"] == "speech"]
    assert len(speech) == 76
    assert any(model["id"] == "vosk-model-de-0.21" for model in models)
    recasepunc = next(model for model in models if model["id"] == RECASEPUNC_ID)
    assert recasepunc["kind"] == "punctuation"
    assert recasepunc["url"].startswith("https://alphacephei.com/")


def test_health_and_models():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        models = client.get("/api/models")
        assert models.status_code == 200
        body = models.json()
        assert len(body["vosk"]) >= 70
        assert len(body["whisper"]) == 5
        assert body["recasepunc"]["id"] == RECASEPUNC_ID
        # Zusatzmodelle sind in der Liste, aber nicht als Sprachmodell markiert
        punctuation = [entry for entry in body["vosk"] if entry["kind"] == "punctuation"]
        assert punctuation and all(entry["note"] is not None for entry in punctuation)


def test_unknown_download_model():
    with TestClient(app) as client:
        response = client.post("/api/downloads/vosk/vosk-model-gibtsnicht")
        assert response.status_code == 404


def test_empty_transcribe_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/api/transcribe",
            files={"file": ("empty.wav", b"")},
            data={"engine": "vosk", "model": SMALL_DE},
        )
        assert response.status_code == 400


@pytest.mark.skipif(
    not (get_settings().models_dir / SMALL_DE).is_dir(),
    reason=f"{SMALL_DE} nicht heruntergeladen",
)
def test_transcribe_fixture():
    with TestClient(app) as client:
        response = client.post(
            "/api/transcribe",
            files={"file": ("speech.wav", FIXTURE_WAV.read_bytes())},
            data={"engine": "vosk", "model": SMALL_DE},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["text"].strip()
        assert body["duration"] and body["duration"] > 1
        assert body["words"], "Wortzeiten werden erwartet"
        assert body["segments"]


@pytest.mark.skipif(
    not (get_settings().models_dir / SMALL_DE).is_dir(),
    reason=f"{SMALL_DE} nicht heruntergeladen",
)
def test_stream_silence():
    import struct

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as socket:
            socket.send_text(json.dumps({"type": "config", "engine": "vosk", "modelId": SMALL_DE}))
            assert json.loads(socket.receive_text())["type"] == "ready"
            socket.send_bytes(struct.pack("<16000h", *([0] * 16000)))
            socket.send_text(json.dumps({"type": "stop"}))
            types = []
            while True:
                message = json.loads(socket.receive_text())
                types.append(message["type"])
                if message["type"] == "end":
                    break
            assert "final" in types


@pytest.mark.skipif(
    not (get_settings().models_dir / RECASEPUNC_ID / "checkpoint").is_file(),
    reason="recasepunc-Checkpoint nicht heruntergeladen",
)
def test_recasepunc_matches_reference():
    base = get_settings().models_dir / RECASEPUNC_ID
    source = " ".join((base / "de-test.txt").read_text().splitlines())
    reference = (base / "de-test.txt.orig").read_text()
    from app.api import get_recasepunc

    output, took = get_recasepunc().process(source)
    assert " ".join(output.split()) == " ".join(reference.split())
    assert took > 0


def test_whisper_model_list_ids():
    from app.catalog import WHISPER_MODELS
    from app.engines.whisper_engine import _hf_id

    ids = [entry["id"] for entry in WHISPER_MODELS]
    assert ids == ["tiny", "base", "small", "medium", "large-v3"]
    assert _hf_id("tiny") == "Systran/faster-whisper-tiny"
    with pytest.raises(Exception):
        _hf_id("gibt-es-nicht")


@pytest.mark.skipif(
    not (get_settings().models_dir / "whisper" / "tiny" / "model.bin").is_file(),
    reason="whisper tiny nicht heruntergeladen",
)
def test_transcribe_whisper_fixture():
    with TestClient(app) as client:
        response = client.post(
            "/api/transcribe",
            files={"file": ("speech.wav", FIXTURE_WAV.read_bytes())},
            data={"engine": "whisper", "model": "tiny"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "test" in body["text"].lower()
        assert body["language"] == "en"
        assert body["segments"]


def test_ffmpeg_available_for_uploads():
    import shutil

    assert shutil.which("ffmpeg"), "ffmpeg wird für Datei-Uploads gebraucht"


def test_device_switch_and_validation():
    with TestClient(app) as client:
        response = client.post("/api/device", json={"device": "cpu"})
        assert response.status_code == 200, response.text
        assert response.json()["resolved"] == "cpu"
        assert response.json()["unloaded"] in ([], ["whisper"], ["recasepunc"], ["whisper", "recasepunc"])

        response = client.post("/api/device", json={"device": "auto"})
        assert response.status_code == 200
        assert response.json()["requested"] == "auto"

        # 'gpu' ist kein gültiger Wert – die UI bietet 'cuda' an
        assert client.post("/api/device", json={"device": "gpu"}).status_code == 422

        health = client.get("/api/health").json()
        assert health["device_requested"] == "auto"
        assert health["device"] in ("cpu", "cuda")
        assert isinstance(health["cuda"], bool)
