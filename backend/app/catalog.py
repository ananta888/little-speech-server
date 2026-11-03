"""Modellkatalog: offizieller Vosk-Katalog (Snapshot), Whisper-Liste, recasepunc."""

import json
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .schemas import ModelEntry

VENDOR_CATALOG = Path(__file__).parent / "vendor" / "vosk-catalog.json"

WHISPER_MODELS = [
    {"id": "tiny", "name": "Whisper tiny", "hf": "Systran/faster-whisper-tiny",
     "size": "75 MB", "bytes": 75_000_000},
    {"id": "base", "name": "Whisper base", "hf": "Systran/faster-whisper-base",
     "size": "145 MB", "bytes": 145_000_000},
    {"id": "small", "name": "Whisper small", "hf": "Systran/faster-whisper-small",
     "size": "484 MB", "bytes": 484_000_000},
    {"id": "medium", "name": "Whisper medium", "hf": "Systran/faster-whisper-medium",
     "size": "1,5 GB", "bytes": 1_500_000_000},
    {"id": "large-v3", "name": "Whisper large-v3", "hf": "Systran/faster-whisper-large-v3",
     "size": "3,1 GB", "bytes": 3_100_000_000},
]

RECASEPUNC_ID = "vosk-recasepunc-de-0.21"


@lru_cache
def vosk_catalog() -> dict:
    return json.loads(VENDOR_CATALOG.read_text())


def vosk_dir(settings: Settings, model_id: str) -> Path:
    return settings.models_dir / model_id


def vosk_available(settings: Settings, model_id: str) -> bool:
    directory = vosk_dir(settings, model_id)
    return directory.is_dir() and any(directory.iterdir())


def whisper_dir(settings: Settings, model_id: str) -> Path:
    return settings.models_dir / "whisper" / model_id


def whisper_available(settings: Settings, model_id: str) -> bool:
    directory = whisper_dir(settings, model_id)
    return (directory / "model.bin").is_file()


def recasepunc_dir(settings: Settings) -> Path:
    return settings.models_dir / RECASEPUNC_ID


def recasepunc_available(settings: Settings) -> bool:
    return (recasepunc_dir(settings) / "checkpoint").is_file()


def build_vosk_entries(settings: Settings, statuses: dict) -> list[ModelEntry]:
    entries = []
    for model in vosk_catalog()["models"]:
        model_id = model["id"]
        status = statuses.get(model_id, {})
        size = model.get("size")
        entries.append(ModelEntry(
            id=model_id,
            engine="vosk",
            name=f"{model['language']} · {model_id}",
            language=model["language"],
            size=size,
            bytes=int(model["bytes"]) if model.get("bytes") else None,
            license=model.get("license"),
            kind=model.get("kind", "speech"),
            available=vosk_available(settings, model_id),
            downloading=status.get("status") == "downloading",
            downloaded=status.get("downloaded"),
            download_total=status.get("total"),
            percent=status.get("percent"),
            error=status.get("error"),
            url=model.get("url"),
            note=("Nur als Zusatzmodell gelistet – kein Spracherkennungsmodell."
                  if model.get("kind") != "speech" else None),
        ))
    return entries


def build_whisper_entries(settings: Settings, statuses: dict) -> list[ModelEntry]:
    entries = []
    for model in WHISPER_MODELS:
        status = statuses.get(f"whisper:{model['id']}", {})
        entries.append(ModelEntry(
            id=model["id"],
            engine="whisper",
            name=model["name"],
            language="Mehrsprachig",
            size=model["size"],
            bytes=model["bytes"],
            license="MIT",
            available=whisper_available(settings, model["id"]),
            downloading=status.get("status") == "downloading",
            downloaded=status.get("downloaded"),
            download_total=status.get("total"),
            percent=status.get("percent"),
            error=status.get("error"),
            note="Wird beim ersten Laden oder über „herunterladen“ geholt (Hugging Face).",
        ))
    return entries


def build_recasepunc_entry(settings: Settings, statuses: dict) -> ModelEntry:
    model = next((m for m in vosk_catalog()["models"] if m["id"] == RECASEPUNC_ID), None)
    status = statuses.get(RECASEPUNC_ID, {})
    return ModelEntry(
        id=RECASEPUNC_ID,
        engine="recasepunc",
        name=f"Zeichensetzung & Großschreibung · {RECASEPUNC_ID}",
        language="Deutsch",
        size=(model or {}).get("size", "1,1G"),
        bytes=int((model or {}).get("bytes") or 1_100_000_000),
        license=(model or {}).get("license", "Apache 2.0"),
        kind="punctuation",
        available=recasepunc_available(settings),
        downloading=status.get("status") == "downloading",
        downloaded=status.get("downloaded"),
        download_total=status.get("total"),
        percent=status.get("percent"),
        error=status.get("error"),
        url=(model or {}).get("url"),
        note="Post-Processing-Modell (BERT); liefert Text mit Satzzeichen und Großschreibung.",
    )
