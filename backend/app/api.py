"""REST-API: Gesundheit, Modellliste, Downloads, Transkription, recasepunc."""

import tempfile
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .catalog import (WHISPER_MODELS, build_recasepunc_entry,
                      build_vosk_entries, build_whisper_entries, vosk_catalog)
from .config import get_settings
from .downloads import DownloadManager
from .engines.vosk_engine import VoskEngine
from .engines.whisper_engine import WhisperEngine
from .postprocess.recasepunc_service import RecasepuncService
from .schemas import (DownloadStatus, HealthResponse, ModelsResponse,
                      PostprocessRequest, PostprocessResponse, TranscribeResponse)

router = APIRouter(prefix="/api")


@lru_cache
def get_downloads() -> DownloadManager:
    return DownloadManager(get_settings())


@lru_cache
def get_vosk_engine() -> VoskEngine:
    return VoskEngine(get_settings())


@lru_cache
def get_whisper_engine() -> WhisperEngine:
    return WhisperEngine(get_settings())


@lru_cache
def get_recasepunc() -> RecasepuncService:
    return RecasepuncService(get_settings())


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    import shutil

    settings = get_settings()
    return HealthResponse(
        device=settings.resolve_device(),
        vosk_loaded=get_vosk_engine().loaded,
        whisper_loaded=get_whisper_engine().loaded,
        recasepunc_loaded=get_recasepunc().loaded,
        ffmpeg=bool(shutil.which("ffmpeg")),
    )


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    settings = get_settings()
    statuses = get_downloads().statuses()
    return ModelsResponse(
        vosk=build_vosk_entries(settings, statuses),
        whisper=build_whisper_entries(settings, statuses),
        recasepunc=build_recasepunc_entry(settings, statuses),
        device=settings.resolve_device(),
        models_dir=str(settings.models_dir),
    )


@router.post("/downloads/vosk/{model_id}", response_model=DownloadStatus)
async def download_vosk(model_id: str) -> DownloadStatus:
    model = next((m for m in vosk_catalog()["models"] if m["id"] == model_id), None)
    if not model:
        raise HTTPException(404, f"Unbekanntes Vosk-Modell: {model_id}")
    task = get_downloads().start_vosk(model_id, model["url"])
    return _status_response(task.id)


@router.post("/downloads/whisper/{model_id}", response_model=DownloadStatus)
async def download_whisper(model_id: str) -> DownloadStatus:
    entry = next((m for m in WHISPER_MODELS if m["id"] == model_id), None)
    if not entry:
        raise HTTPException(404, f"Unbekanntes Whisper-Modell: {model_id}")
    task = get_downloads().start_whisper(model_id, entry["hf"])
    return _status_response(task.id)


@router.post("/downloads/stop/{model_id}", response_model=DownloadStatus)
async def stop_download(model_id: str) -> DownloadStatus:
    get_downloads().stop(model_id)
    return _status_response(model_id)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    engine: str = Form(...),
    model: str = Form(...),
    language: str = Form(""),
) -> TranscribeResponse:
    settings = get_settings()
    if engine not in {"vosk", "whisper"}:
        raise HTTPException(400, f"Unbekannte Engine: {engine}")
    payload = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(413, f"Datei ist größer als {settings.max_upload_mb} MB.")
    if not payload:
        raise HTTPException(400, "Die Datei ist leer.")
    suffix = Path(file.filename or "audio").suffix or ".bin"
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        handle.write(payload)
        handle.flush()
        try:
            if engine == "vosk":
                result = await run_in_threadpool(get_vosk_engine().transcribe_file, model, handle.name)
            else:
                result = await run_in_threadpool(
                    get_whisper_engine().transcribe_file, model, handle.name, language or None
                )
        except Exception as error:  # noqa: BLE001 – als HTTP-Fehler ausliefern
            raise HTTPException(422, str(error)) from error
    return TranscribeResponse(
        engine=engine,
        model=model,
        text=result.get("text", ""),
        language=result.get("language"),
        duration=result.get("duration"),
        words=result.get("words"),
        segments=result.get("segments") or [],
        took_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post("/postprocess", response_model=PostprocessResponse)
async def postprocess(body: PostprocessRequest) -> PostprocessResponse:
    try:
        text, took_ms = await run_in_threadpool(get_recasepunc().process, body.text)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(422, str(error)) from error
    return PostprocessResponse(text=text, took_ms=took_ms)


@router.get("/postprocess/status")
async def postprocess_status() -> dict:
    settings = get_settings()
    service = get_recasepunc()
    from .catalog import recasepunc_available

    return {
        "available": recasepunc_available(settings),
        "loaded": service.loaded,
        "loading": service.loading,
        "device": settings.resolve_device(),
    }


def _status_response(key: str) -> DownloadStatus:
    status = get_downloads().status(key) or {"status": "downloading", "downloaded": 0}
    return DownloadStatus(id=key, **status)
