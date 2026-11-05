"""REST-API: Gesundheit, Modellliste, Downloads, Transkription, recasepunc."""

import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from . import runtime
from .catalog import (RECASEPUNC_ID, WHISPER_MODELS, build_llm_entries,
                      build_recasepunc_entry, build_vosk_entries, build_whisper_entries,
                      vosk_catalog)
from .config import get_settings
from .downloads import DownloadManager
from .engines.vosk_engine import VoskEngine
from .engines.whisper_engine import WhisperEngine
from .postprocess.recasepunc_service import RecasepuncService
from .schemas import (DeleteModelRequest, DeleteModelResponse, DeviceRequest,
                      DeviceResponse, DownloadStatus, HealthResponse, ModelsResponse,
                      PostprocessRequest, PostprocessResponse, TextTaskRequest,
                      TranscribeResponse)
from .text_service import TEXT_MODELS, get_text_service, run_streaming

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

    return HealthResponse(
        device=runtime.resolve_device(),
        device_requested=runtime.requested_device(),
        cuda=runtime.resolve_device() == "cuda",
        vosk_loaded=get_vosk_engine().loaded,
        whisper_loaded=get_whisper_engine().loaded,
        recasepunc_loaded=get_recasepunc().loaded,
        text_loaded=get_text_service().loaded,
        ffmpeg=bool(shutil.which("ffmpeg")),
    )


@router.post("/device", response_model=DeviceResponse)
async def set_device(body: DeviceRequest) -> DeviceResponse:
    """Rechengerät zur Laufzeit wechseln; geladene Modelle werden entladen."""
    before = runtime.resolve_device()
    runtime.set_device(body.device)
    after = runtime.resolve_device()
    unloaded: list[str] = []
    if before != after:
        if get_whisper_engine().unload():
            unloaded.append("whisper")
        if get_recasepunc().unload():
            unloaded.append("recasepunc")
        if get_text_service().unload():
            unloaded.append("text")
    return DeviceResponse(
        requested=runtime.requested_device(),
        resolved=after,
        cuda_available=runtime.resolve_device() == "cuda",
        unloaded=unloaded,
    )


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    settings = get_settings()
    statuses = get_downloads().statuses()
    return ModelsResponse(
        vosk=build_vosk_entries(settings, statuses),
        whisper=build_whisper_entries(settings, statuses),
        recasepunc=build_recasepunc_entry(settings, statuses),
        llm=build_llm_entries(settings, statuses),
        device=runtime.resolve_device(),
        models_dir=str(settings.models_dir),
    )


@router.post("/downloads/llm/{model_id}", response_model=DownloadStatus)
async def download_llm(model_id: str) -> DownloadStatus:
    info = TEXT_MODELS.get(model_id)
    if not info:
        raise HTTPException(404, f"Unbekanntes Textmodell: {model_id}")
    task = get_downloads().start_llm(model_id, info["hf"])
    return _status_response(task.id)


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


@router.post("/models/delete", response_model=DeleteModelResponse)
async def delete_model(body: DeleteModelRequest) -> DeleteModelResponse:
    """Modelldateien vom Server löschen (nach Entladen, falls geladen)."""
    import shutil

    settings = get_settings()
    base = settings.models_dir.resolve()
    known = body.engine == "recasepunc" and body.id == RECASEPUNC_ID
    if body.engine == "vosk":
        known = bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", body.id))
        target = settings.models_dir / body.id
        unloaded = get_vosk_engine().loaded == body.id and get_vosk_engine().unload()
    elif body.engine == "whisper":
        known = any(entry["id"] == body.id for entry in WHISPER_MODELS)
        target = settings.models_dir / "whisper" / body.id
        unloaded = get_whisper_engine().loaded == body.id and get_whisper_engine().unload()
    elif body.engine == "llm":
        known = body.id in TEXT_MODELS
        target = settings.models_dir / "llm" / body.id
        unloaded = get_text_service().loaded == body.id and get_text_service().unload()
    else:
        target = settings.models_dir / RECASEPUNC_ID
        unloaded = get_recasepunc().loaded and get_recasepunc().unload()
    if not known:
        raise HTTPException(404, f"Unbekanntes Modell: {body.id}")
    resolved = target.resolve()
    if not resolved.is_relative_to(base) or resolved == base or not resolved.exists():
        raise HTTPException(404, f"Modellverzeichnis nicht vorhanden: {resolved}")
    shutil.rmtree(resolved)
    return DeleteModelResponse(deleted=body.id, unloaded=bool(unloaded))


@router.post("/models/import")
async def import_model(file: UploadFile = File(...)) -> dict:
    """Eigenes Vosk-Archiv (ZIP oder TAR.GZ) ins Modellverzeichnis entpacken.

    Pfadsicher: Mitglieder außerhalb des Zielverzeichnisses und Symlinks
    führen zu einem Fehler, __MACOSX wird ignoriert.
    """
    import tarfile
    import zipfile

    settings = get_settings()
    base = settings.models_dir.resolve()
    filename = file.filename or "modell.zip"
    temp_path = base / f".import-{abs(hash(filename))}.part"
    model_id: str | None = None
    try:
        with open(temp_path, "wb") as handle:
            while chunk := await file.read(1 << 20):
                handle.write(chunk)
        with open(temp_path, "rb") as probe:
            magic = probe.read(2)
        if magic == b"PK":
            with zipfile.ZipFile(temp_path) as archive:
                names = [name for name in archive.namelist()
                         if not name.endswith("/") and not name.startswith("__MACOSX/")]
                model_id = _pick_import_target(names, filename)
                if _is_nested_top_dir(names, model_id):
                    _safe_extract_zip(archive, base, names)
                else:
                    target = base / model_id
                    target.mkdir(parents=True, exist_ok=True)
                    _safe_extract_zip(archive, target, names)
        elif magic == b"\x1f\x8b":
            with tarfile.open(temp_path, "r:gz") as archive:
                members = [member for member in archive.getmembers()
                           if member.isfile() and not member.name.startswith("__MACOSX/")]
                names = [member.name for member in members]
                model_id = _pick_import_target(names, filename)
                if _is_nested_top_dir(names, model_id):
                    _safe_extract_tar(archive, base, members)
                else:
                    target = base / model_id
                    target.mkdir(parents=True, exist_ok=True)
                    _safe_extract_tar(archive, target, members)
        else:
            raise HTTPException(400, "Datei ist weder ZIP noch TAR.GZ/TGZ.")
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        raise HTTPException(422, f"Archiv konnte nicht entpackt werden: {error}") from error
    finally:
        temp_path.unlink(missing_ok=True)
    if not model_id:
        raise HTTPException(422, "Archiv enthält kein Modellverzeichnis.")
    resolved = (base / model_id).resolve()
    if not resolved.is_relative_to(base) or not resolved.is_dir():
        raise HTTPException(422, "Archiv enthält kein brauchbares Modellverzeichnis.")
    if not any(resolved.iterdir()):
        resolved.rmdir()
        raise HTTPException(422, "Archiv ist leer.")
    return {"id": model_id}


_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


def _pick_import_target(names: list[str], filename: str) -> str:
    """Ein Top-Verzeichnis im Archiv wird übernommen, sonst der Dateiname."""
    top = {name.split("/")[0] for name in names if name.split("/")[0]}
    if len(top) == 1 and all("/" in name for name in names):
        candidate = next(iter(top))
        if _ID_PATTERN.fullmatch(candidate) and candidate not in {".", ".."}:
            return candidate
    return _import_id_from_name(filename)


def _is_nested_top_dir(names: list[str], model_id: str) -> bool:
    return bool(names) and all(name.startswith(model_id + "/") for name in names)


def _safe_extract_zip(archive, destination: Path, names: list[str]) -> None:
    for name in names:
        if not (destination / name).resolve().is_relative_to(destination):
            raise HTTPException(422, f"Archiv enthält unzulässigen Pfad: {name}")
    archive.extractall(destination, members=names)


def _safe_extract_tar(archive, destination: Path, members) -> None:
    for member in members:
        if member.issym() or member.islnk():
            raise HTTPException(422, f"Archiv enthält Symlinks: {member.name}")
        if not (destination / member.name).resolve().is_relative_to(destination):
            raise HTTPException(422, f"Archiv enthält unzulässigen Pfad: {member.name}")
    archive.extractall(destination, members=members)


def _import_id_from_name(filename: str) -> str:
    name = filename
    for ending in (".tar.gz", ".tgz", ".tar", ".zip", ".gz"):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip(".-") or "importiertes-modell"
    return name[:200]


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
        "device": runtime.resolve_device(),
    }


@router.post("/text")
async def text_task(body: TextTaskRequest) -> StreamingResponse:
    """Textaufgabe als NDJSON-Stream (load, preview, section, done|cancelled, error)."""
    if body.model not in TEXT_MODELS:
        raise HTTPException(404, f"Unbekanntes Textmodell: {body.model}")
    service = get_text_service()
    if service.busy:
        raise HTTPException(409, "Es läuft bereits eine Textaufgabe – erst abwarten oder abbrechen.")
    generator = run_streaming(body.text, body.task, body.model, service)
    return StreamingResponse(generator, media_type="application/x-ndjson")


@router.post("/text/cancel")
async def text_cancel() -> dict:
    get_text_service().cancel()
    return {"ok": True}


@router.post("/text/unload")
async def text_unload() -> dict:
    return {"unloaded": get_text_service().unload()}


@router.get("/text/status")
async def text_status() -> dict:
    service = get_text_service()
    return {
        "loaded": service.loaded,
        "loading": service.loading,
        "busy": service.busy,
        "device": runtime.resolve_device(),
        "models": [
            {"id": model_id, "available": (get_settings().models_dir / "llm" / model_id / "config.json").is_file(),
             **{key: TEXT_MODELS[model_id][key] for key in ("name", "size", "min_device")}}
            for model_id in TEXT_MODELS
        ],
    }


def _status_response(key: str) -> DownloadStatus:
    status = get_downloads().status(key) or {"status": "downloading", "downloaded": 0}
    return DownloadStatus(id=key, **status)
