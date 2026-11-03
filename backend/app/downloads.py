"""Hintergrund-Downloads für Vosk-Archive und Hugging-Face-Modelle."""

import logging
import shutil
import tarfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

log = logging.getLogger("lss.downloads")


@dataclass
class Task:
    id: str
    status: str = "downloading"  # downloading | done | error | stopped
    downloaded: int = 0
    total: int | None = None
    error: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)


class DownloadManager:
    """Ein Task pro Modell-ID; JSON-Status für die Models-Liste."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def start_vosk(self, model_id: str, url: str) -> Task:
        return self._start(model_id, lambda task: self._run_vosk(task, model_id, url))

    def start_whisper(self, model_id: str, hf_id: str) -> Task:
        return self._start(f"whisper:{model_id}", lambda task: self._run_whisper(task, model_id, hf_id))

    def _start(self, key: str, runner) -> Task:
        with self._lock:
            existing = self._tasks.get(key)
            if existing and existing.status == "downloading":
                return existing
            task = Task(id=key)
            self._tasks[key] = task
        thread = threading.Thread(target=self._guarded, args=(runner, task), daemon=True, name=f"dl-{key}")
        thread.start()
        return task

    def _guarded(self, runner, task: Task) -> None:
        try:
            runner(task)
            if task.stop.is_set():
                task.status = "stopped"
            else:
                task.status = "done"
        except Exception as error:  # noqa: BLE001 – Status in der API sichtbar machen
            log.exception("download failed: %s", task.id)
            task.status = "error"
            task.error = str(error)

    def status(self, key: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(key)
            if not task:
                return None
            percent = None
            if task.total:
                percent = round(min(100.0, task.downloaded / task.total * 100), 1)
            return {"status": task.status, "downloaded": task.downloaded, "total": task.total,
                    "percent": percent, "error": task.error}

    def statuses(self) -> dict[str, dict]:
        with self._lock:
            keys = list(self._tasks)
        return {key: self.status(key) for key in keys if self.status(key)}

    def stop(self, key: str) -> bool:
        with self._lock:
            task = self._tasks.get(key)
        if not task or task.status != "downloading":
            return False
        task.stop.set()
        return True

    # ------------------------------------------------------------------ worker

    def _run_vosk(self, task: Task, model_id: str, url: str) -> None:
        target = self.settings.models_dir / f"{model_id}.zip"
        self._fetch(task, url, target)
        if task.stop.is_set():
            return
        with zipfile.ZipFile(target) as archive:
            archive.extractall(self.settings.models_dir)
        target.unlink(missing_ok=True)

    def _run_whisper(self, task: Task, model_id: str, hf_id: str) -> None:
        from huggingface_hub import snapshot_download

        local = self.settings.models_dir / "whisper" / model_id
        local.mkdir(parents=True, exist_ok=True)
        snapshot_download(hf_id, local_dir=str(local))
        task.total = None
        task.downloaded = 0

    def _fetch(self, task: Task, url: str, target: Path) -> None:
        part = target.with_suffix(target.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.headers.get("Content-Length")
            resolved_total = (offset + int(total)) if total else None
            task.total = resolved_total
            resumed = offset > 0 and response.status == 206
            mode = "ab" if resumed else "wb"
            if not resumed:
                offset = 0
            with open(part, mode) as handle:
                while True:
                    if task.stop.is_set():
                        task.downloaded = offset
                        return
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    offset += len(chunk)
                    task.downloaded = offset
        part.replace(target)


def extract_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    else:
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination)
    shutil.rmtree(archive_path.parent / "__MACOSX", ignore_errors=True)
