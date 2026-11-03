"""Antwort- und Anfragemodelle der API."""

from typing import Literal

from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    start: float
    end: float
    conf: float | None = None


class Segment(BaseModel):
    text: str
    start: float = 0.0
    end: float = 0.0


class TranscribeResponse(BaseModel):
    engine: str
    model: str
    text: str
    language: str | None = None
    duration: float | None = None
    words: list[Word] | None = None
    segments: list[Segment] = Field(default_factory=list)
    took_ms: int = 0


class PostprocessRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)


class PostprocessResponse(BaseModel):
    text: str
    took_ms: int = 0


class ModelEntry(BaseModel):
    id: str
    engine: Literal["vosk", "whisper", "recasepunc"]
    name: str
    language: str | None = None
    size: str | None = None
    bytes: int | None = None
    license: str | None = None
    kind: str = "speech"
    available: bool = False
    downloading: bool = False
    downloaded: int | None = None
    download_total: int | None = None
    percent: float | None = None
    error: str | None = None
    note: str | None = None
    url: str | None = None


class ModelsResponse(BaseModel):
    vosk: list[ModelEntry]
    whisper: list[ModelEntry]
    recasepunc: ModelEntry
    device: str
    models_dir: str


class DownloadStatus(BaseModel):
    id: str
    status: Literal["downloading", "done", "error", "stopped"]
    downloaded: int = 0
    total: int | None = None
    percent: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    ok: bool = True
    device: str
    device_requested: str = "auto"
    cuda: bool = False
    vosk_loaded: str | None = None
    whisper_loaded: str | None = None
    recasepunc_loaded: bool = False
    ffmpeg: bool = False


class DeviceRequest(BaseModel):
    device: Literal["auto", "cpu", "cuda"]


class DeviceResponse(BaseModel):
    requested: str
    resolved: str
    cuda_available: bool
    unloaded: list[str] = Field(default_factory=list)


class DeleteModelRequest(BaseModel):
    engine: Literal["vosk", "whisper", "recasepunc"]
    id: str = Field(min_length=1, max_length=200)


class DeleteModelResponse(BaseModel):
    deleted: str
    unloaded: bool
