"""Whisper-Engine (faster-whisper) für Dateien und segmentweises Live-Streaming."""

import logging
import threading

from .. import runtime
from ..catalog import WHISPER_MODELS, whisper_available, whisper_dir
from ..config import Settings

log = logging.getLogger("lss.whisper")

CHUNK_SAMPLES = 64_000  # 4 s @ 16 kHz für den Live-Stream


def _free_accelerator_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class ModelNotAvailable(RuntimeError):
    pass


def _hf_id(model_id: str) -> str:
    for entry in WHISPER_MODELS:
        if entry["id"] == model_id:
            return entry["hf"]
    raise ModelNotAvailable(f"Unbekanntes Whisper-Modell: {model_id}")


class WhisperEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model_id: str | None = None
        self._model = None

    @property
    def loaded(self) -> str | None:
        return self._model_id

    def unload(self) -> bool:
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            self._model_id = None
            _free_accelerator_memory()
            return True

    def ensure_loaded(self, model_id: str):
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return self._model
            from faster_whisper import WhisperModel

            device = runtime.resolve_device()
            compute_type = "int8_float16" if device == "cuda" else "int8"
            local = whisper_dir(self.settings, model_id)
            source = str(local) if whisper_available(self.settings, model_id) else _hf_id(model_id)
            log.info("loading whisper model %s (source=%s, device=%s)", model_id, source, device)
            self._model = WhisperModel(source, device=device, compute_type=compute_type)
            self._model_id = model_id
            return self._model

    def transcribe_file(self, model_id: str, path: str, language: str | None = None) -> dict:
        model = self.ensure_loaded(model_id)
        segments_iter, info = model.transcribe(
            path,
            language=language or (self.settings.whisper_language or None),
            vad_filter=True,
            beam_size=5,
        )
        segments = []
        pieces = []
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            pieces.append(text)
            segments.append({"text": text, "start": round(segment.start, 3), "end": round(segment.end, 3)})
        return {"text": " ".join(pieces).strip(), "language": info.language,
                "duration": round(info.duration, 3) if info.duration else None,
                "words": None, "segments": segments}


class WhisperStreamSession:
    """Sammelt PCM16 und liefert alle 4 s ein Ergebnis (ähnlich der Browser-App)."""

    def __init__(self, engine: WhisperEngine, model_id: str):
        self.engine = engine
        self.model_id = model_id
        self._buffer = bytearray()
        self._start_sample = 0
        self._next_sample = 0
        self._lock = threading.Lock()

    def accept(self, pcm: bytes) -> list[dict]:
        import numpy as np

        with self._lock:
            self._buffer.extend(pcm)
            if len(self._buffer) < CHUNK_SAMPLES * 2:
                return []
            chunk = bytes(self._buffer[:CHUNK_SAMPLES * 2])
            del self._buffer[:CHUNK_SAMPLES * 2]
            start = self._next_sample / 16000
            self._next_sample += CHUNK_SAMPLES
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        model = self.engine.ensure_loaded(self.model_id)
        segments_iter, _info = model.transcribe(samples, language=self.settings_whisper_language(),
                                                 vad_filter=False, beam_size=1)
        events = []
        text_parts = []
        end = start + CHUNK_SAMPLES / 16000
        for segment in segments_iter:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
                end = start + min(segment.end, CHUNK_SAMPLES / 16000)
        if text_parts:
            events.append({"type": "final", "text": " ".join(text_parts),
                           "start": round(start, 3), "end": round(end, 3)})
        return events

    def settings_whisper_language(self):
        return self.engine.settings.whisper_language or None

    def finish(self) -> dict:
        import numpy as np

        with self._lock:
            rest = bytes(self._buffer)
            start = self._next_sample / 16000
            self._buffer.clear()
            self._next_sample += len(rest) // 2
        if len(rest) < 3200:  # unter 0,1 s: stumm
            return {"type": "final", "text": "", "start": start, "end": start}
        samples = np.frombuffer(rest, dtype=np.int16).astype(np.float32) / 32768.0
        model = self.engine.ensure_loaded(self.model_id)
        segments_iter, _info = model.transcribe(samples, language=self.settings_whisper_language(),
                                                 vad_filter=False, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments_iter if segment.text.strip())
        return {"type": "final", "text": text, "start": round(start, 3),
                "end": round(start + len(rest) / 2 / 16000, 3)}
