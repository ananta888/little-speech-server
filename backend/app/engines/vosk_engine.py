"""Vosk-Engine: Batch-Transkription (Datei) und Streaming (Live-Mikrofon)."""

import json
import logging
import shutil
import subprocess
import threading

from ..catalog import vosk_available, vosk_dir
from ..config import Settings

log = logging.getLogger("lss.vosk")


class ModelNotAvailable(RuntimeError):
    pass


class VoskSession:
    """Ein KaldiRecognizer für PCM16-LE-Streams mit 16 kHz, mono."""

    def __init__(self, model, sample_rate: int = 16000):
        import vosk

        self.recognizer = vosk.KaldiRecognizer(model, sample_rate)
        self.recognizer.SetWords(True)

    def accept(self, pcm: bytes) -> list[dict]:
        if self.recognizer.AcceptWaveform(pcm):
            return [{"type": "final", **json.loads(self.recognizer.Result())}]
        partial = json.loads(self.recognizer.PartialResult())
        return [{"type": "partial", "text": partial.get("partial", "")}]

    def finish(self) -> dict:
        return {"type": "final", **json.loads(self.recognizer.FinalResult())}


class VoskEngine:
    """Hält genau ein Vosk-Modellresident (Speicher)."""

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
            import gc

            gc.collect()
            return True

    def ensure_loaded(self, model_id: str):
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return self._model
            directory = vosk_dir(self.settings, model_id)
            if not vosk_available(self.settings, model_id):
                raise ModelNotAvailable(f"Vosk-Modell {model_id} ist nicht geladen. "
                                        f"Bitte zuerst im Modellbereich herunterladen (erwartet: {directory}).")
            from ..text_service import get_text_service

            get_text_service().unload()
            import vosk

            vosk.SetLogLevel(-1)
            log.info("loading vosk model %s", model_id)
            model = vosk.Model(str(directory))
            self._model = model
            self._model_id = model_id
            return model

    def create_session(self, model_id: str, sample_rate: int = 16000) -> VoskSession:
        return VoskSession(self.ensure_loaded(model_id), sample_rate)

    def transcribe_file(self, model_id: str, path: str) -> dict:
        import vosk

        vosk.SetLogLevel(-1)
        model = self.ensure_loaded(model_id)
        pcm = _decode_to_pcm16_mono16k(path)
        duration = len(pcm) / 2 / 16000
        recognizer = vosk.KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        segments: list[dict] = []
        words: list[dict] = []
        pieces: list[str] = []
        for offset in range(0, len(pcm), 8000):
            chunk = pcm[offset:offset + 8000]
            if recognizer.AcceptWaveform(chunk):
                segments.append(json.loads(recognizer.Result()))
        segments.append(json.loads(recognizer.FinalResult()))
        for segment in segments:
            text = segment.get("text", "")
            if text:
                pieces.append(text)
            for word in segment.get("result", []):
                words.append({"word": word["word"], "start": word["start"], "end": word["end"],
                              "conf": word.get("conf")})
        return {"text": " ".join(pieces).strip(), "duration": duration, "words": words or None,
                "segments": _segments_from_utterances(segments)}


def _segments_from_utterances(utterances: list[dict]) -> list[dict]:
    segments = []
    for utterance in utterances:
        text = utterance.get("text", "")
        if not text:
            continue
        result = utterance.get("result") or []
        start = result[0]["start"] if result else 0.0
        end = result[-1]["end"] if result else 0.0
        segments.append({"text": text, "start": start, "end": end})
    return segments


def _decode_to_pcm16_mono16k(path: str) -> bytes:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg ist nicht installiert – bitte ffmpeg installieren (z. B. apt install ffmpeg).")
    process = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Audio konnte nicht dekodiert werden: {process.stderr.decode(errors='replace')[:400]}")
    return process.stdout
