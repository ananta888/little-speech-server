"""recasepunc: Zeichensetzung und Großschreibung für deutsche Transkripte."""

import logging
import threading
import time

from ..catalog import recasepunc_available, recasepunc_dir
from ..config import Settings

log = logging.getLogger("lss.recasepunc")


class NotAvailable(RuntimeError):
    pass


class RecasepuncService:
    """Lädt den offiziellen Checkpoint (ca. 1,3 GB, teils RAM) lazy und hält ihn resident."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._predictor = None
        self._loading = False

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    @property
    def loading(self) -> bool:
        return self._loading

    def ensure_loaded(self):
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            if not recasepunc_available(self.settings):
                raise NotAvailable(
                    "recasepunc-Modell ist nicht vorhanden. Bitte zuerst im Modellbereich "
                    f"herunterladen (erwartet: {recasepunc_dir(self.settings) / 'checkpoint'})."
                )
            from .vendor.recasepunc_official import CasePuncPredictor

            device = self.settings.resolve_device()
            self._loading = True
            try:
                log.info("loading recasepunc checkpoint (device=%s)", device)
                self._predictor = CasePuncPredictor(
                    str(recasepunc_dir(self.settings) / "checkpoint"), lang="de", device=device
                )
            finally:
                self._loading = False
            return self._predictor

    def process(self, text: str) -> tuple[str, int]:
        if not text.strip():
            raise ValueError("Bitte zuerst einen Text eingeben.")
        started = time.perf_counter()
        predictor = self.ensure_loaded()
        with self._lock:
            output_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    output_lines.append("")
                    continue
                tokens = list(enumerate(predictor.tokenize(stripped)))
                result = ""
                for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
                    prediction = predictor.map_punc_label(
                        predictor.map_case_label(token[1], case_label), punc_label
                    )
                    result += (" " + prediction) if not token[1].startswith("#") else prediction
                output_lines.append(result.strip())
        return "\n".join(output_lines), int((time.perf_counter() - started) * 1000)
