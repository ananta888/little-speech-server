"""Rechengerät zur Laufzeit (von der UI änderbar), unabhängig von der Config-Datei."""

import threading

from .config import get_settings

_lock = threading.Lock()
_override: str | None = None  # None = Wert aus LSS_DEVICE/Default


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 – ohne torch ohnehin CPU
        return False


def requested_device() -> str:
    with _lock:
        return _override or get_settings().device


def resolve_device() -> str:
    """Was tatsächlich verwendet wird: auto → cuda falls verfügbar, sonst der Wunsch."""
    wanted = requested_device()
    if wanted == "cpu":
        return "cpu"
    if wanted == "cuda":
        return "cuda" if _cuda_available() else "cpu"
    return "cuda" if _cuda_available() else "cpu"


def set_device(value: str) -> str:
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unbekanntes Gerät: {value} (erlaubt: auto, cpu, cuda)")
    global _override
    with _lock:
        _override = value
    return requested_device()
