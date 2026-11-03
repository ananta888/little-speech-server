"""Zentrale Einstellungen (Umgebung: Präfix LSS_, z. B. LSS_PORT=9000)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist" / "frontend" / "browser"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LSS_", env_file=REPO_ROOT / ".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    models_dir: Path = REPO_ROOT / "models"
    device: str = "auto"  # auto | cpu | cuda
    cors_origins: str = "http://localhost:4300,http://127.0.0.1:4300"
    max_upload_mb: int = 512
    whisper_language: str = ""  # leer = automatische Spracherkennung

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def hf_cache(self) -> Path:
        return self.models_dir / "hf-cache"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings
