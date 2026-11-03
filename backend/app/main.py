"""FastAPI-Anwendung: API + WebSocket + Auslieferung des Angular-Builds."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import api, ws
from .config import FRONTEND_DIST, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.environ.setdefault("HF_HOME", str(settings.hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.hf_cache))
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="little speech server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api.router)
    app.include_router(ws.router)
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    return app


app = create_app()
