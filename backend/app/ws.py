"""WebSocket-Streaming für Live-Mikrofon-Aufnahmen (PCM16, 16 kHz, mono).

Protokoll:
  Client → Text:  {"type":"config","engine":"vosk"|"whisper","modelId":"..."}
  Client → Bytes: PCM16-LE-Frames (16 kHz, mono)
  Client → Text:  {"type":"stop"}
  Server → Text:  ready | partial | final | error | end
"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from .api import get_vosk_engine, get_whisper_engine
from .engines.whisper_engine import WhisperStreamSession

log = logging.getLogger("lss.ws")

router = APIRouter()


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = None
    engine_name = None
    try:
        config = json.loads(await websocket.receive_text())
        if config.get("type") != "config":
            raise RuntimeError("Erste Nachricht muss {type:'config', engine, modelId} sein.")
        engine_name = config.get("engine")
        model_id = config.get("modelId")
        sample_rate = int(config.get("sampleRate") or 16000)
        if engine_name == "vosk":
            session = await run_in_threadpool(get_vosk_engine().create_session, model_id, sample_rate)
        elif engine_name == "whisper":
            session = WhisperStreamSession(get_whisper_engine(), model_id)
        else:
            raise RuntimeError(f"Unbekannte Engine: {engine_name}")
        await websocket.send_text(json.dumps({"type": "ready"}))

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            payload = message.get("bytes")
            if payload:
                events = await run_in_threadpool(session.accept, payload)
                for event in events:
                    await websocket.send_text(json.dumps(event))
                continue
            text = message.get("text")
            if not text:
                continue
            command = json.loads(text)
            if command.get("type") == "stop":
                break
    except WebSocketDisconnect:
        session = None
    except Exception as error:  # noqa: BLE001 – Fehler dem Client melden
        log.exception("stream failed")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(error)}))
        except Exception:  # noqa: BLE001
            pass
    finally:
        if session is not None:
            try:
                final = await run_in_threadpool(session.finish)
                await websocket.send_text(json.dumps(final))
            except Exception:  # noqa: BLE001
                log.exception("finish failed")
            try:
                await websocket.send_text(json.dumps({"type": "end"}))
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass
