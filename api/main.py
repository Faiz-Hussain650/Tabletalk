"""Web app: order by text or voice in the browser, and a live kitchen display.

    uv run uvicorn api.main:app --reload --port 8000
    open http://localhost:8000          (customer page)
    open http://localhost:8000/kitchen  (kitchen display)

Voice in the browser: the page records a clip (push-to-talk), sends it over a WebSocket,
the server runs Whisper -> agent -> Kokoro and sends back the text and a WAV to play.
Microphone access only works on http://localhost or over https.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.config import path
from agent.db import OrderDB
from agent.order_state import Menu
from agent.text_agent import OrderingAgent, build_agent

STATIC = Path(__file__).parent / "static"
SESSION_TTL_S = 30 * 60

db = OrderDB(path("db"))
sessions: dict[str, tuple[OrderingAgent, float]] = {}
_voice: dict[str, Any] = {}  # lazily loaded STT/TTS so text mode works without voice extras
_voice_lock = threading.Lock()


def _get_voice() -> tuple[Any, Any]:
    with _voice_lock:
        if not _voice:
            from voice.stt import build_stt
            from voice.tts import build_tts

            menu = Menu.load(path("menu"))
            stt, tts = build_stt(menu), build_tts()
            stt.warm_up()
            tts.warm_up()
            _voice["stt"], _voice["tts"] = stt, tts
    return _voice["stt"], _voice["tts"]


def _warm_up() -> None:
    """Load the LLM, Whisper and Kokoro at startup so the first customer doesn't wait."""
    t = time.perf_counter()
    try:
        build_agent().prime()  # loads the model AND caches the system prompt + tools
        print(f"[warm-up] LLM ready in {time.perf_counter() - t:.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"[warm-up] LLM not ready: {e}")
    try:
        _get_voice()
        print(f"[warm-up] voice ready in {time.perf_counter() - t:.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"[warm-up] voice not available (text still works): {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):  # noqa: ANN201
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="TableTalk", lifespan=lifespan)


def _agent(session_id: str) -> OrderingAgent:
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(404, "Unknown session. Start a new order.")
    sessions[session_id] = (entry[0], time.time())
    return entry[0]


def _cleanup() -> None:
    now = time.time()
    for sid in [s for s, (_, t) in sessions.items() if now - t > SESSION_TTL_S]:
        sessions.pop(sid, None)


# ------------------------------------------------------------------ pages


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/kitchen")
def kitchen() -> FileResponse:
    return FileResponse(STATIC / "kitchen.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "sessions": len(sessions), "voice_loaded": bool(_voice)}


# ------------------------------------------------------------------ text API


class ChatIn(BaseModel):
    session_id: str
    text: str


@app.post("/api/session")
async def new_session() -> dict[str, Any]:
    _cleanup()
    agent = await run_in_threadpool(build_agent)
    sessions[agent.session_id] = (agent, time.time())
    return {"session_id": agent.session_id, "greeting": agent.greet()}


@app.post("/api/chat")
async def chat(body: ChatIn) -> dict[str, Any]:
    agent = _agent(body.session_id)
    turn = await run_in_threadpool(agent.respond, body.text)
    return {
        "reply": turn.reply,
        "cart": turn.cart,
        "done": turn.done,
        "tool_calls": turn.tool_calls,
        "timings": {"llm_ms": round(turn.llm_ms), "total_ms": round(turn.total_ms)},
    }


@app.get("/api/orders")
def orders(include_done: bool = False) -> list[dict]:
    return db.list_orders(include_done=include_done)


@app.post("/api/orders/{order_id}/status/{status}")
def set_status(order_id: int, status: str) -> dict[str, Any]:
    try:
        db.set_status(order_id, status)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


# ------------------------------------------------------------------ voice WebSocket


@app.websocket("/ws/voice/{session_id}")
async def voice_ws(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    try:
        agent = _agent(session_id)
        stt, tts = await run_in_threadpool(_get_voice)
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"error": f"Voice not available: {e}"})
        await ws.close()
        return

    from voice.stt import decode_audio_bytes, is_speech
    from voice.tts import split_sentences, to_wav_bytes

    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            t0 = time.perf_counter()
            audio = await run_in_threadpool(decode_audio_bytes, audio_bytes)
            if not is_speech(audio):  # too short or too quiet: Whisper would invent words
                await ws.send_json({"transcript": "", "reply": "Sorry, I didn't catch that. Hold the button while you talk."})
                continue
            text, stt_ms = await run_in_threadpool(stt.transcribe, audio)
            if not text:
                await ws.send_json({"transcript": "", "reply": "Sorry, I didn't catch that."})
                continue
            turn = await run_in_threadpool(agent.respond, text)
            # 1) send the text right away so it shows on screen
            await ws.send_json({
                "transcript": text,
                "reply": turn.reply,
                "cart": turn.cart,
                "done": turn.done,
                "timings": {"stt_ms": round(stt_ms), "agent_ms": round(turn.total_ms)},
            })
            # 2) then stream the audio sentence by sentence: the first sentence plays while the rest is made
            first_audio_ms = None
            for sentence in split_sentences(turn.reply):
                samples, sr, _ms = await run_in_threadpool(tts.synth, sentence)
                await ws.send_json({"audio_b64": base64.b64encode(to_wav_bytes(samples, sr)).decode()})
                if first_audio_ms is None:
                    first_audio_ms = round((time.perf_counter() - t0) * 1000)
                    await ws.send_json({"first_audio_ms": first_audio_ms})
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------ kitchen feed


@app.websocket("/ws/kitchen")
async def kitchen_ws(ws: WebSocket) -> None:
    """Pushes open orders every 2 seconds. Works for orders from the web, CLI, voice loop or MCP."""
    await ws.accept()
    last_payload = None
    try:
        while True:
            payload = await run_in_threadpool(db.list_orders)
            if payload != last_payload:
                await ws.send_json(payload)
                last_payload = payload
            await asyncio.sleep(2)
    except (WebSocketDisconnect, RuntimeError):
        pass
