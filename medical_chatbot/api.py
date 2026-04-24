"""
api.py — FastAPI backend for the Medical Chatbot UI.

Endpoints:
  GET  /api/health    — model loading status
  POST /api/chat      — SSE streaming inference (original / finetuned / compare)
  GET  /api/metrics   — evaluation results from results.json
  GET  /              — serve frontend/index.html
  GET  /static/...    — static assets (if any)
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_FILE

# ── Model state ───────────────────────────────────────────────
_generate_original  = None
_generate_finetuned = None
_model_status       = "loading"   # loading | ready | error
_model_error        = ""


# ── Lifespan: load models on startup ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generate_original, _generate_finetuned, _model_status, _model_error
    try:
        loop = asyncio.get_event_loop()
        _generate_original, _generate_finetuned = await loop.run_in_executor(
            None, _load_models_sync
        )
        _model_status = "ready"
    except Exception as exc:
        _model_status = "error"
        _model_error  = str(exc)
        print(f"[ERROR] Model loading failed: {exc}")
    yield


def _load_models_sync():
    from utils.inference import load_models
    return load_models()


# ── App ───────────────────────────────────────────────────────

app = FastAPI(title="Medical Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


# ── Request / Response models ─────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    mode: str = "finetuned"   # "original" | "finetuned" | "compare"


# ── Helpers ───────────────────────────────────────────────────

def _run_model(fn, question: str) -> tuple[str, float]:
    """Synchronously run a generation function and return (text, time)."""
    return fn(question)


# ── Routes ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":  _model_status,
        "error":   _model_error,
        "device":  str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
    }


@app.get("/api/metrics")
async def metrics():
    if not os.path.exists(RESULTS_FILE):
        raise HTTPException(404, detail="results.json not found — run 3_evaluate.py first.")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if _model_status != "ready":
        raise HTTPException(503, detail=f"Models not ready ({_model_status})")
    if not req.question.strip():
        raise HTTPException(400, detail="Empty question.")

    async def event_stream() -> AsyncIterator[dict]:
        question = req.question.strip()
        loop = asyncio.get_running_loop()

        try:
            if req.mode in ("original", "compare"):
                yield {"event": "typing", "data": json.dumps({"model": "original"})}
                try:
                    text, elapsed = await loop.run_in_executor(
                        None, _run_model, _generate_original, question
                    )
                    print(f"[INFO] original: {elapsed:.1f}s, {len(text)} chars")
                except Exception as exc:
                    print(f"[ERROR] original generation failed: {exc}")
                    raise
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "model":   "original",
                        "text":    text or "(no response)",
                        "elapsed": round(elapsed, 2),
                    }),
                }

            if req.mode in ("finetuned", "compare"):
                yield {"event": "typing", "data": json.dumps({"model": "finetuned"})}
                try:
                    text, elapsed = await loop.run_in_executor(
                        None, _run_model, _generate_finetuned, question
                    )
                    print(f"[INFO] finetuned: {elapsed:.1f}s, {len(text)} chars")
                except Exception as exc:
                    print(f"[ERROR] finetuned generation failed: {exc}")
                    raise
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "model":   "finetuned",
                        "text":    text or "(no response)",
                        "elapsed": round(elapsed, 2),
                    }),
                }

            yield {"event": "done", "data": "{}"}

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_stream())


# ── Static / frontend ─────────────────────────────────────────

@app.get("/")
async def index():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, detail="frontend/index.html not found.")
    return FileResponse(path)


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    print("[INFO] Starting Medical Chatbot API on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
