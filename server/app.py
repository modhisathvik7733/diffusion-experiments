"""FastAPI inference server for DreamOn-v0-7B.

Endpoints:
  POST /infill   — code autocomplete (prefix + suffix → middle)
  GET  /health   — liveness + VRAM stats

Auth: shared bearer token in DIFFUSION_SHARED_SECRET. Single-tenant
(creor-backend is the only caller).

Concurrency: single asyncio.Lock around model.generate (DreamOn's
variable-length expansion can't batch). Queue depth capped at 8 → 503.
"""
from __future__ import annotations

import asyncio
import os
import time

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import inference

SHARED_SECRET = os.environ.get("DIFFUSION_SHARED_SECRET")
if not SHARED_SECRET:
    raise SystemExit("DIFFUSION_SHARED_SECRET env var not set; refusing to start")

MAX_QUEUE_DEPTH = int(os.environ.get("DIFFUSION_MAX_QUEUE", "8"))

app = FastAPI(title="DreamOn diffusion inference", version="0.1.0")

_inference_lock = asyncio.Lock()
_queue_depth = 0
_started_at = time.time()


def verify_token(request: Request) -> None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = auth[len("Bearer ") :]
    if token != SHARED_SECRET:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


class InfillRequest(BaseModel):
    prefix: str = Field(default="", description="Code before the cursor")
    suffix: str = Field(default="", description="Code after the cursor")
    max_tokens: int = Field(default=64, ge=1, le=128)
    language: str | None = Field(default=None, description="'line-mode' to truncate at first newline")


class InfillResponse(BaseModel):
    completion: str
    tokens: int
    latency_ms: int
    model: str
    finish_reason: str


@app.post("/infill", response_model=InfillResponse, dependencies=[Depends(verify_token)])
async def infill_endpoint(req: InfillRequest):
    global _queue_depth
    if not req.prefix and not req.suffix:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prefix or suffix required")
    if _queue_depth >= MAX_QUEUE_DEPTH:
        return JSONResponse(
            {"error": "queue full", "retryable": True},
            status_code=503,
            headers={"Retry-After": "1"},
        )

    _queue_depth += 1
    try:
        async with _inference_lock:
            result = await asyncio.to_thread(
                inference.infill,
                prefix=req.prefix,
                suffix=req.suffix,
                max_tokens=req.max_tokens,
                language=req.language,
            )
    finally:
        _queue_depth -= 1

    return InfillResponse(model="DreamOn-v0-7B", **result)


@app.get("/health")
async def health():
    used, total = inference.vram_usage()
    return {
        "ok": True,
        "model": "DreamOn-v0-7B",
        "model_loaded": True,
        "gpu": "cuda:0",
        "vram_used_gb": round(used, 2),
        "vram_total_gb": round(total, 2),
        "queue_depth": _queue_depth,
        "queue_max": MAX_QUEUE_DEPTH,
        "uptime_seconds": int(time.time() - _started_at),
    }
