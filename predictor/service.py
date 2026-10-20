"""
Step 6: the prediction API.

    uvicorn service:app --app-dir predictor --port 8000          # run from the repo root

  POST /predict   hint in  -> ranked next tools + the tools to prewarm (P >= theta)
  GET  /healthz   "ok" once the model is loaded
  GET  /metrics   Prometheus metrics (FR-11)
  GET  /docs      interactive test page (FastAPI)

Settings (environment variables):
  PREDICTOR_MODEL  ensemble | trajectory | history | graph | lr   (default ensemble)
  THETA            confidence threshold for prewarming             (default 0.3)
  TTL_S            TTL the controller should give a prewarmed pod  (default 60)
  MODEL_PATH       default predictor/data/model.pkl
  AUDIT_LOG        optional file: one JSON line per prewarm decision (FR-12)
  CRD_MODE         off | dryrun | k8s : write PrewarmRequest objects for the controller
                   (see crd_writer.py; default off)

Example hint:
  {"workflow_id": "W1", "episode_id": "task1-123", "trajectory": ["get_user_details"], "ts": 1792149162.1}
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

import models  # noqa: F401  (needed so pickle can find the model classes)
from crd_writer import CRDWriter
from models import POOLED, prewarm_set

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", HERE / "data" / "model.pkl"))
MODEL_NAME = os.getenv("PREDICTOR_MODEL", "ensemble")
THETA = float(os.getenv("THETA", "0.3"))
TTL_S = int(os.getenv("TTL_S", "60"))
AUDIT_LOG = os.getenv("AUDIT_LOG")

with open(MODEL_PATH, "rb") as f:
    PRED = pickle.load(f)

WRITER = CRDWriter()   # CRD_MODE=off|dryrun|k8s

# ---- Prometheus metrics
LATENCY = Histogram("predictor_request_seconds", "Time to answer one /predict call",
                    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
HINTS = Counter("predictor_hints_total", "Hints received", ["workflow_id"])
PREWARMS = Counter("predictor_prewarm_candidates_total", "Tools at or above theta", ["tool"])


class Hint(BaseModel):
    workflow_id: Optional[str] = Field(None, description="W1, W2 ... (unknown/missing -> pooled model)")
    episode_id: str = ""
    trajectory: list[str] = Field(default_factory=list, description="tools called so far, in order")
    current_tool: Optional[str] = Field(None, description="optional; used if trajectory is empty")
    ts: Optional[float] = None


class ToolProb(BaseModel):
    tool: str
    p: float


class Prediction(BaseModel):
    workflow_id: str          # the model actually used
    episode_id: str
    model: str
    ranked: list[ToolProb]
    prewarm: list[ToolProb]   # tools with p >= theta (never END)
    theta: float
    ttl_s: int
    latency_ms: float


app = FastAPI(title="Pulse next-tool predictor")
app.mount("/metrics", make_asgi_app())


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL_NAME, "workflows": sorted(PRED.models),
            "theta": THETA, "ttl_s": TTL_S,
            "crd": {"mode": WRITER.mode, **WRITER.stats, "last_error": WRITER.last_error}}


@app.post("/predict", response_model=Prediction)
async def predict(h: Hint):
    t0 = time.perf_counter()
    traj = h.trajectory or ([h.current_tool] if h.current_tool else [])
    wf = h.workflow_id if h.workflow_id in PRED.models else POOLED

    ranked = PRED.rank(wf, traj, MODEL_NAME)
    warm = set(prewarm_set(ranked, THETA))
    latency = time.perf_counter() - t0

    LATENCY.observe(latency)
    HINTS.labels(wf).inc()
    for t in warm:
        PREWARMS.labels(t).inc()
    if AUDIT_LOG and warm:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            for t, p in ranked:
                if t in warm:
                    f.write(json.dumps({"time": time.time(), "tool": t, "probability": round(p, 4),
                                        "action": "prewarm", "ttl_s": TTL_S}) + "\n")

    # Step 9: hand the prewarm list to the controller (background thread, returns at once)
    WRITER.submit([(t, p) for t, p in ranked if t in warm], THETA, TTL_S,
                  episode_id=h.episode_id, workflow_id=wf)

    return Prediction(
        workflow_id=wf, episode_id=h.episode_id, model=MODEL_NAME,
        ranked=[ToolProb(tool=t, p=round(p, 6)) for t, p in ranked],
        prewarm=[ToolProb(tool=t, p=round(p, 6)) for t, p in ranked if t in warm],
        theta=THETA, ttl_s=TTL_S, latency_ms=round(latency * 1000, 3))
