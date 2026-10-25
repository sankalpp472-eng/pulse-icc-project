"""
Step 6: the prediction API.

    uvicorn service:app --app-dir predictor --port 8000          # run from the repo root

  POST /predict   hint in  -> ranked next actions + the tools to prewarm (P >= theta), each with k replicas
  GET  /healthz   "ok" once the model is loaded
  GET  /metrics   Prometheus metrics (FR-11)
  GET  /docs      interactive test page (FastAPI)

Settings (environment variables):
  PREDICTOR_MODEL  tool-only models: ensemble | trajectory | history | graph | lr   (default ensemble)
                   next-action models (14 tools + RESPOND, fan-out k): gbt (main; lgbm = alias) |
                   lr_action | ngram_action. They need predictor/data/action_model.pkl
                   (build_action_data.py + train_action.py) and hints that carry "signals"
                   (agent/hints.py); a hint without signals is answered by the ensemble
  ACTION_MODEL_PATH  default predictor/data/action_model.pkl
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

import action_model  # noqa: F401  (needed so pickle can find the action-model classes)
import models  # noqa: F401  (needed so pickle can find the model classes)
from crd_writer import CRDWriter
from models import ACTION_MODELS, MODEL_NAMES, POOLED, prewarm_set

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", HERE / "data" / "model.pkl"))
MODEL_NAME = os.getenv("PREDICTOR_MODEL", "ensemble")
THETA = float(os.getenv("THETA", "0.3"))
TTL_S = int(os.getenv("TTL_S", "60"))
AUDIT_LOG = os.getenv("AUDIT_LOG")

with open(MODEL_PATH, "rb") as f:
    PRED = pickle.load(f)

if MODEL_NAME not in MODEL_NAMES and MODEL_NAME not in ACTION_MODELS:
    raise SystemExit(f"Unknown PREDICTOR_MODEL={MODEL_NAME}; use one of {list(MODEL_NAMES) + list(ACTION_MODELS)}")
ACTION_PATH = Path(os.getenv("ACTION_MODEL_PATH", HERE / "data" / "action_model.pkl"))
PRED.action = None
if MODEL_NAME in ACTION_MODELS:
    if not ACTION_PATH.exists():
        raise SystemExit(f"PREDICTOR_MODEL={MODEL_NAME} needs {ACTION_PATH}: run predictor/build_action_data.py "
                         "and predictor/train_action.py")
    with open(ACTION_PATH, "rb") as f:
        PRED.action = pickle.load(f)

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
    signals: Optional[dict[str, float]] = Field(
        None, description="numbers / yes-no flags from predictor/signals.py (no raw text); "
                          "used by the next-action models")


class ToolProb(BaseModel):
    tool: str                 # a tool, or RESPOND / END
    p: float
    k: Optional[int] = None   # prewarm list only: fan-out width = desiredReplicas in the PrewarmRequest


class Prediction(BaseModel):
    workflow_id: str          # the model actually used
    episode_id: str
    model: str
    ranked: list[ToolProb]
    prewarm: list[ToolProb]   # tools with p >= theta (never END or RESPOND), with k
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
            "action_models": PRED.action.names if PRED.action else [],
            "theta": THETA, "ttl_s": TTL_S,
            "crd": {"mode": WRITER.mode, **WRITER.stats, "last_error": WRITER.last_error}}


@app.post("/predict", response_model=Prediction)
async def predict(h: Hint):
    t0 = time.perf_counter()
    traj = h.trajectory or ([h.current_tool] if h.current_tool else [])
    wf = h.workflow_id if h.workflow_id in PRED.models else POOLED

    ranked = PRED.rank(wf, traj, MODEL_NAME, h.signals)
    warm = set(prewarm_set(ranked, THETA))
    action_used = MODEL_NAME in ACTION_MODELS and PRED.action is not None and h.signals is not None
    # fan-out width k per prewarmed tool: pending-ID rule (action models), else 1
    k = {t: (PRED.action.fanout.k(t, h.signals) if action_used else 1) for t in warm}
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
                                        "action": "prewarm", "replicas": k[t], "ttl_s": TTL_S}) + "\n")

    # Step 9: hand the prewarm list to the controller (background thread, returns at once)
    WRITER.submit([(t, p, k[t]) for t, p in ranked if t in warm], THETA, TTL_S,
                  episode_id=h.episode_id, workflow_id=wf)

    used = MODEL_NAME if MODEL_NAME not in ACTION_MODELS or action_used else "ensemble (no signals in hint)"
    return Prediction(
        workflow_id=wf, episode_id=h.episode_id, model=used,
        ranked=[ToolProb(tool=t, p=round(p, 6)) for t, p in ranked],
        prewarm=[ToolProb(tool=t, p=round(p, 6), k=k[t]) for t, p in ranked if t in warm],
        theta=THETA, ttl_s=TTL_S, latency_ms=round(latency * 1000, 3))
