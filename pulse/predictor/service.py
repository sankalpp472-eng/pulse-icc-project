"""
The prediction API: one service, any of the four predictors (predictor/predictors/), picked by PREDICTOR.

    PREDICTOR=lightgbm uvicorn service:app --app-dir predictor --port 8000          # run from the repo root

  POST /predict   a PredictRequest (contract.py) -> ranked next actions + the tools to prewarm, each with k pods
  GET  /info      the predictor's name, the inputs it reads (trajectory / signals / history), what it supports
                  (fan-out, lookahead), its domains, and the prewarm policy. Hint senders read it to send exactly
                  those inputs (pulse/agent/hints.py, pulse/ui/session.py, pulse/experiments/replay.py).
  GET  /healthz   "ok" once the predictor is loaded, plus its own state
  GET  /metrics   Prometheus metrics (FR-11)
  GET  /docs      interactive test page (FastAPI)

Settings (environment variables):
  PREDICTOR        lightgbm (default) | lightgbm_adapt | markov | speculator | pulse_v3   (old names gbt, ngram_action also work;
                   PREDICTOR_MODEL is read if PREDICTOR is unset). Each predictor's own settings are in its file.
  PREWARM_POLICY   adaptive (default: the TOP_K_MAX most likely tools as candidates, each with its rank and its hit
                   rate learned online (calibration.py); the controller warms those worth a pod slot now, so how many
                   follows the predictor's confidence and the cluster) | topk (exactly the TOP_K most likely tools)
                   (policy.py). Every candidate needs P >= P_MIN
  TOP_K, TOP_K_MAX, P_MIN   (default 2, 3, 0.05)
  LOOKAHEAD        1 = on a step that is most likely a reply (P(RESPOND) >= LOOKAHEAD_MIN_RESPOND, default 0.5),
                   also prewarm the tool the agent calls after the customer's reply (P >= THETA_NEXT, default 0.4).
                   Only predictors that support it (lightgbm, markov with a lookahead model in action_model.pkl)
  TTL_S            TTL the controller should give a prewarmed pod  (default 60; with TTL_MODE=fixed, every tool)
  TTL_MODE         step (default): TTL by when the call is expected -- TTL_STEP_S (default 20) for a next-step
                   prediction (the agent's think time), TTL_AHEAD_S (default 75) for a lookahead one (the customer's
                   reply first) | fixed: TTL_S for all. Only the lease controller (CONTROLLER_MODE=lease) uses it;
                   the ping controller's TTL is the Knative idle window
  AUDIT_LOG        optional file: one JSON line per prewarm decision (FR-12)
  CRD_MODE         off | dryrun | k8s : write PrewarmRequest objects for the controller (crd_writer.py; default off)

Example request:
  {"workflow_id": "W1", "episode_id": "task1-123", "trajectory": ["get_user_details"], "signals": {...}}
"""

import json
import os
import time

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app

import pulse.predictor.predictors as predictors
from pulse.predictor.contract import NO_PREWARM, RESPOND, ActionP, PredictionOut, PredictRequest
from pulse.predictor.crd_writer import CRDWriter
from pulse.predictor.calibration import Calibrator
from pulse.predictor.policy import prewarm_select

NAME = predictors.canonical(os.getenv("PREDICTOR") or os.getenv("PREDICTOR_MODEL") or predictors.DEFAULT)
PREWARM_POLICY = os.getenv("PREWARM_POLICY", "adaptive")    # adaptive | topk (policy.py)
TOP_K = int(os.getenv("TOP_K", "2"))
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "3"))
P_MIN = float(os.getenv("P_MIN", "0.05"))
LOOKAHEAD = os.getenv("LOOKAHEAD", "0") == "1"
THETA_NEXT = float(os.getenv("THETA_NEXT", "0.4"))
LOOKAHEAD_MIN_RESPOND = float(os.getenv("LOOKAHEAD_MIN_RESPOND", "0.5"))
TTL_S = int(os.getenv("TTL_S", "60"))
TTL_MODE = os.getenv("TTL_MODE", "step")                    # step | fixed
TTL_STEP_S = int(os.getenv("TTL_STEP_S", "20"))
TTL_AHEAD_S = int(os.getenv("TTL_AHEAD_S", "75"))
AUDIT_LOG = os.getenv("AUDIT_LOG")

PRED = predictors.load(NAME)
CALIB = Calibrator()          # adaptive: how often a candidate of probability p is really the next call
WRITER = CRDWriter()   # CRD_MODE=off|dryrun|k8s

# ---- Prometheus metrics (the same names for every predictor: infra/monitoring dashboards)
#      own registry, so tests can import the service once per predictor in one process
REGISTRY = CollectorRegistry()
LATENCY = Histogram("predictor_request_seconds", "Time to answer one /predict call",
                    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
                    registry=REGISTRY)
HINTS = Counter("predictor_hints_total", "Hints received", ["workflow_id"], registry=REGISTRY)
PREWARMS = Counter("predictor_prewarm_candidates_total", "Tools chosen for prewarming", ["tool"], registry=REGISTRY)

app = FastAPI(title=f"Pulse next-tool predictor ({NAME})")
app.mount("/metrics", make_asgi_app(registry=REGISTRY))


def policy():
    return {"name": PREWARM_POLICY, "top_k": TOP_K, "top_k_max": TOP_K_MAX, "p_min": P_MIN, "ttl_s": TTL_S,
            "ttl": {"mode": TTL_MODE, "step_s": TTL_STEP_S, "ahead_s": TTL_AHEAD_S},
            "lookahead": {"on": LOOKAHEAD and PRED.supports_lookahead, "theta_next": THETA_NEXT,
                          "min_respond": LOOKAHEAD_MIN_RESPOND}}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/info")
async def info():
    return {**PRED.info(), "policy": policy()}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", **PRED.health(), "predictor": NAME, "policy": policy(),    # the name wins over health()
            "crd": {"mode": WRITER.mode, **WRITER.stats, "last_error": WRITER.last_error},
            "calibration": CALIB.stats()}


@app.post("/predict", response_model=PredictionOut)
def predict(req: PredictRequest):          # sync: FastAPI runs it in a thread pool (the speculator waits on an LLM)
    t0 = time.perf_counter()
    probs = PRED.predict(req)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    warm = prewarm_select(ranked, PREWARM_POLICY, TOP_K, P_MIN, TOP_K_MAX)
    k = {t: PRED.fanout(t, req) if PRED.supports_fanout else 1 for t in warm}
    # adaptive: what the previous step really did scores its candidates; each new candidate carries its rank and
    # its calibrated hit rate, and the controller decides which are worth a pod slot (pulse/controller/adaptive.py)
    CALIB.observe(req.episode_id, req.tools(), [(t, probs[t]) for t in warm])
    adaptive = PREWARM_POLICY == "adaptive"
    rank = {t: i + 1 for i, t in enumerate(warm)}
    hit = {t: round(CALIB.rate(probs[t]), 4) for t in warm} if adaptive else {}

    # Lookahead: this step is most likely a reply to the customer, so also prewarm the tool the agent will call
    # AFTER the customer answers (customer reply + think time = seconds of lead, enough for a whole cold start)
    ahead = {}
    if LOOKAHEAD and PRED.supports_lookahead and probs.get(RESPOND, 0.0) >= LOOKAHEAD_MIN_RESPOND:
        ahead = {t: p for t, p in PRED.lookahead(req, probs).items()
                 if p >= THETA_NEXT and t not in NO_PREWARM and t not in k}
        k.update({t: 1 for t in ahead})
    pw = [(t, probs[t]) for t in warm] + sorted(ahead.items(), key=lambda kv: -kv[1])
    ttl = {t: TTL_S if TTL_MODE == "fixed" else (TTL_AHEAD_S if t in ahead else TTL_STEP_S) for t, _ in pw}
    latency = time.perf_counter() - t0

    wf = req.workflow_id or req.domain
    LATENCY.observe(latency)
    HINTS.labels(wf).inc()
    for t, _ in pw:
        PREWARMS.labels(t).inc()
    if AUDIT_LOG and pw:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            for t, p in pw:
                f.write(json.dumps({"time": time.time(), "predictor": NAME, "tool": t, "probability": round(p, 4),
                                    "action": "prewarm-lookahead" if t in ahead else "prewarm",
                                    "replicas": k[t], "ttl_s": ttl[t]}) + "\n")

    # hand the prewarm list to the controller (background thread, returns at once)
    WRITER.submit([(t, p, k[t], ttl[t], t in ahead, rank.get(t), hit.get(t)) for t, p in pw], P_MIN, TTL_S,
                  episode_id=req.episode_id, workflow_id=wf)

    used = [i for i in PRED.inputs if (i == "trajectory") or (i == "signals" and req.signals is not None)
            or (i == "history" and req.history)]
    note = PRED.note
    if "signals" in PRED.inputs and req.signals is None:
        note = "no signals in the request: answered by the Markov chain"
    return PredictionOut(
        workflow_id=wf, episode_id=req.episode_id, model=NAME,
        ranked=[ActionP(tool=t, p=round(p, 6)) for t, p in ranked],
        prewarm=[ActionP(tool=t, p=round(p, 6), k=k[t], ttl_s=ttl[t], rank=rank.get(t), hit=hit.get(t))
                 for t, p in pw],
        lookahead=[ActionP(tool=t, p=round(p, 6), k=1, ttl_s=ttl[t]) for t, p in ahead.items()],
        policy=PREWARM_POLICY, ttl_s=TTL_S, latency_ms=round(latency * 1000, 3), inputs_used=used, note=note)
