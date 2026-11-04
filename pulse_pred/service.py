"""
pulse-v3 as a live prewarm source: the same API as predictor/service.py, so the replay
(w1_replay.py --predictor-url --hint-context), the UI and the prewarm controller work with it unchanged.

    CRD_MODE=k8s uvicorn service:app --app-dir pulse_pred --port 8000          # run from the repo root

  POST /predict   hint + "history" (compact messages, as for the LLM speculator) -> ranked next tools;
                  the tools picked by PREWARM_POLICY become PrewarmRequests (k = 1)
  GET  /healthz   model, policy, CRD writer state, sessions, embedding server state
  GET  /metrics   Prometheus, the same metric names as predictor/service.py (deploy/monitoring dashboards)

pulse-v3 follows a session: the first hint of an episode starts it (first customer message), and every
new tool step in the trajectory is replayed into it as a call (with its arguments, from the history) and
a result (the tool outputs, from the history). It reads conversation text, so hints need `history`;
without it the step's tool names are still used, with no arguments or results (lower accuracy).
Sessions idle for SESSION_TIMEOUT_S end, and the online adapter learns from them.

Settings: THETA (0.3), PREWARM_POLICY theta | topk, TOP_K (2), P_MIN (0.1), TTL_S (60),
          CRD_MODE (off | dryrun | k8s), PULSE_DOMAIN (airline), SESSION_TIMEOUT_S (300),
          PULSE_EMBED_BASE (http://127.0.0.1:8091/v1; down -> zero embedding, "embedding": "unavailable"),
          PULSE_EMBED_TIMEOUT_S (1.0).
Model: PULSE_MODEL_URI (s3://<bucket>/pulse-v3 = its latest version, or .../pulse-v3/<version>; downloaded
       once by model_store.py), else the local file PULSE_MODEL (default pulse_pred/data/predictor.pkl).
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "predictor"):                 # pulse_pred package; predictor/ for crd_writer, models
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI  # noqa: E402
from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from crd_writer import CRDWriter  # noqa: E402
from models import prewarm_select  # noqa: E402
from pulse_pred import model_store  # noqa: E402
from pulse_pred.contract.hints import Call, End, Hint as SessionHint  # noqa: E402
from pulse_pred.predictors.base import PredictorError  # noqa: E402
from pulse_pred.predictors.pulse_v3.embed import Embedder  # noqa: E402
from pulse_pred.predictors.pulse_v3.predictor import MODEL_PATH, PulseV3Predictor  # noqa: E402

THETA = float(os.getenv("THETA", "0.3"))
PREWARM_POLICY = os.getenv("PREWARM_POLICY", "theta")
TOP_K = int(os.getenv("TOP_K", "2"))
P_MIN = float(os.getenv("P_MIN", "0.1"))
TTL_S = int(os.getenv("TTL_S", "60"))
DOMAIN = os.getenv("PULSE_DOMAIN", "airline")
SESSION_TIMEOUT_S = float(os.getenv("SESSION_TIMEOUT_S", "300"))
MODEL = "pulse-v3"
MODEL_URI = os.getenv("PULSE_MODEL_URI")

MODEL_FILE = model_store.fetch(MODEL_URI) if MODEL_URI else MODEL_PATH
PRED = PulseV3Predictor.load(path=MODEL_FILE, embedder=Embedder(timeout=float(os.getenv("PULSE_EMBED_TIMEOUT_S", "1.0"))))
WRITER = CRDWriter()
LOCK = threading.Lock()
EPISODES = {}                                         # episode_id -> what this service has told pulse-v3

# own registry: predictor/service.py registers the same names when both are imported in one process (tests)
REGISTRY = CollectorRegistry()
LATENCY = Histogram("predictor_request_seconds", "Time to answer one /predict call",
                    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25), registry=REGISTRY)
HINTS = Counter("predictor_hints_total", "Hints received", ["workflow_id"], registry=REGISTRY)
PREWARMS = Counter("predictor_prewarm_candidates_total", "Tools at or above theta", ["tool"], registry=REGISTRY)


class Hint(BaseModel):
    workflow_id: Optional[str] = None
    episode_id: str = ""
    trajectory: list[str] = Field(default_factory=list)
    current_tool: Optional[str] = None
    ts: Optional[float] = None
    signals: Optional[dict[str, float]] = None
    history: list[dict] = Field(default_factory=list, description="compact messages before this step")


class ToolProb(BaseModel):
    tool: str
    p: float
    k: Optional[int] = None


class Prediction(BaseModel):
    workflow_id: str
    episode_id: str
    model: str
    ranked: list[ToolProb]
    prewarm: list[ToolProb]
    lookahead: list[ToolProb] = []
    theta: float
    ttl_s: int
    latency_ms: float
    note: str = ""


def appended(prev, cur):
    """Tools added to a trajectory window since the last hint (the agent sends its last 8 tools)."""
    for s in range(len(prev) + 1):
        if prev[s:] == cur[:len(prev) - s]:
            return cur[len(prev) - s:]
    return cur


def last_step(history):
    """([(tool, args)], [results]) of the latest assistant message with tool calls, and the latest
    customer message, from compact messages {role, content, calls?, name?}."""
    calls, results, user = [], [], None
    for i, m in enumerate(history):
        if m.get("role") == "user":
            user = m.get("content") or ""
        if m.get("role") == "assistant" and m.get("calls"):
            calls, results = [], []
            for name, args in m["calls"]:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                calls.append((name, args if isinstance(args, dict) else {}))
            results = [x.get("content") or "" for x in history[i + 1:] if x.get("role") == "tool"][:len(calls)]
    return calls, results, user


def first_user(history):
    return next((m.get("content") or "" for m in history if m.get("role") == "user"), "")


def expire(now):
    for eid, ep in list(EPISODES.items()):
        if now - ep["t"] > SESSION_TIMEOUT_S:
            EPISODES.pop(eid)
            try:
                PRED.end(End(session_id=eid))         # the online adapter learns from the finished session
            except PredictorError:
                pass


def update(h):
    """Bring pulse-v3's session for this episode up to date; returns its latest Prediction."""
    now = time.time()
    expire(now)
    eid = h.episode_id or "default"
    traj = h.trajectory or ([h.current_tool] if h.current_tool and h.current_tool != "START" else [])
    calls, results, user = last_step(h.history)
    ep = EPISODES.get(eid)
    hint = lambda **kw: PRED.hint(SessionHint(session_id=eid, domain=DOMAIN, **kw))  # noqa: E731
    if ep is None:
        ep = EPISODES[eid] = {"traj": [], "user": first_user(h.history), "results": [], "pred": None}
        ep["pred"] = hint(phase="start", user_text=ep["user"])
    ep["t"] = now
    new = appended(ep["traj"], traj)
    if new:
        step = calls if calls and [c for c, _ in calls] == new[-len(calls):] else [(t, {}) for t in new]
        ep["results"] = results if step is calls else []
        try:
            hint(phase="call", user_text=user, calls=[Call(tool=t, args=a) for t, a in step])
            ep["pred"] = hint(phase="result", results=ep["results"])
        except PredictorError:                        # a tool pulse-v3 doesn't know: keep the last prediction
            pass
    elif user and user != ep["user"]:                 # a new customer message, no new tool step
        ep["pred"] = hint(phase="result", user_text=user, results=ep["results"]) if ep["traj"] \
            else hint(phase="start", user_text=user)
    ep["traj"], ep["user"] = list(traj), user or ep["user"]
    return ep["pred"]


app = FastAPI(title="Pulse next-tool predictor (pulse-v3)")
app.mount("/metrics", make_asgi_app(registry=REGISTRY))


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL, "model_uri": MODEL_URI, "model_file": str(MODEL_FILE), "theta": THETA, "ttl_s": TTL_S,
            "policy": {"name": PREWARM_POLICY, "top_k": TOP_K, "p_min": P_MIN},
            "sessions": len(EPISODES), "embedding": "ok" if PRED.embedding_ok else "unavailable",
            "predictor": PRED.stats(), "crd": {"mode": WRITER.mode, **WRITER.stats, "last_error": WRITER.last_error}}


@app.post("/predict", response_model=Prediction)
def predict(h: Hint):                     # sync: FastAPI runs it in a thread pool; LOCK serialises sessions
    t0 = time.perf_counter()
    with LOCK:
        p = update(h)
    strip = lambda t: t.split(".", 1)[1]  # noqa: E731   "airline.get_user_details" -> "get_user_details"
    ranked = sorted([(strip(t), v) for t, v in p.p1.items()] + [("END", p.p_end)], key=lambda kv: -kv[1])
    warm = set(prewarm_select(ranked, PREWARM_POLICY, THETA, TOP_K, P_MIN))
    pw = [(t, v) for t, v in ranked if t in warm]
    wf = h.workflow_id or "W1"
    LATENCY.observe(time.perf_counter() - t0)
    HINTS.labels(wf).inc()
    for t, _ in pw:
        PREWARMS.labels(t).inc()
    WRITER.submit([(t, v, 1) for t, v in pw], THETA, TTL_S, episode_id=h.episode_id, workflow_id=wf)
    return Prediction(workflow_id=wf, episode_id=h.episode_id, model=MODEL,
                      ranked=[ToolProb(tool=t, p=round(v, 6)) for t, v in ranked],
                      prewarm=[ToolProb(tool=t, p=round(v, 6), k=1) for t, v in pw],
                      theta=THETA, ttl_s=TTL_S, latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                      note="" if PRED.embedding_ok else "embedding server unavailable: zero embedding")
