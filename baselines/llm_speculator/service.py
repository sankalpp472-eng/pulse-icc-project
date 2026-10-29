"""
The LLM speculator as a live prewarm source: the same API as predictor/service.py, so the replay
(w1_replay.py --predictor-url) and the prewarm controller work with it unchanged. C3 with this service
instead of the LightGBM predictor = the speculative-actions baseline on the cluster.

    PREDICTOR_DIR=predictor CRD_MODE=k8s uvicorn service:app --app-dir baselines/llm_speculator --port 8001

  POST /predict   hint + "history" (the conversation before the step, compact messages as in steps.py)
                  -> the LLM's ranked guess; tools with p >= THETA become PrewarmRequests (k = 1)
  GET  /healthz   model, theta, CRD writer state, LLM call counts

Settings: SPEC_MODEL (default openai/gpt-oss-20b), THETA (0.3), TTL_S (60), CRD_MODE (off|dryrun|k8s),
          SPEC_MAX_TOKENS (1024 for gpt-oss, else 150), SPEC_TIMEOUT_S (5: a later guess is useless).
Groq keys: GROQ_API_KEY_1..N in agent/.env, round-robin. A key cooling down after a per-minute 429 is
skipped; if none is free the hint gets NO guess instead of waiting (it would arrive after the call).
The speculator reads conversation text, unlike the feature models: hints carry `history` only for it.
"""

import os
import threading
import time
from typing import Optional

os.environ.setdefault("PREDICTOR_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "predictor"))

from fastapi import FastAPI  # noqa: E402
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import RESPOND  # noqa: E402  (puts predictor/ on sys.path)
from crd_writer import CRDWriter  # noqa: E402
from prompt import build_messages, domain_actions, parse  # noqa: E402
from speculate import KeyPool, groq_keys  # noqa: E402

MODEL = os.getenv("SPEC_MODEL", "openai/gpt-oss-20b")
THETA = float(os.getenv("THETA", "0.3"))
TTL_S = int(os.getenv("TTL_S", "60"))
TIMEOUT_S = float(os.getenv("SPEC_TIMEOUT_S", "5"))
MAX_TOKENS = int(os.getenv("SPEC_MAX_TOKENS", "1024" if MODEL.startswith("openai/gpt-oss") else "150"))
ACTIONS = domain_actions("airline")
NAMES = [n for n, _ in ACTIONS]
POOL = KeyPool(groq_keys(), timeout=TIMEOUT_S)
LOCK = threading.Lock()
STATS = {"hints": 0, "answered": 0, "no_key_free": 0, "errors": 0, "unparsed": 0}
WRITER = CRDWriter()


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
    theta: float
    ttl_s: int
    latency_ms: float
    note: str = ""


app = FastAPI(title="Pulse LLM speculator (speculative-actions baseline)")


def free_key():
    """A key that is neither dead nor cooling down, or None (never sleeps)."""
    with LOCK:
        now = time.time()
        n = len(POOL.clients)
        for off in range(n):
            i = (POOL.next + off) % n
            if i not in POOL.dead and POOL.cool_until[i] <= now:
                POOL.next = i + 1
                return i
    return None


def guess(messages):
    """-> ({action: p} or None, note). One try per free key, no waiting."""
    extra = {"response_format": {"type": "json_object"}}
    if MODEL.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"
    for _ in range(len(POOL.clients)):
        i = free_key()
        if i is None:
            STATS["no_key_free"] += 1
            return None, "no Groq key free (rate limited)"
        try:
            r = POOL.clients[i].chat.completions.create(model=MODEL, messages=messages, temperature=0.0,
                                                        max_tokens=MAX_TOKENS, **extra)
        except RateLimitError as e:
            with LOCK:
                POOL.rate_limited(i, e)
            continue
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            STATS["errors"] += 1
            return None, f"{type(e).__name__}: {str(e)[:120]}"
        p = parse(r.choices[0].message.content, NAMES)
        if p is None:
            STATS["unparsed"] += 1
            return None, "unparsable answer"
        return p, ""
    STATS["no_key_free"] += 1
    return None, "every key rate limited"


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": f"llm-speculator:{MODEL}", "theta": THETA, "ttl_s": TTL_S,
            "keys": len(POOL.clients), "dead_keys": len(POOL.dead), "stats": STATS,
            "crd": {"mode": WRITER.mode, **WRITER.stats, "last_error": WRITER.last_error}}


@app.post("/predict", response_model=Prediction)
def predict(h: Hint):                     # sync: FastAPI runs it in a thread pool, hints overlap
    t0 = time.perf_counter()
    STATS["hints"] += 1
    traj = h.trajectory or ([h.current_tool] if h.current_tool else [])
    p, note = guess(build_messages(h.history, traj, ACTIONS))
    ranked = sorted((p or {}).items(), key=lambda kv: -kv[1])
    warm = [(t, v) for t, v in ranked if v >= THETA and t not in (RESPOND, "END")]
    if p is not None:
        STATS["answered"] += 1
    WRITER.submit([(t, v, 1) for t, v in warm], THETA, TTL_S, episode_id=h.episode_id,
                  workflow_id=h.workflow_id or "W1")
    return Prediction(workflow_id=h.workflow_id or "W1", episode_id=h.episode_id, model=f"llm-speculator:{MODEL}",
                      ranked=[ToolProb(tool=t, p=round(v, 4)) for t, v in ranked],
                      prewarm=[ToolProb(tool=t, p=round(v, 4), k=1) for t, v in warm],
                      theta=THETA, ttl_s=TTL_S, latency_ms=round((time.perf_counter() - t0) * 1000, 1), note=note)
