"""
pulse_v3: conversation text + tool graph + state flags with an online adapter (pulse_pred/), one model for the
airline and retail tools. It follows a session: the first hint of an episode starts it (first customer
message), and every new tool step in the trajectory is replayed into it as a call (with its arguments, from the
history) and a result (the tool outputs, from the history). Without history only the tool names are used.
Sessions idle for SESSION_TIMEOUT_S end, and the online adapter learns from them.

Model: PULSE_MODEL_URI (s3://<bucket>/pulse-v3 = its latest version, or .../pulse-v3/<version>; downloaded once by
pulse_pred/model_store.py), else the local file PULSE_MODEL (default pulse_pred/data/predictor.pkl).
Settings: SESSION_TIMEOUT_S (300), PULSE_EMBED_BASE (http://127.0.0.1:8091/v1; down -> zero embedding),
PULSE_EMBED_TIMEOUT_S (1.0).
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from contract import END, PredictRequest  # noqa: E402
from predictors.base import Predictor  # noqa: E402
from pulse_pred import model_store  # noqa: E402
from pulse_pred.contract.hints import Call, End, Hint as SessionHint  # noqa: E402
from pulse_pred.predictors.base import PredictorError  # noqa: E402
from pulse_pred.predictors.pulse_v3.embed import Embedder  # noqa: E402
from pulse_pred.predictors.pulse_v3.predictor import MODEL_PATH, PulseV3Predictor  # noqa: E402

SESSION_TIMEOUT_S = float(os.getenv("SESSION_TIMEOUT_S", "300"))
MODEL_URI = os.getenv("PULSE_MODEL_URI")


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


class PulseV3(Predictor):
    name = "pulse_v3"
    description = "pulse-v3: conversation text, tool graph and state flags, online adapter (S3 model registry)"
    inputs = ("trajectory", "history")
    domains = ("airline", "retail")

    def __init__(self):
        version = os.getenv("PULSE_MODEL_VERSION")          # pinned by GitOps (pulse-config); else latest
        uri = f"{MODEL_URI.rstrip('/')}/{version}" if MODEL_URI and version else MODEL_URI
        self.model_file = model_store.fetch(uri) if uri else MODEL_PATH
        self.pred = PulseV3Predictor.load(path=self.model_file,
                                          embedder=Embedder(timeout=float(os.getenv("PULSE_EMBED_TIMEOUT_S", "1.0"))))
        self.lock = threading.Lock()
        self.episodes = {}                    # episode_id -> what this service has told pulse-v3

    def _expire(self, now):
        for eid, ep in list(self.episodes.items()):
            if now - ep["t"] > SESSION_TIMEOUT_S:
                self.episodes.pop(eid)
                try:
                    self.pred.end(End(session_id=eid))     # the online adapter learns from the finished session
                except PredictorError:
                    pass

    def _update(self, req: PredictRequest):
        """Bring pulse-v3's session for this episode up to date; returns its latest prediction."""
        now = time.time()
        self._expire(now)
        eid = req.episode_id or "default"
        traj = req.tools()
        calls, results, user = last_step(req.history)
        ep = self.episodes.get(eid)
        hint = lambda **kw: self.pred.hint(SessionHint(session_id=eid, domain=req.domain, **kw))  # noqa: E731
        if ep is None:
            ep = self.episodes[eid] = {"traj": [], "user": first_user(req.history), "results": [], "pred": None}
            ep["pred"] = hint(phase="start", user_text=ep["user"])
        ep["t"] = now
        new = appended(ep["traj"], traj)
        if new:
            step = calls if calls and [c for c, _ in calls] == new[-len(calls):] else [(t, {}) for t in new]
            ep["results"] = results if step is calls else []
            try:
                hint(phase="call", user_text=user, calls=[Call(tool=t, args=a) for t, a in step])
                ep["pred"] = hint(phase="result", results=ep["results"])
            except PredictorError:                      # a tool pulse-v3 doesn't know: keep the last prediction
                pass
        elif user and user != ep["user"]:               # a new customer message, no new tool step
            ep["pred"] = hint(phase="result", user_text=user, results=ep["results"]) if ep["traj"] \
                else hint(phase="start", user_text=user)
        ep["traj"], ep["user"] = list(traj), user or ep["user"]
        return ep["pred"]

    def predict(self, req: PredictRequest) -> dict[str, float]:
        with self.lock:                                  # sessions are shared state
            p = self._update(req)
        out = {t.split(".", 1)[-1]: v for t, v in p.p1.items()}   # "airline.get_user_details" -> "get_user_details"
        out[END] = p.p_end
        return out

    def health(self) -> dict:
        return {"model_uri": MODEL_URI, "model_file": str(self.model_file), "sessions": len(self.episodes),
                "embedding": "ok" if self.pred.embedding_ok else "unavailable", "predictor": self.pred.stats()}

    @property
    def note(self):
        return "" if self.pred.embedding_ok else "embedding server unavailable: zero embedding"
