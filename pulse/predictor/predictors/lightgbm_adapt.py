"""
lightgbm_adapt: lightgbm + pulse-v3's online adapter (pulse/pulse_v3/predictors/pulse_v3/online.py), so the
model adjusts to the agent model it is deployed with, without retraining.

  P = softmax(log P0 + W phi)    P0  = lightgbm's probabilities (the domain's tools + RESPOND)
                                 phi = lightgbm's signals (log1p) + one-hot last tool + bias
  W starts at 0, so a fresh adapter answers exactly as lightgbm does.

It follows each episode (episode_id). What a step really did is read from the NEXT hint of the same episode:
new tools in the trajectory = the tools it called, none = RESPOND. When an episode has been idle for
SESSION_TIMEOUT_S, its labelled steps go into a replay buffer and the adapter takes a few AdaGrad passes over it
(learning rate ADAPT_LR, L2, ADAPT_PASSES). Every step is predicted before the adapter learns from it.

Settings: SESSION_TIMEOUT_S (300), ADAPT_LR (0.05; 0.2 overfitted on 11 live episodes), ADAPT_L2 (0.001),
ADAPT_PASSES (3), ADAPT_BUFFER (5000 steps). The adapter lives in memory: a restart starts again from lightgbm.
"""

import math
import os
import threading
import time

import numpy as np

from pulse.predictor.action_model import NUMERIC, RESPOND
from pulse.predictor.contract import PredictRequest
from pulse.predictor.predictors.lightgbm import LightGBM

SESSION_TIMEOUT_S = float(os.getenv("SESSION_TIMEOUT_S", "300"))
LR = float(os.getenv("ADAPT_LR", "0.05"))
L2 = float(os.getenv("ADAPT_L2", "0.001"))
PASSES = int(os.getenv("ADAPT_PASSES", "3"))
BUFFER = int(os.getenv("ADAPT_BUFFER", "5000"))


def appended(prev: list, cur: list) -> list:
    """Tools added to a trajectory window since the last hint (hints carry the last 8 tools)."""
    for s in range(len(prev) + 1):
        if prev[s:] == cur[:len(prev) - s]:
            return cur[len(prev) - s:]
    return cur


class Adapter:
    """Online residual on top of a frozen model: softmax(log P0 + W phi), AdaGrad + L2 over a replay buffer."""

    def __init__(self, classes: list[str], dim: int, seed: int = 0):
        self.classes = classes
        self.W = np.zeros((len(classes), dim))
        self.G = np.full((len(classes), dim), 1e-8)
        self.buf: list = []                     # (log P0 row, phi row, target row)
        self.rng = np.random.default_rng(seed)

    def predict(self, logp0: np.ndarray, f: np.ndarray) -> np.ndarray:
        z = logp0 + self.W @ f
        q = np.exp(z - z.max())
        return q / q.sum()

    def learn(self, rows: list) -> None:
        self.buf = (self.buf + rows)[-BUFFER:]
        for _ in range(PASSES):
            idx = self.rng.permutation(len(self.buf))
            for s in range(0, len(idx), 32):
                L, X, Y = (np.stack(v) for v in zip(*[self.buf[i] for i in idx[s:s + 32]]))
                Z = L + X @ self.W.T
                P = np.exp(Z - Z.max(1, keepdims=True))
                P /= P.sum(1, keepdims=True)
                g = (P - Y).T @ X / len(X) + L2 * self.W
                self.G += g * g
                self.W -= LR * g / np.sqrt(self.G)


class LightGBMAdapt(LightGBM):
    name = "lightgbm_adapt"
    description = "LightGBM + online adapter: learns the deployed agent model's habits from finished episodes"

    def __init__(self):
        super().__init__()
        self.adapter: Adapter | None = None
        self.episodes: dict[str, dict] = {}     # episode_id -> {"t", "traj", "steps": [[logp0, phi, truth]]}
        self.learned_episodes = 0
        self.lock = threading.Lock()

    def phi(self, req: PredictRequest, classes: list[str]) -> np.ndarray:
        sig = req.signals or {}
        x = [math.log1p(max(0.0, float(sig.get(k) or 0))) for k in NUMERIC]
        traj = req.tools()
        last = np.zeros(len(classes) + 1)       # the last slot: no tool yet (START) or one the model doesn't know
        last[classes.index(traj[-1]) if traj and traj[-1] in classes else -1] = 1
        return np.concatenate([x, last, [1.0]])

    def _expire(self, now: float) -> None:
        for eid, ep in list(self.episodes.items()):
            if now - ep["t"] > SESSION_TIMEOUT_S:
                self.episodes.pop(eid)
                self._learn(ep)

    def _learn(self, ep: dict) -> None:
        rows = []
        for logp0, f, truth in ep["steps"]:
            if not truth:                       # the episode's last step: what it did is never seen
                continue
            y = np.zeros(len(self.adapter.classes))
            for a in truth:
                if a in self.adapter.classes:
                    y[self.adapter.classes.index(a)] += 1 / len(truth)
            if y.sum():
                rows.append((logp0, f, y))
        if rows:
            self.adapter.learn(rows)
            self.learned_episodes += 1

    def predict(self, req: PredictRequest) -> dict[str, float]:
        p0 = super().predict(req)
        if req.signals is None:                 # lightgbm answered with the Markov chain: nothing to adapt
            return p0
        now = time.time()
        with self.lock:                         # episodes and the adapter are shared state
            self._expire(now)
            if self.adapter is None:
                classes = sorted(p0)
                self.adapter = Adapter(classes, len(self.phi(req, classes)))
            classes = self.adapter.classes
            logp0 = np.log(np.array([p0.get(c, 0.0) for c in classes]) + 1e-6)
            f = self.phi(req, classes)
            eid = req.episode_id or "default"
            traj = req.tools()
            ep = self.episodes.setdefault(eid, {"t": now, "traj": [], "steps": []})
            if ep["steps"] and not ep["steps"][-1][2]:            # the previous step's truth, seen now
                ep["steps"][-1][2] = appended(ep["traj"], traj) or [RESPOND]
            ep["t"], ep["traj"] = now, list(traj)
            ep["steps"].append([logp0, f, None])
            q = self.adapter.predict(logp0, f)
        return dict(zip(classes, q.tolist()))

    def health(self) -> dict:
        with self.lock:
            buf = len(self.adapter.buf) if self.adapter else 0
            w = float(np.abs(self.adapter.W).sum()) if self.adapter else 0.0
            return {**super().health(), "adapter": {"open_episodes": len(self.episodes),
                                                    "learned_episodes": self.learned_episodes,
                                                    "learned_steps": buf, "weights_l1": round(w, 3),
                                                    "lr": LR, "session_timeout_s": SESSION_TIMEOUT_S}}
