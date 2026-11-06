"""
PulseV3Predictor: Pulse's predictor behind the service interface (predictors.base.NextToolPredictor).

Per hint: the session's steps so far become a traces.tau.Episode, from which the model-agnostic
features are built exactly as offline (graph ready set + fan-out, state flags, embedding of user text
+ results); the frozen V3 model gives P0, the agent model's adapter corrects it, and an optional
self-hint is mixed in with a learned weight. When a session ends the adapter learns from it.

The trained bundle is models/pulse_v3/predictor.pkl ({"v3": Semantic, "graph": ToolGraph}).

Without the embedding server it still answers, with a zero embedding (lower accuracy) and
info {"embedding": "unavailable"}.
"""

import os

import numpy as np

from pulse import compat

from ... import paths
from ...contract.hints import End, Hint
from ...contract.prediction import Prediction
from ...tools import DOMAINS, TOOL
from ...traces import tau
from ..base import BadHint, NextToolPredictor, OutOfOrder, UnknownSession
from . import graph, state
from .embed import DIM, Embedder
from .models import CLASSES, END, IDX, MASK, contexts
from .online import Adapter, phi

MODEL_PATH = os.getenv("PULSE_MODEL") or paths.MODELS / "predictor.pkl"


def load_bundle(path=None):
    """The trained bundle {"v3": Semantic, "graph": ToolGraph}, old or new module paths (pulse.compat)."""
    return compat.load(path or MODEL_PATH)


def emb_text(c):
    return f"domain {c.domain} | user: {c.user_text} | result: {c.result or ''}"


class _Session:
    def __init__(self, domain, agent_model, user_text):
        self.domain, self.agent_model = domain, agent_model
        self.first_user = user_text or ""
        self.user_text, self.agent_text = user_text or "", ""
        self.user_new = False
        self.steps = []
        self.records = []                       # (key, logP0, phi, mask, P_model, P_hint) per hint

    def episode(self):
        return tau.Episode(self.domain, self.agent_model, 0, 0, 0.0, self.first_user, self.steps)


class PulseV3Predictor(NextToolPredictor):
    name = "pulse-v3"

    def __init__(self, bundle, embedder=None):
        self.v3, self.g = bundle["v3"], bundle["graph"]
        self.embedder = embedder or Embedder()
        self.embedding_ok = True
        self.adapters, self.lams, self.lam_n = {}, {}, {}
        self.sessions = {}
        self.n_hints = 0

    @classmethod
    def load(cls, path=None, embedder=None):
        return cls(load_bundle(path), embedder=embedder)

    def _embed(self, text):
        try:
            v = self.embedder([text])[0]
            self.embedding_ok = True
            return v
        except OSError:
            self.embedding_ok = False
            return np.zeros(DIM, np.float32)

    def _ctx(self, sess, key):
        ep = sess.episode()
        ctx = next(c for c in contexts(ep, 0) if c.key == key)
        fe = graph.ready_by_key(self.g, [ep], CLASSES)[key]
        st = state.state_by_key([ep])[key]
        ctx.extra = np.concatenate([fe, st, self._embed(emb_text(ctx))])
        return ctx

    def hint(self, h: Hint) -> Prediction:
        if h.domain not in DOMAINS:
            raise BadHint(f"unknown domain {h.domain!r}")
        sess = self.sessions.get(h.session_id)
        if h.phase == "start" or sess is None:
            sess = self.sessions[h.session_id] = _Session(h.domain, h.agent_model, h.user_text)
        elif h.user_text and h.user_text != sess.user_text:
            sess.user_text, sess.user_new = h.user_text, True
        if h.agent_text:
            sess.agent_text = h.agent_text
        if h.phase == "call":
            names = [f"{h.domain}.{c.tool}" for c in h.calls]
            unknown = [n for n in names if n not in TOOL]
            if unknown or not names:
                raise BadHint(f"unknown or missing tools: {unknown or 'no calls'}")
            hint = [(f"{h.domain}.{t}" if t != END else END, p) for t, p in (h.selfhint or [])
                    if t == END or f"{h.domain}.{t}" in TOOL]
            sess.steps.append(tau.Step(names, "", sess.user_text, sess.agent_text, "", sess.user_new or not sess.steps,
                                       args=[c.args for c in h.calls], results_full=[""] * len(names),
                                       selfhint=hint or None))
            sess.user_new = False
        elif h.phase == "result":
            if not sess.steps:
                raise OutOfOrder("a result hint needs an earlier call hint")
            s = sess.steps[-1]
            s.results_full = (list(h.results) + [""] * len(s.tools))[:len(s.tools)]
            s.result = " || ".join(r[:tau.RESULT_CHARS] for r in s.results_full)
        elif h.phase != "start":
            raise BadHint(f"unknown phase {h.phase!r}")

        key = (0, h.phase, max(len(sess.steps) - 1, 0))
        ctx = self._ctx(sess, key)
        P1, P2 = (x[0] for x in self.v3.predict([ctx]))
        logP0, F, M = np.log(P1 + 1e-6)[None], phi([ctx]), MASK[ctx.domain][None]
        ad = self.adapters.setdefault(sess.agent_model, Adapter(F.shape[1]))
        P = ad.predict(logP0, F, M)[0]
        Ph = None
        if ctx.selfhint:
            Ph = np.zeros(len(CLASSES))
            for t, p in ctx.selfhint:
                Ph[IDX[t]] += p
            rest = max(0.0, 1 - Ph.sum())
            Ph = np.where(MASK[ctx.domain], Ph + rest / MASK[ctx.domain].sum(), 0)
            Ph /= Ph.sum()
            lam = self.lams.get(sess.agent_model, 0.5)
            P_out = lam * Ph + (1 - lam) * P
        else:
            P_out = P
        sess.records.append((key, logP0[0], F[0], M[0], P, Ph))
        self.n_hints += 1

        tools = lambda P: {CLASSES[j]: float(P[j]) for j in np.nonzero(P)[0] if CLASSES[j] != END}  # noqa: E731
        return Prediction(h.session_id, sess.domain, sess.agent_model, len(sess.steps), h.phase,
                          list(sess.steps[-1].tools) if h.phase == "call" else [],
                          tools(P_out), tools(P2), float(P_out[IDX[END]]),
                          info={"embedding": "ok" if self.embedding_ok else "unavailable"})

    def end(self, e: End) -> dict:
        sess = self.sessions.pop(e.session_id, None)
        if sess is None:
            raise UnknownSession(f"unknown session {e.session_id!r}")
        truth = {c.key: c for c in contexts(sess.episode(), 0)}
        rows = [(r, truth[r[0]]) for r in sess.records if r[0] in truth]
        if not rows:
            return {"session_id": e.session_id, "learned_hints": 0}
        Y = np.zeros((len(rows), len(CLASSES)))
        for i, (_, c) in enumerate(rows):
            for t in c.truth1:
                Y[i, IDX[t]] += 1 / len(c.truth1)
        L, F, M = (np.stack([r[k] for r, _ in rows]) for k in (1, 2, 3))
        self.adapters[sess.agent_model].learn(L, F, Y, M)
        lam = self.lams.get(sess.agent_model, 0.5)
        n = self.lam_n.get(sess.agent_model, 0)
        for i, (r, _) in enumerate(rows):                   # trust weight of the self-hint, decaying step
            if r[5] is None:
                continue
            pa, pb = (r[5] * Y[i]).sum(), (r[4] * Y[i]).sum()
            p = max(lam * pa + (1 - lam) * pb, 1e-9)
            lam = float(np.clip(lam + 0.2 / np.sqrt(1 + n / 25) * (pa - pb) / p, 0, 1))
            n += 1
        self.lams[sess.agent_model], self.lam_n[sess.agent_model] = lam, n
        return {"session_id": e.session_id, "learned_hints": len(rows), "self_hint_weight": round(lam, 3)}

    def stats(self) -> dict:
        return {"sessions": len(self.sessions), "hints": self.n_hints,
                "adapters": {m: len(a.buf) for m, a in self.adapters.items()}, "self_hint_weight": self.lams}
