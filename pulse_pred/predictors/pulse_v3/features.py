"""
pulse-v3's per-hint features for training and evaluation, the same as the service builds live
(predictor.py): the tool graph's ready set, environment-state flags and a sentence embedding of the user
text + results, next to the TF-IDF text fields (TEXT) the model reads itself.
"""

import numpy as np

from . import graph, state
from .models import CLASSES, all_contexts, metrics, truth_matrix, warm_recall
from .predictor import emb_text

TEXT = ("user", "result")
BLOCKS = ("graph", "state", "emb")


def contexts(g, eps, embedder):
    """Every hint of `eps` with its features in Ctx.extra (graph ready set + state + embedding)."""
    ctx = all_contexts(eps)
    fe, st = graph.ready_by_key(g, eps, CLASSES), state.state_by_key(eps)
    E = embedder([emb_text(c) for c in ctx])
    for r, c in enumerate(ctx):
        c.extra = np.concatenate([fe[c.key], st[c.key], E[r]])
    return ctx


def evaluate(model, ctx) -> dict:
    """Held-out quality of the next-step prediction: hit@1, hit@3 and warm@k (the share of the next step's
    tool calls that would find a warm pod if the top-k tools were prewarmed)."""
    P = model.predict(ctx)[0]
    Y = truth_matrix(ctx, 1)
    m = metrics(P, Y)
    return {k: round(float(v), 4) for k, v in {"hit@1": m["hit@1"], "hit@3": m["hit@3"], **warm_recall(P, Y),
                                                 "hints": len(ctx)}.items()}
