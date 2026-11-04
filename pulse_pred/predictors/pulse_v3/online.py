"""
Online adaptation to the deployed agent model (see README, "Continual learning").

Adapter: a residual on top of a frozen predictor, softmax(log P0 + W phi), where phi are the
model-agnostic per-hint features (graph ready set, state flags, embedding) plus phase and bias.
W starts at zero, so a fresh adapter reproduces the frozen predictor exactly; after each finished
episode, learn() adds its hints to a replay buffer and runs a few AdaGrad passes over it.
"""

import numpy as np

from .models import PHASES, V


def phi(ctxs):
    """Adapter features: Ctx.extra (dense model-agnostic features) + phase one-hot + bias."""
    ph = np.zeros((len(ctxs), len(PHASES)))
    ph[np.arange(len(ctxs)), [PHASES.index(c.phase) for c in ctxs]] = 1
    return np.hstack([np.stack([c.extra for c in ctxs]), ph, np.ones((len(ctxs), 1))])


class Adapter:
    """Online residual on top of a frozen predictor: softmax(log P0 + W phi), AdaGrad + L2."""

    def __init__(self, dim, lr=0.05, l2=1e-3, passes=3, seed=0):
        self.W = np.zeros((V, dim))
        self.G = np.full((V, dim), 1e-8)
        self.lr, self.l2, self.passes = lr, l2, passes
        self.rng = np.random.default_rng(seed)
        self.buf = []                                   # (log P0 row, phi row, target row, mask row)

    def predict(self, logP0, F, M):
        """logP0: (n, V) log-probabilities of the frozen predictor; F: (n, dim) phi; M: (n, V) domain mask."""
        z = np.where(M, logP0 + F @ self.W.T, -np.inf)
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        return q / q.sum(1, keepdims=True)

    def learn(self, logP0, F, Y, M, batch=32):
        """Y: (n, V) truth rows (predictors.truth_matrix) of one finished episode."""
        self.buf += list(zip(logP0, F, Y, M))
        for _ in range(self.passes):
            idx = self.rng.permutation(len(self.buf))
            for s in range(0, len(idx), batch):
                L, X, T, Mk = (np.stack(x) for x in zip(*[self.buf[i] for i in idx[s:s + batch]]))
                g = (self.predict(L, X, Mk) - T).T @ X / len(X) + self.l2 * self.W
                self.G += g * g
                self.W -= self.lr * g / np.sqrt(self.G)
