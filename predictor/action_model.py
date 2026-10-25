"""
Next-ACTION models: before every LLM step, predict what the step will do (one of the 14 tools or
RESPOND) and, for a tool, its fan-out width k (= desired replicas in the PrewarmRequest).

  gbt    main model: gradient-boosted trees (LightGBM), ~0.1 ms per prediction
  lr     second model: logistic regression on the same features (interpretable)
  ngram  internal baseline: counts of the next action after the last two tools (back-off to one)
  k      pending-ID rule: k = number of pending IDs for tools that consume them, else 1

All three classifiers see the same input (workflow, last tool, last two tools + signals) and
return probabilities over the same 15 actions that sum to 1 (FR-4).
Trained by predictor/train_action.py, saved to predictor/data/action_model.pkl.
"""

from collections import Counter, defaultdict

import numpy as np

from signals import SIGNAL_KEYS

START, RESPOND = "START", "RESPOND"
CATS = ["wf", "last1", "last2"]
NUMERIC = SIGNAL_KEYS
K_CAP = 5                       # the CRD allows desiredReplicas up to 5


class Encoder:
    """Hint -> one numeric row: categorical codes (unknown = -1) + the signals in a fixed order."""

    def __init__(self, rows):
        src = {"wf": "workload", "last1": "last1", "last2": "last2"}
        self.vocab = {c: sorted(set(rows[src[c]].astype(str))) for c in CATS}
        self.index = {c: {v: i for i, v in enumerate(vs)} for c, vs in self.vocab.items()}
        self.columns = CATS + NUMERIC

    def row(self, wf, last1, last2, signals):
        cats = [self.index["wf"].get(str(wf), -1), self.index["last1"].get(last1, -1),
                self.index["last2"].get(last2, -1)]
        return cats + [float((signals or {}).get(k, 0)) for k in NUMERIC]

    def frame(self, df):
        return np.array([self.row(r.workload, r.last1, r.last2, {k: getattr(r, k) for k in NUMERIC})
                         for r in df.itertuples()], dtype=np.float64)


class GBTModel:
    def __init__(self, booster, classes):
        self.booster, self.classes = booster, list(classes)

    def proba_x(self, X):
        return self.booster.predict(X, num_threads=1)


class LRModel:
    """One-hot (sequence, workflow) + standardised log-scaled numbers -> multinomial logistic regression."""

    def __init__(self, C=1.0):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
        ncat = len(CATS)
        num = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), list(range(ncat))),
                                  ("num", num, list(range(ncat, ncat + len(NUMERIC))))])
        self.pipe = make_pipeline(prep, LogisticRegression(C=C, max_iter=3000))
        self.classes = None

    def fit(self, X, y):
        self.pipe.fit(X, y)
        self.classes = list(self.pipe.classes_)
        return self

    def proba_x(self, X):
        return self.pipe.predict_proba(X)

    def top_weights(self, encoder, cls, n=8):
        """Largest positive weights for one action: what pushes the model towards it (readable names)."""
        prep, lr = self.pipe[0], self.pipe[-1]
        names = []
        for col, cats in zip(CATS, prep.named_transformers_["cat"].categories_):
            vocab = encoder.vocab[col]
            names += [f"{col}={vocab[int(c)] if 0 <= int(c) < len(vocab) else '?'}" for c in cats]
        names += list(NUMERIC)
        w = lr.coef_[self.classes.index(cls)]
        order = np.argsort(-w)[:n]
        return [(names[i], round(float(w[i]), 2)) for i in order]


class NGramModel:
    """The existing counting idea on the new target: P(action | last two tools), backing off to
    P(action | last tool) and then to the overall action frequencies when evidence is thin."""

    def __init__(self, alpha=0.1, k=3.0):
        self.alpha, self.k = alpha, k

    def fit(self, df):
        self.classes = sorted(df.target.unique())
        self.prior = Counter(df.target)
        self.c1, self.c2 = defaultdict(Counter), defaultdict(Counter)
        for r in df.itertuples():
            self.c1[r.last1][r.target] += 1
            self.c2[(r.last2, r.last1)][r.target] += 1
        return self

    def _dist(self, c, fallback):
        n = sum(c.values())
        if n == 0:
            return fallback
        lam = n / (n + self.k)
        return {a: lam * c[a] / n + (1 - lam) * fallback[a] for a in self.classes}

    def proba_one(self, last1, last2):
        tot = sum(self.prior.values())
        V = len(self.classes)
        p0 = {a: (self.prior[a] + self.alpha) / (tot + self.alpha * V) for a in self.classes}
        p1 = self._dist(self.c1.get(last1, Counter()), p0)
        return self._dist(self.c2.get((last2, last1), Counter()), p1)


class FanoutRule:
    """Pending-ID rule: a tool that consumes pending IDs gets k = pending IDs (1..K_CAP); others k = 1.
    Which tools consume IDs is learned from training data (share of calls using a pending ID)."""

    def __init__(self, id_tools, cap=K_CAP):
        self.id_tools, self.cap = set(id_tools), cap

    @classmethod
    def fit(cls, df, min_share=0.5):
        tools = df[df.target != RESPOND]
        share = tools.groupby("target").used_pending.mean()
        return cls(share[share >= min_share].index)

    def k(self, tool, signals):
        if tool not in self.id_tools:
            return 1
        return int(min(self.cap, max(1, (signals or {}).get("pending_ids", 0))))


class ActionPredictor:
    """Everything the API needs, in one pickle."""

    def __init__(self, encoder, models, fanout, feature_importance=None):
        self.encoder, self.models, self.fanout = encoder, models, fanout
        self.feature_importance = feature_importance or {}

    @property
    def names(self):
        return sorted(self.models)

    def proba(self, model, workflow_id, trajectory, signals):
        traj = list(trajectory)
        last1 = traj[-1] if traj else START
        last2 = traj[-2] if len(traj) >= 2 else START
        m = self.models[model]
        if isinstance(m, NGramModel):
            return m.proba_one(last1, last2)
        x = np.array([self.encoder.row(workflow_id, last1, last2, signals)])
        p = m.proba_x(x)[0]
        return {a: float(v) for a, v in zip(m.classes, p)}
