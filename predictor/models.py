"""
Steps 3 + 4: the next-tool models.

Every model answers one question:
    "The agent has called these tools so far (the trajectory). What comes next?"
and returns a probability for every possible next tool (plus END), summing to 1.

  P_G  graph       : equal share for every tool the workflow graph allows next
  P_H  history     : how often tool B followed tool A in past traces   (looks at the last 1 tool)
  P_T  trajectory  : same counting, but looks at the last 2 tools      (falls back to P_H if unseen)
  ensemble         : P = wG*P_G + wH*P_H + wT*P_T  with wG + wH + wT = 1   (PRD "weighted model")
  lr               : scikit-learn LogisticRegression on the last tools (the PRD's "softmax model")
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"
START, END = "START", "END"
POOLED = "airline"
MODEL_NAMES = ("graph", "history", "trajectory", "ensemble", "lr")


# ----------------------------------------------------------------- helpers

def sequences_from_steps(df):
    """DataFrame of transitions -> {workflow: [[START, t1, t2, ..., END], ...]}"""
    out = defaultdict(list)
    for (wf, _), g in df.sort_values(["trace_id", "step"]).groupby(["workload", "trace_id"]):
        seq = [START] + g["to_tool"].tolist()
        out[wf].append(seq)
        out[POOLED].append(seq)
    return dict(out)


def examples(seq):
    """[START, a, b, END] -> [([], a), ([a], b), ([a, b], END)]  (trajectory so far, true next)"""
    tools = seq[1:]
    return [(tools[:i], tools[i]) for i in range(len(tools))]


def last(traj, k):
    """The k-th last tool of the trajectory, or START if the trajectory is shorter."""
    return traj[-k] if len(traj) >= k else START


def normalise(p):
    s = sum(p.values())
    if s <= 0:
        return {j: 1 / len(p) for j in p}
    return {j: v / s for j, v in p.items()}


# ----------------------------------------------------------------- one workflow

class WorkflowModel:
    """Graph + history + trajectory model for ONE workflow (W1, W2 or the pooled 'airline')."""

    def __init__(self, workflow_id, graph, alpha=0.1, backoff_k=3.0):
        self.workflow_id = workflow_id
        self.edges = {a: set(b) for a, b in graph["edges"].items()}
        # Everything that can be predicted: all graph nodes except START
        self.vocab = sorted(n for n in graph["nodes"] if n != START)
        self.alpha = alpha            # smoothing: unseen transitions get a small, non-zero chance
        self.backoff_k = backoff_k    # how much the last-2-tools counts are trusted
        self.c1 = defaultdict(Counter)   # counts: last tool          -> next tool
        self.c2 = defaultdict(Counter)   # counts: (2nd last, last)   -> next tool
        self.weights = (1 / 3, 1 / 3, 1 / 3)

    def fit(self, seqs):
        for seq in seqs:
            for traj, nxt in examples(seq):
                self.c1[last(traj, 1)][nxt] += 1
                self.c2[(last(traj, 2), last(traj, 1))][nxt] += 1
        return self

    # P_G: uniform over the tools the graph allows after the current tool
    def p_graph(self, traj):
        allowed = self.edges.get(last(traj, 1), set()) & set(self.vocab)
        if not allowed:                       # unknown tool -> no information -> uniform
            allowed = set(self.vocab)
        return {j: (1 / len(allowed) if j in allowed else 0.0) for j in self.vocab}

    # P_H: counts of (last tool -> next tool), smoothed
    def p_history(self, traj):
        c = self.c1.get(last(traj, 1), Counter())
        total = sum(c.values())
        V = len(self.vocab)
        return {j: (c[j] + self.alpha) / (total + self.alpha * V) for j in self.vocab}

    # P_T: counts of (2nd last, last -> next); trusted more when seen more often, else P_H
    def p_trajectory(self, traj):
        ph = self.p_history(traj)
        c = self.c2.get((last(traj, 2), last(traj, 1)), Counter())
        n = sum(c.values())
        if n == 0:
            return ph
        lam = n / (n + self.backoff_k)
        return {j: lam * c[j] / n + (1 - lam) * ph[j] for j in self.vocab}

    def p_ensemble(self, traj, weights=None):
        wg, wh, wt = weights or self.weights
        pg, ph, pt = self.p_graph(traj), self.p_history(traj), self.p_trajectory(traj)
        return normalise({j: wg * pg[j] + wh * ph[j] + wt * pt[j] for j in self.vocab})


# ----------------------------------------------------------------- scikit-learn model

def lr_features(workflow_id, traj):
    p1, p2 = last(traj, 1), last(traj, 2)
    return {"wf=" + workflow_id: 1, "p1=" + p1: 1, "p2=" + p2: 1,
            f"p2p1={p2}>{p1}": 1, f"wf_p1={workflow_id}>{p1}": 1,
            "step=" + str(min(len(traj), 8)): 1}


class LRModel:
    """Multinomial logistic regression = the PRD's softmax model, learned by scikit-learn."""

    def __init__(self, C=1.0):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = DictVectorizer()
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, seqs_by_wf):
        X, y = [], []
        for wf, seqs in seqs_by_wf.items():
            if wf == POOLED:
                continue
            for seq in seqs:
                for traj, nxt in examples(seq):
                    X.append(lr_features(wf, traj))
                    y.append(nxt)
        self.clf.fit(self.vec.fit_transform(X), y)
        return self

    def proba(self, workflow_id, traj, vocab):
        probs = self.clf.predict_proba(self.vec.transform([lr_features(workflow_id, traj)]))[0]
        by_tool = dict(zip(self.clf.classes_, probs))
        return normalise({j: by_tool.get(j, 0.0) + 1e-6 for j in vocab})


# ----------------------------------------------------------------- the whole predictor

class Predictor:
    """All workflows together. This is what the API (service.py) loads."""

    def __init__(self, models, lr=None):
        self.models = models          # {workflow_id: WorkflowModel}
        self.lr = lr

    @classmethod
    def load_graph(cls, workflow_id):
        return json.loads((GRAPHS_DIR / f"{workflow_id}.json").read_text())

    def model_for(self, workflow_id):
        """Unknown or missing workflow id -> the pooled airline model."""
        return self.models.get(workflow_id) or self.models[POOLED]

    def proba(self, workflow_id, trajectory, model="ensemble", signals=None):
        """{tool: probability} over every possible next tool (incl. END). Sums to 1.
        model="lgbm" needs the hint's signals and a loaded LightGBM model (predictor/data/lgbm.pkl);
        without either it falls back to the ensemble."""
        m = self.model_for(workflow_id)
        traj = list(trajectory)
        if model == "lgbm":
            lgbm = getattr(self, "lgbm", None)
            if lgbm is not None and signals is not None:
                return lgbm.proba(m.workflow_id, traj, signals, m.vocab)
            model = "ensemble"
        if model == "graph":
            return m.p_graph(traj)
        if model == "history":
            return m.p_history(traj)
        if model == "trajectory":
            return m.p_trajectory(traj)
        if model == "ensemble":
            return m.p_ensemble(traj)
        if model == "lr":
            return self.lr.proba(m.workflow_id, traj, m.vocab)
        raise ValueError(f"unknown model {model!r}; use one of {MODEL_NAMES}")

    def rank(self, workflow_id, trajectory, model="ensemble", signals=None):
        """[(tool, probability), ...] highest first."""
        p = self.proba(workflow_id, trajectory, model, signals)
        return sorted(p.items(), key=lambda kv: -kv[1])


def prewarm_set(ranked, theta):
    """FR-5: tools with probability >= theta. END is never prewarmed (it is not a function)."""
    return [t for t, p in ranked if p >= theta and t != END]


def log_loss(p_true):
    return -math.log(max(p_true, 1e-12))
