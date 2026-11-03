"""
Next-step predictors. A step is a set of tools called concurrently. For a batch of hint contexts,
each predictor returns two probability matrices over CLASSES (all tools + END):
  P1[r, x]  share of the next step's calls that go to tool x   (t+1)
  P2[r, x]  the same for the step after it                    (t+2)
Rows are masked to the domain's tools (FR-3) and sum to 1 (FR-4). A step with k tools counts each
member with weight 1/k when training, and "hit" means a top-ranked tool is in the true step.

  semantic  intent-aware text model: TF-IDF of the user's message, the agent's message, the agent's
            reasoning with the call and the last results, plus similarity to tool descriptions.
            Never sees tool-sequence counts. (Pulse's text model.)
  pulse     semantic + the agent's self-declared next tools, combined and calibrated. Needs live
            traces with self-hints; tau-bench's historical traces have none.
  selfhint  the agent's self-declared next tools only.
  markov    counts of (last two steps) -> next step, back-off to the last step: the other group's
            approach, kept as a baseline.
  oracle    knows the true next steps: upper bound for the simulator.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from ...tools import CLASSES, DOMAINS, END, IDX, MASK, TOOL, TOOL_NAMES, V, domain_tools  # noqa: F401 (re-exported)

PHASES = ("start", "call", "result")
END_STEP = (END,)


@dataclass
class Ctx:
    """What the predictor sees at one hint. truth1/truth2 are read only by the oracle and metrics."""
    key: tuple             # (episode index, phase, step index)
    domain: str
    user_text: str
    agent_text: str
    reasoning: str         # text the agent wrote with the current step's calls ('' at start)
    result: str | None     # the current step's results (None until they return)
    steps: list            # steps so far (tuples of tools), including the current one
    selfhint: list | None  # [(tool, confidence)] declared with the current step (live traces only)
    phase: str
    truth1: tuple = END_STEP
    truth2: tuple = END_STEP
    extra: object = None   # optional dense features (e.g. graph ready-set state, see graph.ready_by_key)


def contexts(ep, k):
    """Hints of one episode: at start (first user message), at each step's calls, at its results."""
    seq = [tuple(s.tools) for s in ep.steps] + [END_STEP, END_STEP]
    out = [Ctx((k, "start", 0), ep.domain, ep.first_user_text, "", "", None, [], None, "start", seq[0], seq[1])]
    for i, s in enumerate(ep.steps):
        hint = getattr(s, "selfhint", None)
        for phase, result in (("call", None), ("result", s.result)):
            out.append(Ctx((k, phase, i), ep.domain, s.user_text, s.agent_text, s.reasoning, result,
                           seq[:i + 1], hint, phase, seq[i + 1], seq[i + 2]))
    return out


def all_contexts(eps):
    return [c for k, ep in enumerate(eps) for c in contexts(ep, k)]


def truth_matrix(ctxs, horizon=1):
    """Y[r, x] = 1/k for each of the k tools in the true step."""
    Y = np.zeros((len(ctxs), V))
    for r, c in enumerate(ctxs):
        step = c.truth1 if horizon == 1 else c.truth2
        for t in step:
            Y[r, IDX[t]] += 1 / len(step)
    return Y


def expand(Y):
    """Truth matrix -> (row, label, weight) training triples, one per tool of the true step."""
    rows, cols = np.nonzero(Y)
    return rows, cols, Y[rows, cols]


def apply_mask(P, ctxs):
    M = np.stack([MASK[c.domain] for c in ctxs])
    P = np.where(M, P, 0.0)
    s = P.sum(1, keepdims=True)
    empty = s[:, 0] <= 0
    P[empty] = M[empty] / M[empty].sum(1, keepdims=True)
    s[empty] = 1.0
    return P / s


def temper(P, T):
    """Temperature scaling on probabilities (same as dividing the logits by T)."""
    Q = np.power(np.clip(P, 1e-12, 1.0), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def set_nll(P, Y):
    """-log P(the next call goes to a tool of the true step)."""
    return -np.mean(np.log(np.clip((P * (Y > 0)).sum(1), 1e-12, 1)))


def fit_temperature(P, Y):
    return minimize_scalar(lambda T: set_nll(temper(P, T), Y), bounds=(0.2, 5.0), method="bounded").x


def _onehot_phase(ctxs):
    Z = np.zeros((len(ctxs), len(PHASES)))
    Z[np.arange(len(ctxs)), [PHASES.index(c.phase) for c in ctxs]] = 1
    return Z


def selfhint_matrix(ctxs):
    H = np.zeros((len(ctxs), V))
    for r, c in enumerate(ctxs):
        for tool, conf in c.selfhint or []:
            H[r, IDX[tool]] += conf
    return H


class Predictor:
    name = "base"

    def fit(self, train_ctxs, val_ctxs=None):
        return self

    def predict(self, ctxs):
        raise NotImplementedError


TEXT_FIELDS = ("user", "agent", "reasoning", "result")


class Semantic(Predictor):
    """fields: which text the model may read. "user" (the user's messages) and "result" (tool results,
    i.e. environment state) come from outside the agent; "agent" (messages to the user) and
    "reasoning" (text written with a call) are the agent model's own words and carry its style.
    use_extra: also use Ctx.extra (dense environment-state features such as the graph's ready set)."""
    name = "semantic"

    def __init__(self, C=4.0, sim_weight=3.0, horizons=(1, 2), fields=TEXT_FIELDS, use_extra=False):
        self.C, self.sim_weight, self.horizons = C, sim_weight, horizons
        self.fields, self.use_extra = tuple(fields), use_extra
        self.T = [1.0, 1.0]

    def text(self, c):
        parts = [f"domain {c.domain}"]
        if "user" in self.fields:
            parts.append("user: " + c.user_text)
        if "agent" in self.fields and c.agent_text:
            parts.append("agent: " + c.agent_text)
        if "reasoning" in self.fields and c.reasoning:
            parts.append("reasoning: " + c.reasoning)
        if "result" in self.fields and c.result:
            parts.append("result: " + c.result)
        return " | ".join(parts)

    def _X(self, ctxs):
        T = self.vec.transform([self.text(c) for c in ctxs])
        S = (T @ self.tool_docs.T).toarray() * self.sim_weight     # zero-shot: similarity to each tool's description
        blocks = [T, sparse.csr_matrix(S), sparse.csr_matrix(_onehot_phase(ctxs))]
        if self.use_extra:
            blocks.append(sparse.csr_matrix(np.stack([c.extra for c in ctxs])))
        return sparse.hstack(blocks).tocsr()

    def fit(self, train_ctxs, val_ctxs=None):
        docs = [f"{t.name.split('.')[1].replace('_', ' ')} {t.desc}" for t in TOOL.values()]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        self.vec.fit([self.text(c) for c in train_ctxs] + docs)
        self.tool_docs = normalize(self.vec.transform(docs))
        X = self._X(train_ctxs)
        self.lr = []
        for h in self.horizons:
            rows, y, w = expand(truth_matrix(train_ctxs, h))
            self.lr.append(LogisticRegression(C=self.C, max_iter=3000).fit(X[rows], y, sample_weight=w))
        if val_ctxs:
            self.T = [fit_temperature(P, truth_matrix(val_ctxs, h)) for P, h in zip(self.predict(val_ctxs), self.horizons)]
        return self

    def _proba(self, lr, X, ctxs, T):
        P = np.zeros((X.shape[0], V))
        P[:, lr.classes_] = lr.predict_proba(X)
        return apply_mask(temper(apply_mask(P, ctxs), T), ctxs)

    def predict(self, ctxs):
        X = self._X(ctxs)
        P1 = self._proba(self.lr[0], X, ctxs, self.T[0])
        return P1, (self._proba(self.lr[1], X, ctxs, self.T[1]) if len(self.lr) > 1 else None)


class SelfHint(Predictor):
    name = "selfhint"

    def predict(self, ctxs):
        H = selfhint_matrix(ctxs)
        M = np.stack([MASK[c.domain] for c in ctxs]).astype(float)
        rest = np.clip(1 - H.sum(1, keepdims=True), 0.0, 1.0)      # unclaimed mass spread over the domain's tools
        P1 = apply_mask(H + rest * M / M.sum(1, keepdims=True), ctxs)
        return P1, apply_mask(M, ctxs)


class Pulse(Predictor):
    """Semantic + self-hint, combined by a small logistic model and temperature-calibrated.

    The semantic model is trained on the train split. The combiner must learn how far to trust it on
    hints it has NOT seen, so it is trained on half of the validation episodes and temperature-
    calibrated on the other half (split by episode, never by hint).
    """
    name = "pulse"

    def __init__(self, C=1.0):
        self.C = C
        self.T = 1.0

    def _Z(self, Psem, ctxs):
        return np.hstack([np.log(Psem + 1e-6), selfhint_matrix(ctxs), _onehot_phase(ctxs)])

    def fit(self, train_ctxs, val_ctxs=None):
        if not val_ctxs:
            raise ValueError("Pulse needs validation hints to train its combiner")
        comb_ctxs = [c for c in val_ctxs if c.key[0] % 2 == 0]
        cal_ctxs = [c for c in val_ctxs if c.key[0] % 2 == 1]
        self.sem = Semantic().fit(train_ctxs, cal_ctxs)
        rows, y, w = expand(truth_matrix(comb_ctxs, 1))
        Z = self._Z(self.sem.predict(comb_ctxs)[0], comb_ctxs)
        self.comb = LogisticRegression(C=self.C, max_iter=3000).fit(Z[rows], y, sample_weight=w)
        self.T = fit_temperature(self._combine(cal_ctxs, self.sem.predict(cal_ctxs)[0], 1.0), truth_matrix(cal_ctxs, 1))
        return self

    def _combine(self, ctxs, Psem1, T):
        P = np.zeros((len(ctxs), V))
        P[:, self.comb.classes_] = self.comb.predict_proba(self._Z(Psem1, ctxs))
        return apply_mask(temper(apply_mask(P, ctxs), T), ctxs)

    def predict(self, ctxs):
        Psem1, Psem2 = self.sem.predict(ctxs)
        return self._combine(ctxs, Psem1, self.T), Psem2


class Markov(Predictor):
    """Baseline: P(next step's tools | last 2 steps) with back-off to the last step, add-alpha smoothing."""
    name = "markov"

    def __init__(self, alpha=0.1, backoff_k=3.0):
        self.alpha, self.k = alpha, backoff_k

    def fit(self, train_ctxs, val_ctxs=None):
        self.c1, self.c2 = {}, {}
        Y = truth_matrix(train_ctxs, 1)
        for r, c in enumerate(train_ctxs):
            if c.phase == "result":           # one example per transition
                continue
            h = [("START",), ("START",)] + c.steps
            for key, store in (((c.domain, h[-1]), self.c1), ((c.domain, h[-2], h[-1]), self.c2)):
                store.setdefault(key, np.zeros(V))
                store[key] += Y[r]
        return self

    def _p(self, domain, steps):
        h = [("START",), ("START",)] + list(steps)
        n1 = self.c1.get((domain, h[-1]), np.zeros(V))
        p1 = (n1 + self.alpha * MASK[domain]) / (n1.sum() + self.alpha * MASK[domain].sum())
        n2 = self.c2.get((domain, h[-2], h[-1]))
        if n2 is None:
            return p1
        lam = n2.sum() / (n2.sum() + self.k)
        return lam * n2 / n2.sum() + (1 - lam) * p1

    def predict(self, ctxs):
        P1, P2 = np.zeros((len(ctxs), V)), np.zeros((len(ctxs), V))
        for r, c in enumerate(ctxs):
            p1 = self._p(c.domain, c.steps)
            P1[r] = p1
            # two steps ahead: chain through the likely next tools, each treated as a one-tool step
            for j in np.nonzero(p1 > 1e-3)[0]:
                P2[r] += p1[j] * (np.eye(V)[IDX[END]] if CLASSES[j] == END else self._p(c.domain, c.steps + [(CLASSES[j],)]))
        return apply_mask(P1, ctxs), apply_mask(P2, ctxs)


class Oracle(Predictor):
    name = "oracle"

    def predict(self, ctxs):
        return truth_matrix(ctxs, 1), truth_matrix(ctxs, 2)


class MultiTool(Semantic):
    """Per-tool probabilities instead of one softmax: P[r, x] = P(tool x is called in the next step),
    one binary logistic regression per tool on the same features as Semantic. A parallel step's tools
    can all score high, and the scores are what the warming policy needs (expected calls per tool).
    Rows are masked to the domain but NOT normalised (they need not sum to 1)."""
    name = "multitool"

    def fit(self, train_ctxs, val_ctxs=None):
        docs = [f"{t.name.split('.')[1].replace('_', ' ')} {t.desc}" for t in TOOL.values()]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        self.vec.fit([self.text(c) for c in train_ctxs] + docs)
        self.tool_docs = normalize(self.vec.transform(docs))
        X = self._X(train_ctxs)
        Y = truth_matrix(train_ctxs, 1) > 0
        self.clf = {}
        for j in range(V):
            if Y[:, j].any() and not Y[:, j].all():
                self.clf[j] = LogisticRegression(C=self.C, max_iter=1000).fit(X, Y[:, j])
        return self

    def predict(self, ctxs):
        X = self._X(ctxs)
        P = np.zeros((len(ctxs), V))
        for j, clf in self.clf.items():
            P[:, j] = clf.predict_proba(X)[:, 1]
        M = np.stack([MASK[c.domain] for c in ctxs])
        return np.where(M, P, 0.0), None


# ----------------------------------------------------------------- offline metrics

def warm_recall(P, Y, ks=(1, 2, 3)):
    """If the top-k tools (END excluded: it is not a pod) were warm, what share of the next step's tool
    calls would find a warm pod? Each next step weighs 1, split evenly over its calls (Y from
    truth_matrix); hints whose next step is END are skipped."""
    S = P.copy()
    S[:, IDX[END]] = -np.inf
    order = np.argsort(-S, 1)
    T = Y.copy()
    T[:, IDX[END]] = 0
    keep = T.sum(1) > 0
    out = {}
    for k in ks:
        warm = np.zeros_like(T, dtype=bool)
        np.put_along_axis(warm, order[:, :k], True, 1)
        out[f"warm@{k}"] = float((T[keep] * warm[keep]).sum() / T[keep].sum())
    return out

def metrics(P, Y, bins=10):
    """hit@k: a top-k tool is in the true next step. nll: -log P(true step). ece: top-1 calibration."""
    order = np.argsort(-P, 1)
    in_step = Y > 0
    top_in = np.take_along_axis(in_step, order[:, :3], 1)
    conf = P.max(1)
    hit1 = top_in[:, 0]
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit1[m].mean() - conf[m].mean())
    return {"hit@1": hit1.mean(), "hit@3": top_in.any(1).mean(), "nll": set_nll(P, Y), "ece": ece, "n": len(Y)}
