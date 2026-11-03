"""
Offline: how many tool calls each prewarm policy would have prewarmed in time, on the held-out airline steps
(predictor/data/action_rows.csv, test split), with the trained predictor/data/action_model.pkl.

A tool call counts as prewarmed if the call-step hint prewarmed its tool (PREWARM_POLICY), or, with lookahead,
a reply-step hint since the previous tool call did (prewarmed during the customer's turn, ~seconds of lead).

    python predictor/eval_policies.py        # from the repo root

Same settings as service.py: theta / topk (TOP_K, P_MIN), LOOKAHEAD (THETA_NEXT, LOOKAHEAD_MIN_RESPOND).
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import NO_PREWARM, prewarm_select  # noqa: E402
from train_action import RESPOND, predict_rows  # noqa: E402

pred = pickle.load(open(HERE / "data" / "action_model.pkl", "rb"))
rows = pd.read_csv(HERE / "data" / "action_rows.csv")
test = rows[rows.split == "test"].reset_index(drop=True)
P = predict_rows(pred, "gbt", test)
look = getattr(pred, "lookahead", None)
N = predict_rows(look, "gbt", test) if look is not None else None
if look is None:
    print("(this action_model.pkl has no lookahead model: retrain with train_action.py)")

POLICIES = [("theta 0.3 (current)", dict(policy="theta", theta=0.3), None),
            ("topk 2, p_min 0.1", dict(policy="topk", top_k=2, p_min=0.1), None),
            ("topk 3, p_min 0.05", dict(policy="topk", top_k=3, p_min=0.05), None),
            ("theta 0.3 + lookahead 0.4", dict(policy="theta", theta=0.3), 0.4),
            ("theta 0.3 + lookahead 0.3", dict(policy="theta", theta=0.3), 0.3),
            ("topk 2, p_min 0.1 + lookahead 0.4", dict(policy="topk", top_k=2, p_min=0.1), 0.4),
            ("topk 2, p_min 0.1 + lookahead 0.3", dict(policy="topk", top_k=2, p_min=0.1), 0.3)]
print(f"{len(test)} held-out LLM steps, {int((test.target != RESPOND).sum())} tool steps\n")
print(f"{'policy':36} {'tool calls prewarmed':>21} {'after a customer turn':>22} {'prewarms/step':>14}")
for name, kw, theta_next in POLICIES:
    if theta_next is not None and N is None:
        continue
    hit = n = hit_turn = n_turn = warmed = 0
    ahead, prev = set(), None
    for i, r in test.iterrows():
        if r.trace != prev:                    # a new conversation
            ahead, prev, after_turn = set(), r.trace, False
        ranked = sorted(P[i].items(), key=lambda kv: -kv[1])
        warm = set(prewarm_select(ranked, **kw))
        if theta_next is not None and P[i].get(RESPOND, 0) >= 0.5:
            nxt = {t for t, p in N[i].items() if p >= theta_next and t not in NO_PREWARM} - warm
            warm |= nxt
            ahead |= nxt
        warmed += len(warm)
        if r.target == RESPOND:
            after_turn = True                   # the customer answers before the next step
            continue
        ok = r.target in warm or r.target in ahead
        hit += ok
        n += 1
        if after_turn:
            hit_turn += ok
            n_turn += 1
        ahead, after_turn = set(), False        # lookahead covers the first call after the customer's turn
    print(f"{name:36} {hit / n:>21.3f} {hit_turn / max(1, n_turn):>22.3f} {warmed / len(test):>14.2f}")
