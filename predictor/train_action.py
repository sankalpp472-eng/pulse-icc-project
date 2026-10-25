"""
Action model, step 2: train and evaluate the next-action predictor.

  Target 1  next action = one of the 14 tools or RESPOND (one prediction per LLM step)
  Target 2  fan-out k for the predicted tool (pending-ID rule) -> desiredReplicas in the CRD
  Models    gbt (LightGBM gradient-boosted trees, main) · lr (logistic regression, interpretable)
            · ngram (counting baseline)

    python predictor/train_action.py                  # from the repo root: train on the 40 training tasks,
                                                      # score on the 10 held-out tasks, save the model
    python predictor/train_action.py --cv 5           # also 5-fold cross-validation over all 50 tasks

Reads  predictor/data/action_rows.csv (build_action_data.py)
Writes predictor/data/action_model.pkl, predictor/results/action_eval_<time>.json
"""

import argparse
import json
import math
import pickle
import random
import statistics as st
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from action_model import (CATS, NUMERIC, RESPOND, ActionPredictor, Encoder, FanoutRule, GBTModel, LRModel,
                          NGramModel)

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
NO_WARM = {RESPOND, "END"}
GBT_PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
                  cat_smooth=10, verbose=-1, seed=252, num_threads=2)


# ----------------------------------------------------------------- training

def fit_all(train, seed=252):
    enc = Encoder(train)
    X = enc.frame(train)
    classes = sorted(train.target.unique())
    y = train.target.map({c: i for i, c in enumerate(classes)}).to_numpy()

    # GBT: early stopping on held-out TRAINING tasks, then refit on all training tasks
    tasks = sorted(train.task_id.unique())
    random.Random(seed).shuffle(tasks)
    val = train.task_id.isin(set(tasks[:max(1, round(len(tasks) * 0.2))])).to_numpy()
    params = {**GBT_PARAMS, "num_class": len(classes)}
    cat_idx = list(range(len(CATS)))
    probe = lgb.train(params, lgb.Dataset(X[~val], y[~val], categorical_feature=cat_idx), 1000,
                      valid_sets=[lgb.Dataset(X[val], y[val], categorical_feature=cat_idx)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    booster = lgb.train(params, lgb.Dataset(X, y, categorical_feature=cat_idx), max(20, probe.best_iteration))
    models = {"gbt": GBTModel(booster, classes),
              "lr": LRModel().fit(X, train.target.to_numpy()),
              "ngram": NGramModel().fit(train)}
    importance = dict(zip(enc.columns, (booster.feature_importance("gain")).round(1).tolist()))
    return ActionPredictor(enc, models, FanoutRule.fit(train), importance)


# ----------------------------------------------------------------- evaluation

def predict_rows(pred, name, df):
    m = pred.models[name]
    if isinstance(m, NGramModel):
        return [m.proba_one(r.last1, r.last2) for r in df.itertuples()]
    P = m.proba_x(pred.encoder.frame(df))
    return [dict(zip(m.classes, row)) for row in P]


def score(pred, name, df, thetas):
    probs = predict_rows(pred, name, df)
    t = df.target.tolist()
    n = len(t)
    top1 = top3 = 0
    ll = 0.0
    bins = [[0, 0.0, 0] for _ in range(10)]
    tool_n = tool_top1 = resp_pred = resp_hit = 0
    for p, y in zip(probs, t):
        ranked = sorted(p, key=p.get, reverse=True)
        top1 += ranked[0] == y
        top3 += y in ranked[:3]
        ll += -math.log(max(p.get(y, 0.0), 1e-12))
        b = bins[min(9, int(p[ranked[0]] * 10))]
        b[0] += 1
        b[1] += p[ranked[0]]
        b[2] += ranked[0] == y
        if y != RESPOND:
            tool_n += 1
            tool_top1 += ranked[0] == y
        if ranked[0] == RESPOND:
            resp_pred += 1
            resp_hit += y == RESPOND
    n_resp = t.count(RESPOND)
    out = {"steps": n, "top1": top1 / n, "top3": top3 / n, "log_loss": ll / n,
           "ece": sum(b[0] / n * abs(b[2] / b[0] - b[1] / b[0]) for b in bins if b[0]),
           "tool_steps_top1": tool_top1 / tool_n,
           "respond_precision": resp_hit / max(1, resp_pred), "respond_recall": resp_hit / max(1, n_resp)}
    for th in thetas:
        hits = warmed = wasted = quiet = 0
        for p, y in zip(probs, t):
            warm = [a for a, v in p.items() if v >= th and a not in NO_WARM]
            warmed += len(warm)
            wasted += sum(a != y for a in warm)
            if y == RESPOND:
                quiet += not warm
            else:
                hits += y in warm
        out[f"hit@{th}"] = hits / tool_n                      # tool steps whose tool was prewarmed
        out[f"wasted@{th}"] = wasted / n                      # prewarmed but not the next action, per step
        out[f"quiet_on_respond@{th}"] = quiet / max(1, n_resp)  # RESPOND steps with nothing prewarmed
    return out, probs


def score_fanout(pred, df):
    tools = df[df.target != RESPOND]
    k_hat = np.array([pred.fanout.k(r.target, {"pending_ids": r.pending_ids}) for r in tools.itertuples()])
    k_true = np.minimum(tools.k_true.to_numpy(), pred.fanout.cap)
    fan = tools.k_true.to_numpy() > 1
    return {"id_tools": sorted(pred.fanout.id_tools),
            "rule_exact": float((k_hat == k_true).mean()), "k1_exact": float((k_true == 1).mean()),
            "rule_mae": float(np.abs(k_hat - k_true).mean()), "k1_mae": float(np.abs(1 - k_true).mean()),
            "fanout_steps": int(fan.sum()),
            "rule_exact_on_fanout": float((k_hat[fan] == k_true[fan]).mean()) if fan.any() else None,
            "extra_replicas_per_tool_step": float(np.clip(k_hat - k_true, 0, None).mean()),
            "missing_replicas_per_tool_step": float(np.clip(k_true - k_hat, 0, None).mean())}


def speed(pred, name, df, n=500):
    rows = list(df.itertuples())[:100]
    sig = lambda r: {k: getattr(r, k) for k in NUMERIC}
    pred.proba(name, rows[0].workload, [rows[0].last2, rows[0].last1], sig(rows[0]))
    ts = []
    for i in range(n):
        r = rows[i % len(rows)]
        traj = [x for x in (r.last2, r.last1) if x != "START"]
        s = sig(r)
        t0 = time.perf_counter()
        pred.proba(name, r.workload, traj, s)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50_ms": ts[n // 2], "p95_ms": ts[int(n * 0.95)]}


MODEL_ORDER = ["ngram", "lr", "gbt"]
LABEL = {"ngram": "n-gram baseline", "lr": "Logistic regression", "gbt": "LightGBM GBT (main)"}


def table(res, thetas, speeds=None):
    lines = ["| Model | Top-1 | Top-3 | Log-loss | ECE | Tool-step top-1 | RESPOND prec / recall |"
             + (" p50 / p95 |" if speeds else ""),
             "|---|---|---|---|---|---|---|" + ("---|" if speeds else "")]
    for m in MODEL_ORDER:
        if m not in res:
            continue
        r = res[m]
        sp = f" {speeds[m]['p50_ms']:.3f} / {speeds[m]['p95_ms']:.3f} ms |" if speeds else ""
        lines.append(f"| {LABEL[m]} | {r['top1']:.3f} | {r['top3']:.3f} | {r['log_loss']:.3f} | {r['ece']:.3f} "
                     f"| {r['tool_steps_top1']:.3f} | {r['respond_precision']:.2f} / {r['respond_recall']:.2f} |{sp}")
    lines += ["", "| Model | " + " | ".join(f"θ={t}: hit / wasted / quiet-on-RESPOND" for t in thetas) + " |",
              "|---|" + "---|" * len(thetas)]
    for m in MODEL_ORDER:
        if m in res:
            r = res[m]
            lines.append(f"| {LABEL[m]} | " + " | ".join(
                f"{r[f'hit@{t}']:.2f} / {r[f'wasted@{t}']:.2f} / {r[f'quiet_on_respond@{t}']:.2f}" for t in thetas) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    ap.add_argument("--cv", type=int, default=0, help="also run K-fold cross-validation by task over all tasks")
    ap.add_argument("--seed", type=int, default=252)
    a = ap.parse_args()

    rows = pd.read_csv(DATA / "action_rows.csv")
    train, test = rows[rows.split == "train"], rows[rows.split == "test"]
    pred = fit_all(train, a.seed)
    res = {m: score(pred, m, test, a.thetas)[0] for m in MODEL_ORDER}
    speeds = {m: speed(pred, m, test) for m in MODEL_ORDER}
    fan = score_fanout(pred, test)

    out_txt = [f"Held-out test: {test.task_id.nunique()} unseen tasks, {len(test)} LLM steps "
               f"({(test.target == RESPOND).mean():.0%} RESPOND). Trained on {train.task_id.nunique()} tasks, {len(train)} steps.",
               "", table(res, a.thetas, speeds), "",
               f"Fan-out k (pending-ID rule; ID tools learned from training: {', '.join(fan['id_tools'])}), on tool steps:",
               f"  exact k: rule {fan['rule_exact']:.2f} vs always-1 {fan['k1_exact']:.2f} · "
               f"mean error: rule {fan['rule_mae']:.2f} vs always-1 {fan['k1_mae']:.2f} · "
               f"steps with a real fan-out (k>1): {fan['fanout_steps']}, rule exact there {fan['rule_exact_on_fanout']:.2f}",
               f"  replicas: {fan['extra_replicas_per_tool_step']:.2f} too many and "
               f"{fan['missing_replicas_per_tool_step']:.2f} too few per tool step",
               "", "GBT feature importance (gain, top 10): " + ", ".join(
                   f"{k}" for k, _ in sorted(pred.feature_importance.items(), key=lambda kv: -kv[1])[:10]),
               "LR: what pushes towards RESPOND: " + ", ".join(f"{n} {w:+}" for n, w in pred.models['lr'].top_weights(pred.encoder, RESPOND, 5))]

    cv_out = None
    if a.cv:
        tasks = sorted(rows.task_id.unique())
        random.Random(a.seed).shuffle(tasks)
        folds = [set(tasks[i::a.cv]) for i in range(a.cv)]
        per = {m: [] for m in MODEL_ORDER}
        fans = []
        for f in folds:
            tr, te = rows[~rows.task_id.isin(f)], rows[rows.task_id.isin(f)]
            p = fit_all(tr, a.seed)
            for m in MODEL_ORDER:
                per[m].append(score(p, m, te, a.thetas)[0])
            fans.append(score_fanout(p, te))
        cv_out = {m: {k: (st.mean(x[k] for x in per[m]), st.pstdev([x[k] for x in per[m]]))
                      for k in per[m][0] if k != "steps"} for m in MODEL_ORDER}
        mean_only = {m: {k: v[0] for k, v in cv_out[m].items()} for m in MODEL_ORDER}
        out_txt += ["", f"{a.cv}-fold cross-validation by task (all {len(tasks)} tasks, mean over folds; "
                        f"top-1 std: " + ", ".join(f"{m} ±{cv_out[m]['top1'][1]:.3f}" for m in MODEL_ORDER) + ")",
                    "", table(mean_only, a.thetas), "",
                    f"Fan-out over folds: rule exact {st.mean(x['rule_exact'] for x in fans):.2f} vs always-1 "
                    f"{st.mean(x['k1_exact'] for x in fans):.2f}; mean error {st.mean(x['rule_mae'] for x in fans):.2f} vs "
                    f"{st.mean(x['k1_mae'] for x in fans):.2f}"]

    print("\n".join(out_txt))
    with open(DATA / "action_model.pkl", "wb") as f:
        pickle.dump(pred, f)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"action_eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"heldout": res, "speed": speeds, "fanout": fan, "cv": cv_out,
                               "thetas": a.thetas, "importance": pred.feature_importance}, indent=2))
    print(f"\n-> {DATA / 'action_model.pkl'}\n-> {out}")


if __name__ == "__main__":
    main()
