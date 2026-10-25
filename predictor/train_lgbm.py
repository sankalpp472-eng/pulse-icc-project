"""
LightGBM step 2: train gradient-boosted trees and compare them with the ensemble
on the SAME unseen test tasks.

Three versions, so you see what each kind of information adds:
  names    workflow, last tool, second-last tool, step          (same input as the ensemble)
  results  + what the tools returned (error? empty? basic economy? insurance? ...)
  intent   + customer intent flags (mentions cancel / change / bags / refund / ...)

    python predictor/train_lgbm.py                      # run from the repo root
    python predictor/train_lgbm.py --save intent        # which version to save for the API

Reads  predictor/data/lgbm_rows.csv (build_lgbm_data.py), predictor/data/model.pkl (train.py)
Writes predictor/data/lgbm.pkl, predictor/results/lgbm_eval_<time>.json
"""

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from lgbm_model import CATS, LGBMNextTool
from models import END
from signals import INTENT_KEYS, RESULT_KEYS

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
FEATURE_SETS = {
    "names": CATS + ["step"],
    "results": CATS + ["step"] + RESULT_KEYS,
    "intent": CATS + ["step"] + RESULT_KEYS + INTENT_KEYS,
}
PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=10, verbose=-1, seed=252, num_threads=2)


def prepare(df, categories, classes):
    df = df.rename(columns={"workload": "wf"}).copy()
    for c in CATS:
        df[c] = pd.Categorical(df[c], categories=categories[c])
    y = df["target"].map({c: i for i, c in enumerate(classes)})
    return df, y


def fit(train, feats, categories, classes, seed):
    """Early stopping on held-out TRAINING tasks, then refit on all training tasks."""
    tasks = sorted(train.task_id.unique())
    random.Random(seed).shuffle(tasks)
    val_tasks = set(tasks[:max(1, round(len(tasks) * 0.2))])
    tr, va = train[~train.task_id.isin(val_tasks)], train[train.task_id.isin(val_tasks)]
    params = {**PARAMS, "num_class": len(classes)}
    Xtr, ytr = prepare(tr, categories, classes)
    Xva, yva = prepare(va, categories, classes)
    probe = lgb.train(params, lgb.Dataset(Xtr[feats], ytr, categorical_feature=CATS),
                      num_boost_round=1000,
                      valid_sets=[lgb.Dataset(Xva[feats], yva, categorical_feature=CATS)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    rounds = max(20, probe.best_iteration)
    Xall, yall = prepare(train, categories, classes)
    return lgb.train(params, lgb.Dataset(Xall[feats], yall, categorical_feature=CATS), num_boost_round=rounds), rounds


def score(probs_list, targets, thetas):
    """probs_list: list of {tool: p}. Same metrics as evaluate.py."""
    n = len(targets)
    top1 = top3 = 0
    ll = 0.0
    for p, t in zip(probs_list, targets):
        ranked = sorted(p, key=p.get, reverse=True)
        top1 += ranked[0] == t
        top3 += t in ranked[:3]
        ll += -math.log(max(p.get(t, 0.0), 1e-12))
    sweep = []
    for th in thetas:
        calls = hits = warmed = wasted = 0
        for p, t in zip(probs_list, targets):
            warm = [j for j, v in p.items() if v >= th and j != END]
            warmed += len(warm)
            wasted += sum(j != t for j in warm)
            if t != END:
                calls += 1
                hits += t in warm
        sweep.append({"theta": th, "hit_rate": round(hits / calls, 3),
                      "warmed_per_step": round(warmed / n, 3), "wasted_per_step": round(wasted / n, 3)})
    return {"top1": round(top1 / n, 3), "top3": round(top3 / n, 3), "log_loss": round(ll / n, 3)}, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="intent", choices=list(FEATURE_SETS), help="version saved for the API")
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    ap.add_argument("--seed", type=int, default=252)
    a = ap.parse_args()

    rows = pd.read_csv(DATA / "lgbm_rows.csv")
    train, test = rows[rows.split == "train"], rows[rows.split == "test"]
    with open(DATA / "model.pkl", "rb") as f:
        pred = pickle.load(f)

    classes = sorted(train.target.unique())
    categories = {"wf": sorted(rows.workload.unique()),
                  "last1": sorted(set(rows.last1) | set(rows.last2)),
                  "last2": sorted(set(rows.last1) | set(rows.last2))}

    def vocab(wf):
        return pred.model_for(wf).vocab

    # Baseline: the ensemble on exactly these test rows
    def traj_of(r):
        return [t for t in (r.last2, r.last1) if t != "START"]
    ens = [pred.proba(r.workload, traj_of(r), "ensemble") for r in test.itertuples()]
    results = {"ensemble": score(ens, list(test.target), a.thetas)}

    saved = None
    Xte, _ = prepare(test, categories, classes)
    for name, feats in FEATURE_SETS.items():
        booster, rounds = fit(train, feats, categories, classes, a.seed)
        model = LGBMNextTool(booster, feats, classes, categories, name)
        raw = booster.predict(Xte[feats])
        probs = []
        for r, row in zip(test.itertuples(), raw):
            p = dict(zip(classes, row))
            out = {j: p.get(j, 0.0) + 1e-6 for j in vocab(r.workload)}
            s = sum(out.values())
            probs.append({j: v / s for j, v in out.items()})
        results[f"lgbm_{name}"] = score(probs, list(test.target), a.thetas)
        results[f"lgbm_{name}"][0]["rounds"] = rounds
        if name == a.save:
            saved = model
            imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda x: -x[1])
            top_features = [(f, round(float(g))) for f, g in imp[:12]]

    print(f"Test: {test.task_id.nunique()} unseen tasks, {len(test)} steps\n")
    print(f"{'model':16} {'top1':>6} {'top3':>6} {'logloss':>8}   " +
          "   ".join(f"hit@θ{t}/wasted" for t in a.thetas))
    for k, (m, sw) in results.items():
        print(f"{k:16} {m['top1']:6.3f} {m['top3']:6.3f} {m['log_loss']:8.3f}   " +
              "   ".join(f"{s['hit_rate']:.2f}/{s['wasted_per_step']:.2f}       " for s in sw))
    print(f"\nMost useful features ({a.save}):", ", ".join(f"{f}" for f, _ in top_features))

    with open(DATA / "lgbm.pkl", "wb") as f:
        pickle.dump(saved, f)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"lgbm_eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({k: {"metrics": m, "theta_sweep": sw} for k, (m, sw) in results.items()}
                              | {"saved": a.save, "top_features": top_features}, indent=2))
    print(f"-> {DATA / 'lgbm.pkl'} ({a.save})\n-> {out}")


if __name__ == "__main__":
    main()
