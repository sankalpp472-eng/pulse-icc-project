"""
Action model, step 2: train and evaluate the next-action predictor.

  Target 1  next action = one of the 14 tools or RESPOND (one prediction per LLM step)
  Target 2  fan-out k for the predicted tool (pending-ID rule) -> desiredReplicas in the CRD
  Models    gbt (LightGBM gradient-boosted trees: the lightgbm predictor) · ngram (order-2 Markov chain with
            back-off: the markov predictor)

    python predictor/train_action.py                  # from the repo root: train on the 40 training tasks,
                                                      # score on the 10 held-out tasks, save the model
    python predictor/train_action.py --cv 5           # also 5-fold cross-validation over all 50 tasks
    python predictor/train_action.py --domain retail  # retail (16 tools + RESPOND): 92 training tasks,
                                                      # the speculator's 23 held-out tasks

Reads  predictor/data/action_rows.csv (build_action_data.py; retail: retail_action_rows.csv)
Writes predictor/data/action_model.pkl (retail: action_model_retail.pkl),
       predictor/results/action_eval_<time>.json (retail: action_eval_retail_<time>.json)

With the S3 model registry (MLOps; airline, default split only):

    python predictor/train_action.py --uri s3://<bucket>/action-model                  # gate decides
    python predictor/train_action.py --uri s3://<bucket>/action-model --promote never --notes "..."

  1. data     pull <uri>/_data/action_rows.csv into predictor/data/ (push it once:
              python pulse_pred/model_store.py data push <uri> --local predictor/data --files action_rows.csv)
  2. train    as above; held-out = the same 10 unseen tasks
  3. evaluate the new model AND the registry's current latest on the same held-out steps
  4. gate     --promote auto: the new model becomes latest only if lightgbm's top-1 beats the current one's;
              otherwise it is published as a candidate. always / never override.
  5. publish  a new version (predictor/data/models/<version>/action_model.pkl) with the metrics in its manifest;
              the predictors load it with ACTION_MODEL_URI (predictor/predictors/action.py)
"""

import argparse
import json
import math
import os
import pickle
import random
import statistics as st
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from action_model import CATS, NUMERIC, RESPOND, ActionPredictor, Encoder, FanoutRule, GBTModel, NGramModel

HERE = Path(__file__).resolve().parent
# overridable for the training Job, whose image folders are read-only (deploy/training/train-action-job.yaml)
DATA = Path(os.getenv("ACTION_DATA_DIR", HERE / "data"))
RESULTS = Path(os.getenv("ACTION_RESULTS_DIR", HERE / "results"))
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
    models = {"gbt": GBTModel(booster, classes), "ngram": NGramModel().fit(train)}
    importance = dict(zip(enc.columns, (booster.feature_importance("gain")).round(1).tolist()))
    return ActionPredictor(enc, models, FanoutRule.fit(train), importance)


def with_next_tool(df):
    """Lookahead rows: the RESPOND steps only, with target = the tool the agent calls next, after the customer
    replies (the next non-RESPOND action of the same conversation; END if it never calls one).
    Rows of one conversation are in step order (build_action_data.py writes them that way)."""
    out = []
    for _, g in df.groupby("trace", sort=False):
        t = g.target.tolist()
        nxt = [next((x for x in t[i + 1:] if x != RESPOND), "END") for i in range(len(t))]
        out.append(g.assign(target=nxt)[g.target.to_numpy() == RESPOND])
    return pd.concat(out, ignore_index=True)


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


MODEL_ORDER = ["ngram", "gbt"]
LABEL = {"ngram": "Markov chain (markov)", "gbt": "LightGBM GBT (lightgbm, main)"}


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
    ap.add_argument("--domain", choices=["airline", "retail"], default="airline")
    ap.add_argument("--split", choices=["steps_csv", "speculator"], default="steps_csv",
                    help="airline only: speculator = the LLM speculator's held-out tasks "
                         "(build_action_data.py --split speculator), saved as *_airline_specsplit")
    ap.add_argument("--uri", help="model registry, e.g. s3://pulse-models-<account>/action-model: pull the data, "
                                  "gate against the current latest, publish a new version")
    ap.add_argument("--version", default=time.strftime("%Y-%m-%d_%H%M"))
    ap.add_argument("--promote", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.domain == "retail":
        tag, rows_file = "_retail", "retail_action_rows.csv"
    elif a.split == "speculator":
        tag, rows_file = "_airline_specsplit", "airline_specsplit_action_rows.csv"
    else:                                                     # airline keeps the original file names
        tag, rows_file = "", "action_rows.csv"
    store = None
    if a.uri:
        if tag:
            raise SystemExit("--uri: the registry holds the airline model with the default split only")
        store = registry()
        print(f"data: {len(store.data('pull', a.uri, local=DATA))} file(s) downloaded from {a.uri}/_data/")
        if not (DATA / rows_file).exists():
            raise SystemExit(f"no {DATA / rows_file}: push it once (python pulse_pred/model_store.py data push "
                             f"{a.uri} --local predictor/data --files {rows_file})")

    rows = pd.read_csv(DATA / rows_file)
    train, test = rows[rows.split == "train"], rows[rows.split == "test"]
    pred = fit_all(train, a.seed)
    # lookahead model (service.py LOOKAHEAD=1): on a reply step, which tool comes after the customer's reply.
    # Saved inside the same pickle; old pickles simply have no .lookahead.
    nxt_train, nxt_test = with_next_tool(train), with_next_tool(test)
    pred.lookahead = fit_all(nxt_train, a.seed)
    nxt_tools = nxt_test[nxt_test.target != "END"]          # reply steps that are followed by a tool call
    look = score(pred.lookahead, "gbt", nxt_tools, a.thetas)[0]
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
               "", f"Lookahead (gbt, trained on {len(nxt_train)} reply steps): on the {len(nxt_tools)} held-out reply steps "
                   f"that are followed by a tool call, the next tool is top-1 {look['top1']:.3f} / top-3 {look['top3']:.3f}; "
                   "prewarmed at " + ", ".join(f"θ={t}: {look[f'hit@{t}']:.2f} (wasted {look[f'wasted@{t}']:.2f}/step)"
                                                for t in a.thetas)]

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
    gate = None
    if store:
        model_path = DATA / "models" / a.version / f"action_model{tag}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        gate = gate_and_metrics(store, a, res, look, test, nxt_tools)
    else:
        model_path = DATA / f"action_model{tag}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pred, f)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"action_eval{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"domain": a.domain, "test_tasks": sorted(test.task_id.unique()),
                               "heldout": res, "lookahead": look, "speed": speeds, "fanout": fan, "cv": cv_out,
                               "thetas": a.thetas, "importance": pred.feature_importance,
                               **({"registry": gate} if gate else {})}, indent=2))
    print(f"\n-> {model_path}\n-> {out}")
    if store:
        print(f"publishing version {a.version}" + (" as latest" if gate["promoted"] else " as a candidate"))
        store.publish(model_path, a.uri, a.version, a.notes, latest=gate["promoted"], metrics=gate)
        print(f"GATE: {'promoted' if gate['promoted'] else 'candidate'} {a.version} ({gate['why']})")


# ----------------------------------------------------------------- model registry (MLOps)

GATE_METRIC = "lightgbm_top1"


def registry():
    """pulse_pred/model_store.py: the S3 model registry, shared with pulse-v3."""
    import sys
    root = str(HERE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from pulse_pred import model_store
    return model_store


def summary(res, look):
    """The numbers a version is judged and listed by (one flat dict, stored in its manifest)."""
    g, m = res["gbt"], res["ngram"]
    return {"lightgbm_top1": round(g["top1"], 4), "lightgbm_top3": round(g["top3"], 4),
            "lightgbm_hit@0.3": round(g["hit@0.3"], 4) if "hit@0.3" in g else None,
            "lightgbm_log_loss": round(g["log_loss"], 4), "markov_top1": round(m["top1"], 4),
            "lookahead_top1": round(look["top1"], 4) if look else None}


def gate_and_metrics(store, a, res, look, test, nxt_tools):
    """Score the registry's current latest on the same held-out steps and decide whether the new one replaces it."""
    new, current, current_version = summary(res, look), None, None
    try:
        _, _, current_version = store.resolve(a.uri)
        with open(store.fetch(a.uri, dest=DATA / "models"), "rb") as f:
            cur = pickle.load(f)
        cur_res = {m: score(cur, m, test, [0.3])[0] for m in ("gbt", "ngram")}
        cur_look = score(cur.lookahead, "gbt", nxt_tools, [0.3])[0] if getattr(cur, "lookahead", None) else None
        current = summary(cur_res, cur_look)
    except Exception as e:                                    # an empty registry: nothing to beat
        print(f"no current model in {a.uri} ({type(e).__name__}: {str(e)[:120]})")
    if a.promote == "always" or current is None:
        promoted, why = a.promote != "never", "no current model" if current is None else "--promote always"
    elif a.promote == "never":
        promoted, why = False, "--promote never"
    else:
        promoted = new[GATE_METRIC] > current[GATE_METRIC]
        why = (f"{GATE_METRIC} {new[GATE_METRIC]} {'>' if promoted else '<='} {current[GATE_METRIC]} "
               f"({current_version})")
    print(f"gate: {'promote' if promoted else 'keep as candidate'}: {why}")
    return {"gate_metric": GATE_METRIC, "new": new, "current": current, "current_version": current_version,
            "promoted": promoted, "why": why, "test_tasks": int(test.task_id.nunique()), "test_steps": len(test)}


if __name__ == "__main__":
    main()
