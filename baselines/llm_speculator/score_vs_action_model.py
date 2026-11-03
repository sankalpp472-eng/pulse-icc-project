"""
Score the LLM speculator against the next-action models (LightGBM gbt, logistic regression, n-gram from
predictor/action_model.py) on the SAME sampled steps.

score.py needs the v2 predictor (tau_data.py, evaluate.py), which is no longer in the repo; this script
needs only the speculator's step files and an action model trained on the speculator's split:

    python predictor/build_action_data.py --tau-bench <tau-bench> --split speculator     # airline rows, run_id + llm_step
    python predictor/train_action.py --split speculator                                  # -> action_model_airline_specsplit.pkl
    python baselines/llm_speculator/score_vs_action_model.py results/<speculate.py folder>

Same definitions as score.py (accuracy, timing) and predictor/train_action.py (theta columns):
  tool top1 / top3  tool steps, tools ranked with RESPOND removed       all top1  every step, all actions
  log-loss          every step (probabilities clipped at 1e-6)
  in time           guesses back within --lead-s (hint -> call, ~1.0 s in our C3 runs)
  timely top3       tool steps: tool in the top 3 AND the guess back in time
  cold hidden       tool steps: min(lead - latency, cold) / cold when the tool is in the top 3, else 0
  hit / wasted @ θ  tool steps whose tool had p >= θ / prewarmed tools that were not the next action, per step
LLM answers that could not be parsed count as misses; unlisted actions share the leftover mass (as score.py).
Writes results/compare_action_model_<time>.json
"""

import argparse
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "predictor"))
import action_model  # noqa: E402,F401  (pickle needs the classes)
from action_model import START  # noqa: E402
from signals import SIGNAL_KEYS  # noqa: E402

RESPOND = "RESPOND"
THETAS = [0.2, 0.3, 0.5]


def llm_distribution(pred, names):
    """As score.py: listed actions keep their p; the rest share what is left, each below the smallest listed p."""
    if not pred:
        return {}
    listed = {a: max(p, 1e-6) for a, p in pred.items()}
    rest = [n for n in names if n not in listed]
    left = max(0.0, 1 - sum(listed.values()))
    each = min(left / max(len(rest), 1), 0.5 * min(listed.values())) if rest else 0
    s = sum(listed.values()) + each * len(rest)
    return {**{a: p / s for a, p in listed.items()}, **{a: max(each, 1e-9) / s for a in rest}}


def tools_ranked(p):
    return [a for a, _ in sorted(((a, v) for a, v in p.items() if a != RESPOND), key=lambda kv: -kv[1])]


def score(P, actions, lat_s, lead_s, cold_s):
    tool = [(p, y, l) for p, y, l in zip(P, actions, lat_s) if y != RESPOND]
    n, nt = len(actions), max(len(tool), 1)
    out = {"steps": n, "tool_steps": len(tool),
           "tool_top1": sum(bool(p) and tools_ranked(p)[:1] == [y] for p, y, _ in tool) / nt,
           "tool_top3": sum(y in tools_ranked(p)[:3] for p, y, _ in tool) / nt,
           "all_top1": sum(bool(p) and max(p, key=p.get) == y for p, y in zip(P, actions)) / n,
           "log_loss": statistics.mean(-math.log(max(p.get(y, 0.0), 1e-6)) for p, y in zip(P, actions))}
    known = sorted(l for l in lat_s if l is not None)
    out["latency_p50_s"] = known[len(known) // 2] if known else None
    out["latency_p95_s"] = known[int(0.95 * (len(known) - 1))] if known else None
    out["in_time"] = sum(l <= lead_s for l in known) / n
    out["timely_tool_top3"] = sum(y in tools_ranked(p)[:3] and l is not None and l <= lead_s for p, y, l in tool) / nt
    out["cold_hidden_top3"] = sum(min(max(lead_s - l, 0.0), cold_s) / cold_s
                                  for p, y, l in tool if l is not None and y in tools_ranked(p)[:3]) / nt
    for th in THETAS:
        warm = [[a for a, v in p.items() if v >= th and a not in (RESPOND, "END")] for p in P]
        out[f"hit@{th}"] = sum(y in w for w, y in zip(warm, actions) if y != RESPOND) / nt
        out[f"wasted@{th}"] = sum(sum(a != y for a in w) for w, y in zip(warm, actions)) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="speculate.py result folders (same sample)")
    ap.add_argument("--rows", default=str(ROOT / "predictor" / "data" / "airline_specsplit_action_rows.csv"))
    ap.add_argument("--model", default=str(ROOT / "predictor" / "data" / "action_model_airline_specsplit.pkl"))
    ap.add_argument("--lead-s", type=float, default=1.0)
    ap.add_argument("--cold-s", type=float, default=1.8)
    a = ap.parse_args()

    run_dirs = [Path(r) if Path(r).exists() else HERE / r for r in a.runs]
    cfgs = [json.loads((d / "run.json").read_text()) for d in run_dirs]
    sample = json.loads(Path(cfgs[0]["sample"]).read_text())
    preds = [{r["key"]: r for r in map(json.loads, open(d / "predictions.jsonl", encoding="utf-8"))} for d in run_dirs]
    keys = [k for k in sample["keys"] if all(k in pr for pr in preds)]      # steps every run answered

    rows = pd.read_csv(a.rows)
    rows = rows[rows.split == "test"]
    rows.index = rows.run_id + "#" + rows.llm_step.astype(str)
    missing = [k for k in keys if k not in rows.index]
    if missing:
        raise SystemExit(f"{len(missing)} sampled steps are not in {a.rows} (built with --split speculator?)")
    with open(a.model, "rb") as f:
        pred = pickle.load(f)
    names = sorted(set(pred.models["gbt"].classes))
    print(f"{sample['domain']}: {len(keys)} sampled steps ({sum(rows.loc[k, 'target'] != RESPOND for k in keys)} tool steps) "
          f"of {len(rows)} held-out | lead {a.lead_s} s, cold start {a.cold_s} s")

    def model_preds(name, idx):
        P, lat = [], []
        for k in idx:
            r = rows.loc[k]
            traj = [t for t in (r.last2, r.last1) if t != START]
            t0 = time.perf_counter()
            P.append(pred.proba(name, r.workload, traj, {s: r[s] for s in SIGNAL_KEYS}))
            lat.append(time.perf_counter() - t0)
        return P, lat

    results = []
    for name in ("ngram", "lr", "gbt"):
        P, lat = model_preds(name, keys)
        results.append({"method": name, **score(P, [rows.loc[k, "target"] for k in keys], lat, a.lead_s, a.cold_s)})
        Pf, latf = model_preds(name, list(rows.index))           # all held-out steps: is the sample representative?
        results[-1]["full_test_tool_top1"] = score(Pf, list(rows.target), latf, a.lead_s, a.cold_s)["tool_top1"]
    for d, cfg, pr in zip(run_dirs, cfgs, preds):
        rs = [pr[k] for k in keys]
        P = [llm_distribution(r["pred"], names) if r.get("ok") else {} for r in rs]
        results.append({"method": f"llm:{cfg['model']}", "run": d.name,
                        "parse_fail": sum(not r.get("ok") for r in rs) / len(rs),
                        **score(P, [rows.loc[k, "target"] for k in keys], [r.get("latency_s") for r in rs],
                                a.lead_s, a.cold_s)})

    def f(x, w=7, d=3):
        return f"{x:{w}.{d}f}" if x is not None else f"{'-':>{w}}"
    print(f"\n  {'method':26} {'tool top1':>9} {'top3':>6} {'all top1':>8} {'logloss':>7} {'p50 s':>8} {'p95 s':>8} "
          f"{'in time':>7} {'timely3':>7} {'cold hid':>8}  " + "  ".join(f"θ={t} hit/wasted" for t in THETAS))
    for r in results:
        print(f"  {r['method']:26} {f(r['tool_top1'], 9)} {f(r['tool_top3'], 6)} {f(r['all_top1'], 8)} {f(r['log_loss'], 7)} "
              f"{f(r['latency_p50_s'], 8, 4)} {f(r['latency_p95_s'], 8, 4)} {f(r['in_time'], 7, 2)} "
              f"{f(r['timely_tool_top3'], 7)} {f(r['cold_hidden_top3'], 8)}  "
              + "  ".join(f"{r[f'hit@{t}']:.2f} / {r[f'wasted@{t}']:.2f}".ljust(16) for t in THETAS)
              + (f"  parse fail {r['parse_fail']:.2f}" if "parse_fail" in r else ""))
    print("  (full held-out tool top1: " + ", ".join(f"{r['method']} {r['full_test_tool_top1']:.3f}"
                                                     for r in results if "full_test_tool_top1" in r) + ")")

    out = HERE / "results" / f"compare_action_model_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"domain": sample["domain"], "steps": len(keys), "lead_s": a.lead_s, "cold_s": a.cold_s,
                               "rows": a.rows, "model": a.model, "keys": keys, "methods": results}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
