"""
Step 5: how good are the predictions, on tasks the models have NEVER seen (test.csv)?

Part A, accuracy (all models):
  top-1     : the highest-ranked tool was the real next tool
  top-3     : the real next tool was among the 3 highest
  log-loss  : how "surprised" the model was by the real next tool (lower is better)

Part B, prewarming (one model, threshold sweep, feeds experiment E6):
  For each threshold theta, we "prewarm" every tool with P >= theta.
  hit rate     : of the real tool calls, how many found their tool prewarmed
  wasted/step  : prewarmed tools that were NOT called next (idle pods = cost)
  warmed/step  : how many tools were prewarmed per step

Part C, lead time: is the LLM's thinking time long enough to hide a cold start?
  Uses agent/runs/*.jsonl (think time) and deploy/knative/results/*.csv (cold start).

    python predictor/evaluate.py                         # run from the repo root
    python predictor/evaluate.py --model lr --thetas 0.2 0.3 0.5
"""

import argparse
import glob
import json
import pickle
import statistics
import time
from pathlib import Path

import pandas as pd

from models import END, MODEL_NAMES, examples, log_loss, prewarm_set, sequences_from_steps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
RESULTS = HERE / "results"


def accuracy(pred, seqs_by_wf, model):
    n = top1 = top3 = 0
    ll = 0.0
    for wf, seqs in seqs_by_wf.items():
        for seq in seqs:
            for traj, nxt in examples(seq):
                ranked = pred.rank(wf, traj, model)
                tools = [t for t, _ in ranked]
                n += 1
                top1 += tools[0] == nxt
                top3 += nxt in tools[:3]
                ll += log_loss(dict(ranked)[nxt])
    return {"model": model, "steps": n, "top1": top1 / n, "top3": top3 / n, "log_loss": ll / n}


def theta_sweep(pred, seqs_by_wf, model, thetas):
    rows = []
    for theta in thetas:
        steps = calls = hits = warmed = wasted = 0
        for wf, seqs in seqs_by_wf.items():
            for seq in seqs:
                for traj, nxt in examples(seq):
                    warm = prewarm_set(pred.rank(wf, traj, model), theta)
                    steps += 1
                    warmed += len(warm)
                    wasted += sum(t != nxt for t in warm)
                    if nxt != END:          # END is not a function call
                        calls += 1
                        hits += nxt in warm
        rows.append({"theta": theta, "hit_rate": hits / calls,
                     "warmed_per_step": warmed / steps, "wasted_per_step": wasted / steps})
    return rows


def lead_time():
    think = []
    for f in glob.glob(str(ROOT / "agent" / "runs" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            ep = json.loads(line)
            think += [s["think_s"] for s in ep.get("steps", []) if s.get("kind") == "tool"]
    cold = []
    for f in glob.glob(str(ROOT / "deploy" / "knative" / "results" / "*.csv")):
        df = pd.read_csv(f)
        cold += (df["cold_ms"] / 1000).tolist()
    if not think or not cold:
        return None
    t, c = statistics.median(think), statistics.median(cold)
    hidden = [min(x, c) / c for x in think]   # share of a median cold start hidden by each think
    return {"tool_steps": len(think), "think_s_median": t, "cold_s_median": c,
            "share_of_cold_start_hidden_median": statistics.median(hidden),
            "calls_where_think_covers_cold_start": sum(x >= c for x in think) / len(think)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ensemble", choices=MODEL_NAMES, help="model for the theta sweep")
    p.add_argument("--thetas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    a = p.parse_args()

    with open(DATA / "model.pkl", "rb") as f:
        pred = pickle.load(f)
    test = sequences_from_steps(pd.read_csv(DATA / "test.csv"))
    test.pop("airline", None)          # score each trace once, with its own workflow model

    print("A. accuracy on unseen test tasks")
    acc = [accuracy(pred, test, m) for m in MODEL_NAMES]
    print(pd.DataFrame(acc).to_string(index=False, float_format="%.3f"))

    print(f"\nB. prewarming with model={a.model}")
    sweep = theta_sweep(pred, test, a.model, a.thetas)
    print(pd.DataFrame(sweep).to_string(index=False, float_format="%.3f"))

    print("\nC. lead time (LLM think time vs cold start)")
    lt = lead_time()
    print(json.dumps(lt, indent=2) if lt else "  no agent runs or cold-start results found")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"accuracy": acc, "theta_sweep": sweep, "lead_time": lt,
                               "sweep_model": a.model}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
