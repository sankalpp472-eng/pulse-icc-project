"""
LightGBM step 1: turn the recorded tau-bench airline conversations into training rows.

One row per decision "what does the agent call next?" (and a final row with target END),
with what was known at that moment:
  workflow, last tool, second-last tool                    (same as the counting models)
  + signals from predictor/signals.py                       (tool results + customer intent flags)

Uses the SAME tasks for train / test as export_traces.py (read from predictor/data/steps.csv),
so LightGBM and the ensemble are compared on the same unseen tasks.

Needs the full conversations (tool results included), which the workbook does not store:
  git clone --depth 1 https://github.com/sierra-research/tau-bench <somewhere>
  python predictor/build_lgbm_data.py --tau-bench <somewhere>/tau-bench      # from the repo root

Writes predictor/data/lgbm_rows.csv
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from signals import SIGNAL_KEYS, compute_signals

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FILES = ["gpt-4o-airline.json", "sonnet-35-new-airline.json"]
START, END = "START", "END"


def tool_names(msg):
    calls = msg.get("tool_calls")
    if isinstance(calls, str):              # some files store the list as a string
        try:
            calls = ast.literal_eval(calls)
        except (ValueError, SyntaxError):
            calls = None
    return [c["function"]["name"] for c in (calls or [])]


def rows_for(traj, meta):
    events, called, rows = [], [], []

    def row(target):
        r = {**meta, "step": len(called) + 1,
             "last1": called[-1] if called else START,
             "last2": called[-2] if len(called) >= 2 else START,
             "target": target}
        r.update(compute_signals(events))
        rows.append(r)

    for m in traj:
        role = m.get("role")
        if role == "user":
            events.append(("user", m.get("content") or ""))
        elif role == "assistant":
            for name in tool_names(m):
                row(name)                       # decision made with what was known so far
                called.append(name)
        elif role == "tool":
            events.append(("tool", m.get("name") or (called[-1] if called else ""), str(m.get("content") or "")))
    row(END)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-bench", required=True, type=Path, help="clone of sierra-research/tau-bench")
    a = ap.parse_args()

    steps = pd.read_csv(DATA / "steps.csv")      # from export_traces.py: task -> workload, split
    task_info = steps.drop_duplicates("task_id").set_index("task_id")[["workload", "split"]].to_dict("index")

    rows = []
    for fname in FILES:
        for i, ep in enumerate(json.loads((a.tau_bench / "historical_trajectories" / fname).read_text())):
            task = f"airline-{ep['task_id']}"
            if task not in task_info:
                continue
            meta = {"trace": f"{fname}#{i}", "task_id": task, **task_info[task]}
            rows += rows_for(ep["traj"], meta)

    df = pd.DataFrame(rows)[["trace", "task_id", "workload", "split", "step", "last1", "last2", *SIGNAL_KEYS, "target"]]
    df.to_csv(DATA / "lgbm_rows.csv", index=False)
    for split in ("train", "test"):
        d = df[df.split == split]
        print(f"{split:5}: {d.task_id.nunique():3} tasks, {d.trace.nunique():4} traces, {len(d):5} rows")
    print(f"-> {DATA / 'lgbm_rows.csv'}")


if __name__ == "__main__":
    main()
