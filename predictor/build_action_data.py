"""
Action model, step 1: turn the recorded tau-bench airline conversations into training rows.

One row per LLM STEP, because the agent sends a hint before every LLM step:
  target   what that step did: one of the 14 tools, or RESPOND (the agent answered the customer;
           about half of all steps). Prewarming nothing on RESPOND steps is the right answer.
  k_true   fan-out: how many times in a row the agent then calls that same tool
           (e.g. get_reservation_details once per pending reservation). 0 for RESPOND.
  features last tool, last two tools, workflow + the signals in predictor/signals.py
           (all numbers / flags, computed from what was known before the step).

Uses the SAME train / test tasks as export_traces.py (from predictor/data/steps.csv).

Needs the full conversations (tool results included), which the workbook does not store:
  git clone --depth 1 https://github.com/sierra-research/tau-bench ../tau-bench
  python predictor/build_action_data.py --tau-bench ../tau-bench            # from the repo root

Writes predictor/data/action_rows.csv
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from signals import SIGNAL_KEYS, compute_signals, pending_pool, uses_pending

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FILES = ["gpt-4o-airline.json", "sonnet-35-new-airline.json"]
START, RESPOND = "START", "RESPOND"


def tool_calls(msg):
    calls = msg.get("tool_calls")
    if isinstance(calls, str):                 # some files store the list as a string
        try:
            calls = ast.literal_eval(calls)
        except (ValueError, SyntaxError):
            calls = None
    out = []
    for c in calls or []:
        args = c["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        out.append((c["function"]["name"], args or {}))
    return out


def rows_for(traj, meta):
    events, called, steps = [], [], []
    for m in traj:
        role = m.get("role")
        if role == "user":
            events.append(("user", m.get("content") or ""))
        elif role == "tool":
            events.append(("tool", m.get("name") or (called[-1] if called else ""), str(m.get("content") or "")))
        elif role == "assistant":
            calls = tool_calls(m)
            row = {**meta, "last1": called[-1] if called else START,
                   "last2": called[-2] if len(called) >= 2 else START,
                   **compute_signals(events)}
            if calls:
                name, args = calls[0]
                row["target"] = name
                row["used_pending"] = int(uses_pending(args, pending_pool(events)))
                for n, a in calls:
                    events.append(("call", n, a))
                    called.append(n)
            else:
                row["target"] = RESPOND
                row["used_pending"] = 0
                events.append(("respond",))
            steps.append(row)
    # fan-out: run length of the same tool from this step on (a RESPOND or another tool ends it)
    for i, r in enumerate(steps):
        if r["target"] == RESPOND:
            r["k_true"] = 0
            continue
        k = 1
        while i + k < len(steps) and steps[i + k]["target"] == r["target"]:
            k += 1
        r["k_true"] = k
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-bench", required=True, type=Path, help="clone of sierra-research/tau-bench")
    a = ap.parse_args()

    steps = pd.read_csv(DATA / "steps.csv")      # from export_traces.py: task -> workload, split
    info = steps.drop_duplicates("task_id").set_index("task_id")[["workload", "split"]].to_dict("index")

    rows = []
    for fname in FILES:
        for i, ep in enumerate(json.loads((a.tau_bench / "historical_trajectories" / fname).read_text())):
            task = f"airline-{ep['task_id']}"
            if task in info:
                rows += rows_for(ep["traj"], {"trace": f"{fname}#{i}", "task_id": task, **info[task]})

    cols = ["trace", "task_id", "workload", "split", "last1", "last2", *SIGNAL_KEYS, "target", "k_true", "used_pending"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(DATA / "action_rows.csv", index=False)
    resp = (df.target == RESPOND).mean()
    for split in ("train", "test"):
        d = df[df.split == split]
        print(f"{split:5}: {d.task_id.nunique():3} tasks, {d.trace.nunique():4} traces, {len(d):5} LLM steps")
    print(f"RESPOND share: {resp:.0%}   classes: {df.target.nunique()} (14 tools + RESPOND)")
    print(f"-> {DATA / 'action_rows.csv'}")


if __name__ == "__main__":
    main()
