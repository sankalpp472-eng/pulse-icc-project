"""
Action model, step 1: turn the recorded tau-bench airline conversations into training rows.

One row per LLM STEP, because the agent sends a hint before every LLM step:
  target   what that step did: one of the 14 tools, or RESPOND (the agent answered the customer;
           about half of all steps). Prewarming nothing on RESPOND steps is the right answer.
  k_true   fan-out: how many times in a row the agent then calls that same tool
           (e.g. get_reservation_details once per pending reservation). 0 for RESPOND.
  features last tool, last two tools, workflow + the signals in predictor/signals.py
           (all numbers / flags, computed from what was known before the step).

Airline uses the SAME train / test tasks as export_traces.py (from predictor/data/steps.csv).
Retail (--domain retail, workload W3: 16 tools + RESPOND) uses the LLM-speculator baseline's split
(baselines/llm_speculator/steps.py): runs de-duplicated, task ids sorted, shuffled with seed 252,
the first 20% held out. The held-out tasks are written to predictor/data/retail_split.json so the
speculator and this model are scored on the same tasks.

Needs the full conversations (tool results included), which the workbook does not store:
  git clone --depth 1 https://github.com/sierra-research/tau-bench ../tau-bench
  python predictor/build_action_data.py --tau-bench ../tau-bench            # from the repo root
  python predictor/build_action_data.py --tau-bench ../tau-bench --domain retail

Writes predictor/data/action_rows.csv (airline) or predictor/data/retail_action_rows.csv
"""

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from signals import SIGNAL_KEYS, compute_signals, pending_pool, uses_pending

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FILES = {"airline": ["gpt-4o-airline.json", "sonnet-35-new-airline.json"],
         "retail": ["gpt-4o-retail.json", "sonnet-35-new-retail.json"]}
START, RESPOND = "START", "RESPOND"


def speculator_split(runs, seed=252, test_frac=0.2):
    """The LLM-speculator baseline's split (baselines/llm_speculator/steps.py): drop duplicate runs,
    then hold out the first test_frac of the sorted task ids after a seeded shuffle.
    runs = [(..., run dict)] in file order -> (kept runs, held-out task ids)."""
    seen, kept = set(), []
    for item in runs:
        key = hashlib.sha1(json.dumps(item[-1]["traj"], sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            kept.append(item)
    tasks = sorted({item[-1]["task_id"] for item in kept})
    random.Random(seed).shuffle(tasks)
    return kept, set(tasks[:max(1, round(len(tasks) * test_frac))])


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
        r["llm_step"] = i                      # index among the run's LLM steps (the speculator's step key)
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
    ap.add_argument("--domain", choices=list(FILES), default="airline")
    ap.add_argument("--seed", type=int, default=252, help="speculator split seed (must match the speculator's)")
    ap.add_argument("--split", choices=["steps_csv", "speculator"], default=None,
                    help="airline: steps_csv (default, same tasks as export_traces.py) or speculator (its 10 "
                         "held-out tasks, for a like-for-like comparison); retail: always speculator")
    a = ap.parse_args()
    split = "speculator" if a.domain == "retail" else (a.split or "steps_csv")

    def load(fname):
        path = a.tau_bench / "historical_trajectories" / fname
        return json.loads(path.read_text(encoding="utf-8"))

    rows = []
    if split == "steps_csv":
        steps = pd.read_csv(DATA / "steps.csv")      # from export_traces.py: task -> workload, split
        info = steps.drop_duplicates("task_id").set_index("task_id")[["workload", "split"]].to_dict("index")
        for fname in FILES["airline"]:
            for i, ep in enumerate(load(fname)):
                task = f"airline-{ep['task_id']}"
                if task in info:
                    rows += rows_for(ep["traj"], {"trace": f"{fname}#{i}", "task_id": task, **info[task]})
        out = DATA / "action_rows.csv"
    else:
        runs = [(fname, i, ep) for fname in FILES[a.domain] for i, ep in enumerate(load(fname))]
        kept, test = speculator_split(runs, a.seed)
        wl = {}                                              # airline: W1 / W2 per task, as in steps.csv
        if a.domain == "airline":
            wl = pd.read_csv(DATA / "steps.csv").drop_duplicates("task_id").set_index("task_id").workload.to_dict()
        for fname, i, ep in kept:
            model = fname.split("-" + a.domain)[0]          # same run ids as the speculator's steps.py
            task = f"{a.domain}-{ep['task_id']}"
            rows += rows_for(ep["traj"], {
                "trace": f"{fname}#{i}", "task_id": task, "workload": wl.get(task, "W3"),
                "split": "test" if ep["task_id"] in test else "train",
                "run_id": f"{a.domain}/{model}/task{ep['task_id']}/trial{ep.get('trial', 0)}"})
        DATA.mkdir(exist_ok=True)
        (DATA / f"{a.domain}_split.json").write_text(json.dumps(
            {"domain": a.domain, "rule": "baselines/llm_speculator/steps.py: de-duplicated runs, sorted task ids, "
                                         f"random.Random({a.seed}).shuffle, first 20% held out",
             "seed": a.seed, "runs_kept": len(kept), "duplicates_dropped": len(runs) - len(kept),
             "test_tasks": sorted(test)}, indent=1))
        out = DATA / ("retail_action_rows.csv" if a.domain == "retail" else "airline_specsplit_action_rows.csv")

    cols = ["trace", "task_id", "workload", "split", "last1", "last2", *SIGNAL_KEYS, "target", "k_true", "used_pending"]
    if split == "speculator":
        cols = ["run_id", "llm_step"] + cols     # step key = f"{run_id}#{llm_step}"
    df = pd.DataFrame(rows)[cols]
    df.to_csv(out, index=False)
    resp = (df.target == RESPOND).mean()
    for split in ("train", "test"):
        d = df[df.split == split]
        print(f"{split:5}: {d.task_id.nunique():3} tasks, {d.trace.nunique():4} traces, {len(d):5} LLM steps")
    print(f"RESPOND share: {resp:.0%}   classes: {df.target.nunique()} ({df.target.nunique() - 1} tools + RESPOND)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
