"""
Step 1: export the real airline traces from the workload spreadsheet to CSV,
and split them into train / test BY TASK (not by trace).

Why by task: every tau-bench task was run ~12 times (several models, several
trials). If runs of the same task land in both train and test, the model has
"seen the answers" and the test score looks better than it really is.

    python predictor/export_traces.py            # run from the repo root

Writes:
    predictor/data/steps.csv   every transition  (workload, trace_id, task_id, step, from_tool, to_tool)
    predictor/data/train.csv   transitions of the training tasks
    predictor/data/test.csv    transitions of the test tasks
"""

import argparse
import random
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "workloads" / "Pulse_Workloads.xlsx"
OUT = Path(__file__).resolve().parent / "data"

WORKLOADS = ("W1", "W2")   # the two airline workloads (real traces)


def read_sheet(wb, name):
    rows = wb[name].iter_rows(values_only=True)
    header = next(rows)
    return pd.DataFrame(list(rows), columns=header)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-frac", type=float, default=0.2, help="share of TASKS kept for testing")
    p.add_argument("--seed", type=int, default=252)
    a = p.parse_args()

    print(f"reading {XLSX.name} (takes ~10-20 s) ...")
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    steps = read_sheet(wb, "Trace_Steps")
    traces = read_sheet(wb, "Traces")

    steps = steps[steps["workload"].isin(WORKLOADS)]
    traces = traces[traces["workload"].isin(WORKLOADS)][["trace_id", "task_id", "model"]]
    steps = steps.merge(traces, on="trace_id", how="left")
    steps = steps.sort_values(["trace_id", "step"]).reset_index(drop=True)

    # Seeded split of the task ids
    tasks = sorted(steps["task_id"].unique())
    rng = random.Random(a.seed)
    rng.shuffle(tasks)
    n_test = max(1, round(len(tasks) * a.test_frac))
    test_tasks = set(tasks[:n_test])
    steps["split"] = steps["task_id"].map(lambda t: "test" if t in test_tasks else "train")

    OUT.mkdir(parents=True, exist_ok=True)
    steps.to_csv(OUT / "steps.csv", index=False)
    steps[steps.split == "train"].to_csv(OUT / "train.csv", index=False)
    steps[steps.split == "test"].to_csv(OUT / "test.csv", index=False)

    for split in ("train", "test"):
        s = steps[steps.split == split]
        print(f"{split:5}: {s.task_id.nunique():3} tasks, {s.trace_id.nunique():4} traces, {len(s):5} transitions")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
