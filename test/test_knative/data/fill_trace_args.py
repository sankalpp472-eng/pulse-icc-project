"""
Fill in the tool-call arguments of the replay traces (w1_tau_traces.jsonl, w1_eks5_traces.jsonl)
from tau-bench's published airline trajectories.

The first export kept arguments only when they were a JSON string (gpt-4o's format); claude-3.5-sonnet
stores them as an object, so all its calls were exported with args = {}. This script matches every
trace to its source run (task, model, trial), checks that the tool sequence is identical, and only then
copies the arguments. Everything else (order of traces, user turns, rewards) stays as it was, so a
seeded replay picks the same episodes as before.

Get the two source files once (https://github.com/sierra-research/tau-bench/tree/main/historical_trajectories):
  gpt-4o-airline.json, sonnet-35-new-airline.json

  agent/.venv/Scripts/python test/test_knative/data/fill_trace_args.py --tau-dir <folder with the two files>
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {"gpt-4o": "gpt-4o-airline.json", "claude-3.5-sonnet": "sonnet-35-new-airline.json"}


def source_calls(run: dict) -> list[tuple[str, dict]]:
    out = []
    for m in run["traj"]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                a = tc["function"].get("arguments") or {}
                out.append((tc["function"]["name"], json.loads(a) if isinstance(a, str) else a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-dir", required=True, help="folder with gpt-4o-airline.json and sonnet-35-new-airline.json")
    ap.add_argument("--files", nargs="+", default=["w1_tau_traces.jsonl", "w1_eks5_traces.jsonl"])
    a = ap.parse_args()

    runs = {}
    for model, name in SOURCES.items():
        for r in json.loads((Path(a.tau_dir) / name).read_text(encoding="utf-8")):
            runs[(r["task_id"], model, r["trial"])] = r

    for name in a.files:
        path = HERE / name
        traces = [json.loads(line) for line in open(path, encoding="utf-8")]
        filled = mismatched = 0
        for tr in traces:
            src = source_calls(runs[(tr["task_id"], tr["source_model"], tr["trial"])])
            if [t for t, _ in src] != [c["tool"] for c in tr["calls"]]:
                mismatched += 1
                print(f"  {name}: task {tr['task_id']} {tr['source_model']} trial {tr['trial']}: "
                      f"tool sequence differs from the source run, left unchanged")
                continue
            for c, (_, args) in zip(tr["calls"], src):
                filled += c["args"] != args
                c["args"] = args
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for tr in traces:
                f.write(json.dumps(tr) + "\n")
        print(f"{name}: {len(traces)} traces, {filled} calls got their arguments, {mismatched} traces left unchanged")


if __name__ == "__main__":
    main()
