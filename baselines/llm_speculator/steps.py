"""
Step 1: the steps the speculator is asked about, plus a seeded sample that fits the Groq quota.

Same tau-bench runs, de-duplication and task split as predictor/build_action_data.py --split speculator
(seed 252, 20% of TASKS held out), so every test step here is a row of airline_specsplit_action_rows.csv
and is scored against the same gbt / lr / ngram models. Unlike those rows, this keeps the conversation
text: the speculator reads it, the feature models never do.

(It used to import the v2 predictor's tau_data.py, which was never committed; build_action_data.py
already re-implements that split, so this file now uses it and needs nothing outside the repo.)

    python baselines/llm_speculator/steps.py --tau-bench ../tau-bench                # airline, 300-step sample
    python baselines/llm_speculator/steps.py --tau-bench ../tau-bench --sample 100 --sample-seed 1

Writes (data/ is gitignored):
    data/<domain>_test_runs.jsonl    one held-out conversation per line  (system prompt dropped)
    data/<domain>_test_steps.jsonl   one held-out LLM step per line: run_id, step, msg_index, action...
    data/<domain>_sample_<n>_seed<s>.json   the step keys the speculator is run on
"""

import argparse
import json
import random
import sys
from pathlib import Path

from common import DATA, RESPOND, ROOT, step_key

sys.path.insert(0, str(ROOT / "predictor"))
from build_action_data import FILES, speculator_split, tool_calls  # noqa: E402


def compact(m):
    """A tau-bench message without the fields we never show (system prompt is dropped by the caller)."""
    out = {"role": m["role"], "content": m.get("content") or ""}
    calls = tool_calls(m) if m["role"] == "assistant" else []
    if calls:
        out["calls"] = [[n, a] for n, a in calls]
    if m["role"] == "tool":
        out["name"] = m.get("name", "")
    return out


def run_steps(msgs):
    """One record per assistant message: action + its arguments, tools called before it, fan-out width."""
    steps, called = [], []
    for j, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        calls = m.get("calls") or []
        steps.append({"msg_index": j, "action": calls[0][0] if calls else RESPOND,
                      "args": calls[0][1] if calls else {}, "trajectory": list(called)})
        called += [n for n, _ in calls]
    for i, s in enumerate(steps):       # width: how many steps in a row call the same tool (0 for RESPOND)
        k = 0
        while s["action"] != RESPOND and i + k < len(steps) and steps[i + k]["action"] == s["action"]:
            k += 1
        s["width"] = k
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="airline", choices=list(FILES))
    p.add_argument("--tau-bench", required=True, type=Path, help="clone of sierra-research/tau-bench")
    p.add_argument("--seed", type=int, default=252, help="task split seed (as build_action_data.py)")
    p.add_argument("--sample", type=int, default=300, help="test steps to run the speculator on (0 = all)")
    p.add_argument("--sample-seed", type=int, default=252)
    a = p.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    runs = [(fname, i, ep) for fname in FILES[a.domain]
            for i, ep in enumerate(json.loads((a.tau_bench / "historical_trajectories" / fname)
                                              .read_text(encoding="utf-8")))]
    kept, test_tasks = speculator_split(runs, a.seed)

    test_steps, n_runs = [], 0
    with open(DATA / f"{a.domain}_test_runs.jsonl", "w", encoding="utf-8") as fruns:
        for fname, _, r in kept:
            if r["task_id"] not in test_tasks:
                continue
            model = fname.split("-" + a.domain)[0]
            run_id = f"{a.domain}/{model}/task{r['task_id']}/trial{r.get('trial', 0)}"
            msgs = [compact(m) for m in r["traj"] if m["role"] != "system"]
            fruns.write(json.dumps({"run_id": run_id, "task_id": r["task_id"], "source_model": model,
                                    "messages": msgs}) + "\n")
            n_runs += 1
            for i, s in enumerate(run_steps(msgs)):
                test_steps.append({"run_id": run_id, "task_id": r["task_id"], "source_model": model, "step": i, **s})

    with open(DATA / f"{a.domain}_test_steps.jsonl", "w", encoding="utf-8") as f:
        for s in test_steps:
            f.write(json.dumps(s) + "\n")

    keys = [step_key(s) for s in test_steps]
    if a.sample and a.sample < len(keys):
        keys = sorted(random.Random(a.sample_seed).sample(keys, a.sample))
    n = len(keys)
    sample_path = DATA / f"{a.domain}_sample_{n}_seed{a.sample_seed}.json"
    sample_path.write_text(json.dumps({"domain": a.domain, "split_seed": a.seed, "sample_seed": a.sample_seed,
                                       "of_steps": len(test_steps), "keys": keys}, indent=1))

    n_tool = sum(s["action"] != RESPOND for s in test_steps)
    print(f"{a.domain}: {len(kept)} runs ({len(runs) - len(kept)} duplicates skipped), "
          f"test tasks {sorted(test_tasks)}")
    print(f"  test runs {n_runs}, test steps {len(test_steps)} ({n_tool} tool steps)")
    print(f"  sample: {n} steps -> {sample_path}")


if __name__ == "__main__":
    main()
