"""
Step 1: the steps the speculator is asked about, plus a seeded sample that fits the Groq quota.

Same tau-bench runs, de-duplication and task split as the v2 predictor's tau_data.py (seed 252,
20% of TASKS held out), so every test step here is a step the n-gram / lr / gbdt models are also
scored on. Unlike examples.jsonl, this keeps the conversation text: the speculator reads it, the
feature models never do.

    python baselines/llm_speculator/steps.py                       # airline, 300-step sample
    python baselines/llm_speculator/steps.py --sample 100 --sample-seed 1

Writes (data/ is gitignored):
    data/<domain>_train_steps.jsonl  trajectory + features + action      (to fit the n-gram here)
    data/<domain>_test_runs.jsonl    one held-out conversation per line  (system prompt dropped)
    data/<domain>_test_steps.jsonl   one held-out LLM step per line: run_id, step, msg_index, action...
    data/<domain>_sample_<n>_seed<s>.json   the step keys the speculator is run on
"""

import argparse
import hashlib
import json
import random

from common import DATA, PREDICTOR_DIR, RESPOND, step_key
from tau_data import SOURCES, _args, add_width, fetch, walk


def compact(m):
    """A tau-bench message without the fields we never show (system prompt is dropped by the caller)."""
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m["role"] == "assistant" and m.get("tool_calls"):
        out["calls"] = [[c["function"]["name"], _args(c)] for c in m["tool_calls"]]
    if m["role"] == "tool":
        out["name"] = m.get("name", "")
    return out


def load_runs(domain, tau_dir):
    """(model, run) pairs, de-duplicated exactly like tau_data.main."""
    seen, runs, dupes = set(), [], 0
    for fname in SOURCES[domain]:
        model = fname.split("-" + domain)[0]
        for r in fetch(fname, tau_dir):
            key = hashlib.sha1(json.dumps(r["traj"], sort_keys=True).encode()).hexdigest()
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            runs.append((model, r))
    return runs, dupes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="airline", choices=list(SOURCES))
    p.add_argument("--tau-dir", default=str(PREDICTOR_DIR / "data" / "tau"))
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=252, help="task split seed (must match tau_data.py)")
    p.add_argument("--sample", type=int, default=300, help="test steps to run the speculator on (0 = all)")
    p.add_argument("--sample-seed", type=int, default=252)
    a = p.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    runs, dupes = load_runs(a.domain, a.tau_dir)
    tasks = sorted({r["task_id"] for _, r in runs})
    random.Random(a.seed).shuffle(tasks)
    test_tasks = set(tasks[:max(1, round(len(tasks) * a.test_frac))])

    n_train = 0
    test_steps = []
    with open(DATA / f"{a.domain}_train_steps.jsonl", "w", encoding="utf-8") as ftr, \
         open(DATA / f"{a.domain}_test_runs.jsonl", "w", encoding="utf-8") as fruns:
        for model, r in runs:
            run_id = f"{a.domain}/{model}/task{r['task_id']}/trial{r.get('trial', 0)}"
            steps = add_width(list(walk(a.domain, r["traj"])))
            if r["task_id"] not in test_tasks:
                for i, s in enumerate(steps):
                    ftr.write(json.dumps({"run_id": run_id, "task_id": r["task_id"], "step": i,
                                          "trajectory": s["trajectory"], "features": s["features"],
                                          "action": s["action"]}) + "\n")
                    n_train += 1
                continue
            msgs = [compact(m) for m in r["traj"] if m["role"] != "system"]
            fruns.write(json.dumps({"run_id": run_id, "task_id": r["task_id"], "source_model": model,
                                    "messages": msgs}) + "\n")
            asst = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
            assert len(asst) == len(steps), run_id
            for i, (j, s) in enumerate(zip(asst, steps)):
                assert s["action"] == (msgs[j]["calls"][0][0] if msgs[j].get("calls") else RESPOND)
                test_steps.append({"run_id": run_id, "task_id": r["task_id"], "source_model": model,
                                   "step": i, "msg_index": j, "action": s["action"], "width": s["width"],
                                   "trajectory": s["trajectory"], "features": s["features"]})

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
    print(f"{a.domain}: {len(runs)} runs ({dupes} duplicates skipped), {len(tasks)} tasks, "
          f"test tasks {sorted(test_tasks)}")
    print(f"  train steps {n_train}, test steps {len(test_steps)} ({n_tool} tool steps)")
    print(f"  sample: {n} steps -> {sample_path}")


if __name__ == "__main__":
    main()
