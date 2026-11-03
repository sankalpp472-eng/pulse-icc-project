"""
Step 3: score the LLM speculator against the n-gram (and lr / gbdt, if a held-out predictor.pkl
exists) on the SAME sampled steps, with the v2 predictor's own scoring (evaluate.accuracy,
evaluate.theta_sweep).

Accuracy alone flatters the LLM, so each method also gets its prediction latency and two timing
scores. The guess is asked for at hint time; the tool call comes --lead-s later (~1.0 s measured in
our C3 runs); a cold start takes --cold-s (~1.8 s on our cluster).
  timely top3 : tool steps with the tool in the top 3 AND the guess back before the call
  cold hidden : share of the cold start hidden, min(lead - latency, cold) / cold, averaged over tool
                steps (0 for misses). Even an instant predictor hides only lead/cold = 56% of it.

    python baselines/llm_speculator/score.py results/<folder> [results/<folder2> ...]
    python baselines/llm_speculator/score.py results/<folder> --lead-s 1.0

LLM answers that could not be parsed count as misses (reported as parse failures).
Unlisted actions share the leftover probability mass, kept below the smallest listed p, so the
log-loss is finite; the ranking is the LLM's own.
Writes results/score_<time>.json
"""

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path

from common import DATA, HERE, PREDICTOR_DIR, RESPOND, read_jsonl, step_key
from evaluate import accuracy, latency, predict_all, theta_sweep
from models import NGramModel
from prompt import domain_actions

THETAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else None


def llm_distribution(pred, names):
    """Listed actions keep their p; the rest share what is left, each below the smallest listed p."""
    if not pred:
        return {}
    listed = {a: max(p, 1e-6) for a, p in pred.items()}
    rest = [n for n in names if n not in listed]
    left = max(0.0, 1 - sum(listed.values()))
    each = min(left / max(len(rest), 1), 0.5 * min(listed.values())) if rest else 0
    s = sum(listed.values()) + each * len(rest)
    return {**{a: p / s for a, p in listed.items()}, **{a: max(each, 1e-9) / s for a in rest}}


def timely(P, steps, lat_s, lead_s, k=3):
    """Tool steps whose tool is in the top-k (tools only) AND whose guess arrived within the lead time."""
    n = hit = 0
    for p, s, l in zip(P, steps, lat_s):
        if s["action"] == RESPOND:
            continue
        n += 1
        top = [a for a, _ in sorted(((a, v) for a, v in p.items() if a != RESPOND), key=lambda kv: -kv[1])[:k]]
        hit += s["action"] in top and l is not None and l <= lead_s
    return hit / max(n, 1)


def cold_hidden(P, steps, lat_s, lead_s, cold_s, k=3):
    """Share of cold-start time hidden, averaged over tool steps: if the tool is in the top-k, the pod
    starts `lat` after the hint and the call comes `lead_s` after it, so min(lead - lat, cold) is hidden."""
    n, hidden = 0, 0.0
    for p, s, l in zip(P, steps, lat_s):
        if s["action"] == RESPOND:
            continue
        n += 1
        top = [a for a, _ in sorted(((a, v) for a, v in p.items() if a != RESPOND), key=lambda kv: -kv[1])[:k]]
        if s["action"] in top and l is not None:
            hidden += min(max(lead_s - l, 0.0), cold_s) / cold_s
    return hidden / max(n, 1)


def report_row(name, P, steps, lat_s, lead_s, cold_s, thetas, max_replicas, extra=None):
    acc = accuracy(P, steps)
    known = [l for l in lat_s if l is not None]
    return {"method": name, **acc,
            "latency_p50_s": pct(known, 0.5), "latency_p95_s": pct(known, 0.95),
            "in_time_share": sum(l <= lead_s for l in known) / max(len(lat_s), 1),
            "timely_tool_top1": timely(P, steps, lat_s, lead_s, 1),
            "timely_tool_top3": timely(P, steps, lat_s, lead_s, 3),
            "cold_hidden_top3": cold_hidden(P, steps, lat_s, lead_s, cold_s, 3),
            "theta_sweep": theta_sweep(P, steps, thetas, max_replicas), **(extra or {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="speculate.py result folders (same sample)")
    p.add_argument("--lead-s", type=float, default=1.0, help="hint-to-call lead time (s)")
    p.add_argument("--cold-s", type=float, default=1.8, help="cold start of a tool pod (s)")
    p.add_argument("--max-replicas", type=int, default=4)
    a = p.parse_args()

    run_dirs = [Path(r) if Path(r).exists() else HERE / r for r in a.runs]
    cfgs = [json.loads((d / "run.json").read_text()) for d in run_dirs]
    samples = {c["sample"] for c in cfgs}
    if len(samples) != 1:
        raise SystemExit(f"runs use different samples: {samples}")
    dom = cfgs[0]["domain"]
    names = [n for n, _ in domain_actions(dom)]
    all_test = read_jsonl(DATA / f"{dom}_test_steps.jsonl")
    by_key = {step_key(s): s for s in all_test}

    # score every method on the steps EVERY run answered (resumed runs may still be partial)
    preds = [{r["key"]: r for r in read_jsonl(d / "predictions.jsonl")} for d in run_dirs]
    keys = [k for k in json.loads(Path(samples.pop()).read_text())["keys"] if all(k in pr for pr in preds)]
    steps = [by_key[k] for k in keys]
    print(f"{dom}: scoring {len(steps)} steps ({sum(s['action'] != RESPOND for s in steps)} tool steps) "
          f"of {len(all_test)} held-out; lead time {a.lead_s} s")

    rows = []
    ngram = NGramModel().fit(read_jsonl(DATA / f"{dom}_train_steps.jsonl"))
    local = {"ngram": ngram}
    pkl = PREDICTOR_DIR / "data" / "predictor.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            pred = pickle.load(f)
        if pred.meta.get("final"):
            print(f"  skipping {pkl}: trained with --final (saw the test tasks)")
        elif dom in pred.models:
            local.update({n: m for n, m in pred.models[dom].items() if n != "ngram"})
    for name, m in local.items():
        lat = latency(m, steps)
        rows.append(report_row(name, predict_all(m, steps), steps, [lat["p95_ms"] / 1000] * len(steps),
                               a.lead_s, a.cold_s, THETAS, a.max_replicas))
        rows[-1]["latency_p50_s"] = lat["p50_ms"] / 1000

    for d, cfg, pr in zip(run_dirs, cfgs, preds):
        rs = [pr[k] for k in keys]
        P = [llm_distribution(r["pred"], names) for r in rs]
        rows.append(report_row(f"llm:{cfg['model']}", P, steps, [r.get("latency_s") for r in rs], a.lead_s,
                               a.cold_s, THETAS, a.max_replicas,
                               {"run": d.name, "parse_fail": sum(not r["ok"] for r in rs) / len(rs),
                                "prompt_tokens_mean": statistics.mean(r.get("prompt_tokens") or 0 for r in rs)}))

    full = accuracy(predict_all(ngram, all_test), all_test)
    print(f"  (n-gram on all {len(all_test)} held-out steps: tool top-1 {full['tool_top1']:.3f}, "
          f"top-3 {full['tool_top3']:.3f}; checks the sample is representative)\n")
    print(f"  {'method':32} {'tool top1':>9} {'top3':>6} {'all top1':>8} {'p50 s':>7} {'p95 s':>7} "
          f"{'in time':>7} {'timely top3':>11} {'cold hidden':>11}")
    def sec(x):
        return f"{x:7.3f}" if x is not None else f"{'-':>7}"
    for r in rows:
        print(f"  {r['method']:32} {r['tool_top1']:9.3f} {r['tool_top3']:6.3f} {r['all_top1']:8.3f} "
              f"{sec(r['latency_p50_s'])} {sec(r['latency_p95_s'])} {r['in_time_share']:7.2f} "
              f"{r['timely_tool_top3']:11.3f} {r['cold_hidden_top3']:11.3f}" + (f"   parse fail {r['parse_fail']:.2f}" if "parse_fail" in r else ""))

    out = HERE / "results" / f"score_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"domain": dom, "steps": len(steps), "lead_s": a.lead_s, "cold_s": a.cold_s, "keys": keys,
                               "ngram_full_test": full, "methods": rows}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
