"""
W1 C0 baseline, LIVE: the real agent (Groq) runs tau-bench airline change/cancel tasks
against the 14 MCP tools on Knative (scale-to-zero, no prewarming).

Every tool call is recorded and labelled cold or warm:
  pod signal     -- a new pod for that tool appeared during the call (background pod watcher)
  latency signal -- the call took > 1 s (warm calls take ~50-100 ms, cold ones seconds)
A call is cold if either signal says so; both are kept so they can be compared.

Before every episode the shared database (Redis) is reset to the original airline data,
so every episode starts from the same state. Runs until all tasks are done or Groq's
quota runs out (then stops cleanly, data kept).
Run with the agent's Python, from the project root:
  agent/.venv/Scripts/python test/test_knative/w1_live.py                 # all 23 tasks (safe first)
  agent/.venv/Scripts/python test/test_knative/w1_live.py --set safe      # 12 one-write tasks only
  agent/.venv/Scripts/python test/test_knative/w1_live.py --tasks 1 6     # specific tasks
  ... --agent-model openai/gpt-oss-120b --user-model openai/gpt-oss-20b    # choose the Groq models

Output: results/local_runs/ (kind) or results/cloud_runs/ (EKS), <date>_<time>_<config>_W1-live_<tasks>_<model>/  (e.g. 2026-10-21_1527_C0_W1-live_task9_gpt-oss-120b)
  run_config.json models, Knative settings (read from the cluster), image, database, git commit ...
  calls.csv       one row per tool call (cold/warm, latency, idle gap, think time ...)
  episodes.jsonl  one agent episode per line (full trace: steps, tool sequence, timings)
  pods.csv        every tool pod seen (tool, created, first/last seen)
  summary.json    headline numbers
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[1] / "agent"
sys.path.insert(0, str(AGENT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# W1 = tau-bench airline "change / cancel a booking" tasks (23), split by how they write
SETS = {
    "safe": [1, 6, 7, 14, 20, 27, 31, 43, 2, 22, 28, 30],   # one write, or repeated writes with one tool
    "multi": [3, 4, 5, 8, 9, 10, 19, 23, 26, 33, 34],      # writes with different tools (need the shared DB)
}
SETS["all"] = SETS["safe"] + SETS["multi"]
# The 5 tasks with the most branching that together cover 13 of the 14 tools (chosen from the
# 276 recorded tau-bench runs: 10-12 different tool sequences per task, median 6-10 calls).
SETS["best5"] = [4, 9, 10, 23, 27]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="all")
    ap.add_argument("--tasks", type=int, nargs="+", help="explicit task ids (overrides --set)")
    ap.add_argument("--repeats", type=int, default=1, help="run every task this many times")
    ap.add_argument("--backend", choices=["groq", "openrouter"], default=None,
                    help="LLM provider (default: LLM_BACKEND in agent/.env)")
    ap.add_argument("--agent-model", default="openai/gpt-oss-20b", help="model for the agent (same ids on Groq and OpenRouter)")
    ap.add_argument("--user-model", default="openai/gpt-oss-20b", help="model for the simulated customer")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the conversation")
    ap.add_argument("--label", default="", help="extra text at the end of the result folder name (e.g. eks)")
    ap.add_argument("--predictor-url", help="C3: the agent sends hints to this predictor (e.g. http://127.0.0.1:8000); "
                                            "the prewarm controller must be running")
    ap.add_argument("--workflow-id", default="W1", help="workflow id sent in hints")
    ap.add_argument("--user-delay-median", type=float, default=8.0,
                    help="median seconds the simulated customer takes to read and answer (lognormal, as a "
                         "customer turn in w1_replay.py); 0 = answer as fast as the LLM (~0.75 s)")
    ap.add_argument("--user-delay-sigma", type=float, default=0.5)
    ap.add_argument("--user-delay-seed", type=int, default=252)
    return ap.parse_args()


# Parsed before importing the agent: llm.py reads the model from the environment at import
ARGS = parse_args()
if ARGS.backend:
    os.environ["LLM_BACKEND"] = ARGS.backend      # set before llm.py loads .env (which won't override it)
os.environ["GROQ_MODEL"] = os.environ["OPENROUTER_MODEL"] = ARGS.agent_model
os.environ["USER_MODEL"] = ARGS.user_model
# user.py reads these at import: the simulated customer's reading + typing time
os.environ["USER_DELAY_MEDIAN_S"] = str(ARGS.user_delay_median)
os.environ["USER_DELAY_SIGMA"] = str(ARGS.user_delay_sigma)
os.environ["USER_DELAY_SEED"] = str(ARGS.user_delay_seed)
os.environ.setdefault("TOOL_BACKEND", "mcp")
os.environ.setdefault("MCP_URL_TEMPLATE", "http://127.0.0.1:8081/mcp")
os.environ.setdefault("MCP_HOST_TEMPLATE", "{svc}.default.example.com")
# hints.py reads these at import: without --predictor-url no hints are sent (C0)
os.environ["PREDICTOR_URL"] = ARGS.predictor_url or ""
os.environ["WORKFLOW_ID"] = ARGS.workflow_id

import hints  # noqa: E402
import llm  # noqa: E402
from agent import run_episode  # noqa: E402
from tau_airline import load_tasks  # noqa: E402
from tools import CALL_LOG  # noqa: E402
from c0_common import (COLD_LATENCY_S, PodWatcher, peak_pods, reset_db, run_config, run_dir,  # noqa: E402
                       save_predictor_health,
                       shared_db, tasks_tag, wait_all_zero)

CALL_FIELDS = ["episode", "task_id", "task_set", "call_idx", "tool", "t_start", "latency_ms", "cold",
               "cold_by_pod", "cold_by_latency", "pod", "idle_gap_s", "think_s", "rate_wait_s",
               "transport_ok", "tool_error"]


def think_times(ep: dict | None) -> list[tuple[float, float]]:
    """(think_s, rate_wait_s) per tool call, in call order, from the agent's step records."""
    out = []
    for s in (ep or {}).get("steps", []):
        for k, _ in enumerate(s.get("calls", [])):   # several calls in one step share its think time
            out.append((s["think_s"] if k == 0 else 0.0, s["rate_wait_s"] if k == 0 else 0.0))
    return out


def main():
    a = ARGS
    tasks = {t["id"]: t for t in load_tasks()}
    ids = (a.tasks or SETS[a.set]) * a.repeats
    set_of = {i: name for name in ("safe", "multi") for i in SETS[name]}
    n_trimmed = 0
    par_widths: list[int] = []          # calls per reply that ran at once (parallel fan-out)
    print(f"W1 live {'C3' if a.predictor_url else 'C0'} | tasks {ids} | agent {llm.DEFAULT_MODEL}, user sim {os.environ['USER_MODEL']} "
          f"| {llm.BACKEND}, {len(llm.KEYS)} key(s)")

    db = "redis (shared, reset per episode)" if shared_db() else "in-memory per pod (no shared DB!)"
    cfg = run_config(ids, db, "W1 airline change/cancel (tau-bench), live agent",
                     {"backend": llm.BACKEND, "agent": llm.DEFAULT_MODEL, "user_simulator": os.environ["USER_MODEL"],
                      "api_keys_available": len(llm.KEYS)},
                     {"db_reset_per_episode": db.startswith("redis"), "repeats": a.repeats,
                      "context_budget_tokens": llm.CONTEXT_BUDGET,
                      "context_trimming": "sent copy only, oldest first, newest 4 messages kept: old tool results "
                                          "compacted, then replaced by a neutral note, then old long messages cut "
                                          "to 300 chars; saved traces are complete",
                      "fixed_prompt_tokens_estimate": llm.FIXED_TOKENS,
                      "customer_reply_delay": {"median_s": a.user_delay_median, "sigma": a.user_delay_sigma,
                                               "seed": a.user_delay_seed, "clip_s": [1, 60],
                                               "rule": "each simulated reply takes max(LLM time, delay)"}})
    if a.predictor_url:   # scale-to-zero tools + predictor + prewarm controller = C3 (same as w1_replay.py)
        cfg["config"] = "C3" if cfg["config"] == "C0" else f"{cfg['config']}+prewarm"
        try:
            with urllib.request.urlopen(a.predictor_url + "/healthz", timeout=5) as r:
                health = json.loads(r.read())
        except Exception as e:
            raise SystemExit(f"Predictor not reachable at {a.predictor_url}: {e}")
        if health.get("crd", {}).get("mode") != "k8s":
            print("WARNING: predictor is not in CRD_MODE=k8s -> no PrewarmRequests, nothing gets prewarmed")
        cfg["prewarm"] = {"predictor_url": a.predictor_url, "workflow_id": a.workflow_id,
                          "predictor": {k: health.get(k) for k in ("model", "action_models", "theta", "ttl_s")},
                          "crd_mode": health.get("crd", {}).get("mode"), "hint_signals": hints.USE_SIGNALS,
                          "mechanism": "Option A: controller sends an MCP ping per PrewarmRequest; "
                                       "TTL = Knative idle window",
                          "hints": "sent by the agent before every LLM step (agent/hints.py)"}
    detail = (tasks_tag(ids) if a.tasks else a.set) + (f"-x{a.repeats}" if a.repeats > 1 else "")
    out_dir = run_dir(cfg["config"], "live", f"{detail}_{llm.DEFAULT_MODEL.split('/')[-1]}", a.label)
    print(f"-> {out_dir}")
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    hints.HINT_LOG = str(out_dir / "hints.jsonl")   # every hint + the predictor's answer, kept with the run
    print(f"config: {cfg['config']} | window {cfg['knative']['tool_window']} | images {cfg['tool_images']} | db: {db}")
    watcher = PodWatcher()
    watcher.start()
    wait_all_zero(watcher)
    watcher.forget()                    # pods from before the run (e.g. an old revision) are not part of it
    print("all 14 tools at zero; starting\n")

    calls_f = open(out_dir / "calls.csv", "w", newline="", encoding="utf-8")
    calls_w = csv.DictWriter(calls_f, fieldnames=CALL_FIELDS)
    calls_w.writeheader()
    last_end: dict[str, float] = {}   # tool -> end of its previous call (for idle gaps)
    all_calls, stop_reason = [], "all tasks done"

    for n, tid in enumerate(ids, 1):
        CALL_LOG.clear()
        episode = f"w1-task{tid}-{int(time.time())}"
        print(f"[{n}/{len(ids)}] task {tid} ({set_of.get(tid, '?')})")
        ep, status = None, "ok"
        if db.startswith("redis"):
            reset_db()
        try:
            ep = run_episode(tasks[tid], user="sim", verbose=a.verbose)
        except llm.QuotaExhausted as e:
            status, stop_reason = "quota_exhausted", f"Groq quota: {e}"
        except Exception as e:  # keep going with the next task
            status = f"error: {type(e).__name__}: {e}"

        tt = think_times(ep)
        n_trimmed += sum(s.get("context_trimmed", 0) for s in (ep or {}).get("steps", []))
        par_widths += [len(s["calls"]) for s in (ep or {}).get("steps", []) if s.get("parallel")]
        for i, c in enumerate(CALL_LOG):
            t_end = c["t_start"] + c["latency_s"]
            # a new pod counts only if the tool had none before; else it may be extra capacity for parallel calls
            pod = not watcher.had_pod_before(c["tool"], c["t_start"]) and \
                watcher.new_pod_during(c["tool"], c["t_start"], t_end)
            by_lat = c["latency_s"] > COLD_LATENCY_S
            prev = last_end.get(c["tool"])
            row = {"episode": episode, "task_id": tid, "task_set": set_of.get(tid, "?"), "call_idx": i,
                   "tool": c["tool"], "t_start": round(c["t_start"], 3), "latency_ms": round(c["latency_s"] * 1000, 1),
                   "cold": bool(pod) or by_lat, "cold_by_pod": bool(pod), "cold_by_latency": by_lat, "pod": pod or "",
                   "idle_gap_s": round(c["t_start"] - prev, 1) if prev else "",
                   "think_s": tt[i][0] if i < len(tt) else "", "rate_wait_s": tt[i][1] if i < len(tt) else "",
                   "transport_ok": c["transport_ok"], "tool_error": c["tool_error"]}
            last_end[c["tool"]] = t_end
            calls_w.writerow(row)
            all_calls.append(row)
        calls_f.flush()

        with open(out_dir / "episodes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"episode": episode, "task_id": tid, "task_set": set_of.get(tid, "?"),
                                "status": status, "model": llm.DEFAULT_MODEL, "user_model": os.environ["USER_MODEL"],
                                "calls": len(CALL_LOG), **(ep or {})}) + "\n")
        cold = sum(r["cold"] for r in all_calls[-len(CALL_LOG):]) if CALL_LOG else 0
        print(f"    {status} | {len(CALL_LOG)} tool calls, {cold} cold | "
              f"{' -> '.join(c['tool'] for c in CALL_LOG) or '(no tool calls)'}")
        if status == "quota_exhausted":
            break

    watcher.stop_flag = True
    calls_f.close()
    with open(out_dir / "pods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pod", "tool", "created", "first_seen", "last_seen", "deleting_seen"])
        w.writeheader()
        w.writerows(watcher.pods.values())

    ok = [r for r in all_calls if r["transport_ok"]]
    cold = [r["latency_ms"] for r in ok if r["cold"]]
    warm = [r["latency_ms"] for r in ok if not r["cold"]]
    per_tool = {}
    for r in ok:
        d = per_tool.setdefault(r["tool"], {"calls": 0, "cold": 0})
        d["calls"] += 1
        d["cold"] += r["cold"]
    summary = {
        "stop_reason": stop_reason, "episodes": len({r['episode'] for r in all_calls}),
        "tool_calls": len(all_calls), "transport_failures": len(all_calls) - len(ok),
        "cold_calls": len(cold), "cold_share": round(len(cold) / len(ok), 3) if ok else None,
        "cold_p50_ms": statistics.median(cold) if cold else None,
        "warm_p50_ms": statistics.median(warm) if warm else None,
        "all_calls_p50_ms": statistics.median([r["latency_ms"] for r in ok]) if ok else None,
        "signals_agree": sum(r["cold_by_pod"] == r["cold_by_latency"] for r in ok),
        "per_tool": per_tool, "agent_model": llm.DEFAULT_MODEL, "user_model": os.environ["USER_MODEL"], "database": db,
        "config": cfg["config"], "tool_window": cfg["knative"]["tool_window"], "tool_images": cfg["tool_images"],
        "api_keys_available": len(llm.KEYS), "key_switches": llm.KEY_LOG,
        "tool_results_trimmed_from_context": n_trimmed,
        "parallel_steps": {"replies": len(par_widths), "calls": sum(par_widths), "widths": par_widths,
                           "peak_pods_per_tool": peak_pods(watcher.pod_rows())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_predictor_health(a.predictor_url, out_dir)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_tool"}, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
