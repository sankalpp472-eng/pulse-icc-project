"""
W1 C0 baseline, REPLAY: real tau-bench agent traces replayed against the 14 MCP tools on Knative.

No LLM is called. Each episode is one recorded agent run (GPT-4o or Claude 3.5 Sonnet on one
of the 23 W1 change/cancel tasks; data/w1_tau_traces.jsonl) and replays its tool calls, with
their recorded arguments, through the same MCP client the agent uses. Between calls it waits
  think time (lognormal, median --think-median s)          -- the LLM deciding the next call
  + one customer turn per recorded agent->customer message  -- the customer answering
Episodes start at random times (Poisson, --rate per minute), so several overlap, as with
many users. The --seed fixes the trace order, start times and gaps: the same seed gives the
same request sequence on C0, C1 or C3 (FR-10).

Every call is labelled cold/warm afterwards (latency > 1 s, or a new pod for that tool appeared
during the call) and gets its idle gap (time since that tool's previous call ended).

Run with the agent's Python, from the project root:
  agent/.venv/Scripts/python test/test_knative/w1_replay.py                       # 46 episodes, 2/min
  agent/.venv/Scripts/python test/test_knative/w1_replay.py --episodes 92 --rate 4
  # replay OUR agent's live runs (w1_live result folders), with their real recorded gaps:
  agent/.venv/Scripts/python test/test_knative/w1_replay.py --traces test/test_knative/results/<date>_C0_W1-live_<...> ...

Output: test/test_knative/results/<date>_<time>_<config>_W1-replay_<traces>/  (e.g. 2026-10-22_1336_C3_W1-replay_task9-x3)
  run_config.json  replay parameters + Knative/cluster settings (read from the cluster), code version
  calls.csv        one row per tool call
  episodes.jsonl   one row per episode (trace source, planned gaps, timings)
  pods.csv         every tool pod seen (tool, created, first/last seen)
  summary.json     cold share, latency percentiles, per-tool cold share, pod-seconds
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "agent"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("TOOL_BACKEND", "mcp")
os.environ.setdefault("MCP_URL_TEMPLATE", "http://127.0.0.1:8081/mcp")
os.environ.setdefault("MCP_HOST_TEMPLATE", "{svc}.default.example.com")

from c0_common import (COLD_LATENCY_S, PodWatcher, reset_db, run_config, run_dir,  # noqa: E402
                       shared_db, tasks_tag, wait_all_ready, wait_all_zero)
from tools import MCPToolBackend  # noqa: E402

TRACES = HERE / "data" / "w1_tau_traces.jsonl"
CALL_FIELDS = ["episode", "trace", "task_id", "task_set", "source_model", "call_idx", "tool", "t_start",
               "latency_ms", "cold", "cold_by_pod", "cold_by_latency", "pod", "idle_gap_s", "planned_gap_s",
               "user_turns_before", "concurrent_episodes", "transport_ok", "tool_error"]


def lognormal(rng: random.Random, median: float, sigma: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, rng.lognormvariate(math.log(median), sigma)))


def live_traces(run_dir: Path) -> list[dict]:
    """Our agent's episodes from a w1_live result folder, in the replay trace format.
    Gap before a call = agent think time + for every customer turn since the previous call:
    the agent's reply time + the customer's reply time (user_s; sampled later if not recorded)."""
    cfg = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
    model = cfg.get("models", {}).get("agent", "our agent")
    out = []
    for n, line in enumerate(open(run_dir / "episodes.jsonl", encoding="utf-8")):
        ep = json.loads(line)
        # hints: the live agent sent a hint at the START of every LLM step (agent.emit_hint), including
        # steps where it only answered the customer -> keep their offsets within the gap before each call
        calls, known, unknown_turns, turns, hints = [], 0.0, 0, 0, []
        for st in ep.get("steps") or []:
            if st["kind"] == "respond":
                hints.append(round(known, 3))
                known += st["think_s"]
                turns += 1
                if "user_s" in st:
                    known += st["user_s"]
                else:
                    unknown_turns += 1
                continue
            for k, c in enumerate(st.get("calls", [])):
                if k == 0:
                    hints.append(round(known, 3))
                calls.append({"tool": c["tool"], "args": c.get("args", {}), "user_turns_before": turns if k == 0 else 0,
                              "recorded_gap_s": round(known + (st["think_s"] if k == 0 else 0.05), 3),
                              "unknown_user_turns": unknown_turns if k == 0 else 0,
                              "hint_offsets_s": hints})
                known, unknown_turns, turns, hints = 0.0, 0, 0, []
        if calls:
            out.append({"task_id": ep["task_id"], "task_set": ep.get("task_set", "?"),
                        "source_model": f"{model} (our agent, {run_dir.name})", "trial": n,
                        "reward": ep.get("reward"), "status": ep.get("status"), "calls": calls})
    return out


def load_traces(paths: list[str] | None) -> list[dict]:
    if not paths:
        return [json.loads(l) for l in open(TRACES, encoding="utf-8")]
    out = []
    for p in map(Path, paths):
        out += live_traces(p) if p.is_dir() else [json.loads(l) for l in open(p, encoding="utf-8")]
    return out


def plan(a) -> list[dict]:
    """Episodes to run: which trace, when it starts, and the gap before every call (all from the seed)."""
    rng = random.Random(a.seed)
    traces = [t for t in load_traces(a.traces) if t["calls"] and (not a.tasks or t["task_id"] in a.tasks)]
    if not traces:
        raise SystemExit("No traces to replay (check --traces / --tasks)")
    traces = traces * a.repeat

    def hints_for(tr):
        """When to send hints within the gap before each call (seconds from the gap's start)."""
        return [[o * a.gap_scale for o in c["hint_offsets_s"]] if "hint_offsets_s" in c else [0.0]
                for c in tr["calls"]]

    def gaps_for(tr):
        out = []
        for c in tr["calls"]:
            if "recorded_gap_s" in c:      # our agent: real timing, scaled; sample only missing customer turns
                g = c["recorded_gap_s"] * a.gap_scale
                turns = c.get("unknown_user_turns", 0)
            else:                          # tau-bench recordings carry no timing: sample think time
                g = lognormal(rng, a.think_median, a.think_sigma, 0.2, 15.0)
                turns = c["user_turns_before"]
            out.append(g + sum(lognormal(rng, a.user_turn_median, a.user_turn_sigma, 1.0, 60.0) for _ in range(turns)))
        return out

    if a.episodes is None and a.traces:    # our own traces: replay every one once, in a seeded order
        order = traces[:]
        rng.shuffle(order)
        episodes, t = [], 0.0
        for tr in order:
            t += rng.expovariate(a.rate / 60.0)
            episodes.append({"episode": f"ep{len(episodes) + 1:03d}", "start_offset_s": round(t, 2), "trace": tr,
                             "gaps": [round(g, 2) for g in gaps_for(tr)], "hints": hints_for(tr)})
        return episodes
    a.episodes = a.episodes or 46
    by_task: dict[int, list] = {}
    for t in traces:
        by_task.setdefault(t["task_id"], []).append(t)
    # cover the 23 tasks evenly: round-robin over shuffled tasks, a random recorded run each time
    tasks = sorted(by_task)
    episodes, t = [], 0.0
    while len(episodes) < a.episodes:
        rng.shuffle(tasks)
        for tid in tasks:
            if len(episodes) == a.episodes:
                break
            tr = rng.choice(by_task[tid])
            gaps = gaps_for(tr)
            t += rng.expovariate(a.rate / 60.0)
            episodes.append({"episode": f"ep{len(episodes) + 1:03d}", "start_offset_s": round(t, 2), "trace": tr,
                             "gaps": [round(g, 2) for g in gaps], "hints": hints_for(tr)})
    return episodes


class Replay:
    def __init__(self, predictor_url: str | None = None, workflow_id: str = "W1", hint_log: Path | None = None):
        self.rows, self.lock, self.active = [], threading.Lock(), 0
        self.predictor_url, self.workflow_id, self.hint_log = predictor_url, workflow_id, hint_log
        self.hints_sent = self.hints_ok = 0

    def send_hint(self, episode: str, trajectory: list[str]) -> None:
        """What agent.emit_hint does: POST the trajectory to the predictor in the background (never waits)."""
        if not self.predictor_url:
            return
        hint = {"workflow_id": self.workflow_id, "episode_id": episode, "trajectory": trajectory[-8:],
                "current_tool": trajectory[-1] if trajectory else "START", "ts": time.time()}

        def post():
            rec, t0 = {"hint": hint}, time.perf_counter()
            try:
                req = urllib.request.Request(self.predictor_url + "/predict", data=json.dumps(hint).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    ans = json.loads(r.read())
                rec.update(ok=True, prewarm=ans["prewarm"], top3=ans["ranked"][:3], theta=ans["theta"])
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            rec["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            with self.lock:
                self.hints_sent += 1
                self.hints_ok += rec["ok"]
                if self.hint_log:
                    with open(self.hint_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
        threading.Thread(target=post, daemon=True).start()

    def episode(self, ep: dict, t0: float) -> None:
        time.sleep(max(0.0, t0 + ep["start_offset_s"] - time.time()))
        backend = MCPToolBackend()          # one MCP client per episode (thread-safe: separate event loops)
        tr = ep["trace"]
        with self.lock:
            self.active += 1
        ep["t_begin"] = time.time()
        trajectory: list[str] = []
        for i, (call, gap) in enumerate(zip(tr["calls"], ep["gaps"])):
            gap_start = time.time()
            for off in ep["hints"][i]:      # hints at the recorded LLM-step starts within this gap
                time.sleep(max(0.0, gap_start + off - time.time()))
                self.send_hint(ep["episode"], trajectory)
            time.sleep(max(0.0, gap_start + gap - time.time()))
            with self.lock:
                concurrent = self.active
            t_start = time.time()
            out, latency = backend.call(call["tool"], call["args"])
            with self.lock:
                self.rows.append({"episode": ep["episode"], "trace": f"{tr['source_model']}/task{tr['task_id']}/trial{tr['trial']}",
                                  "task_id": tr["task_id"], "task_set": tr["task_set"], "source_model": tr["source_model"],
                                  "call_idx": i, "tool": call["tool"], "t_start": t_start, "latency_s": latency,
                                  "planned_gap_s": gap, "user_turns_before": call["user_turns_before"],
                                  "concurrent_episodes": concurrent,
                                  "transport_ok": "MCP call" not in out, "tool_error": out.startswith("Error")})
            trajectory.append(call["tool"])
        ep["t_end"] = time.time()
        with self.lock:
            self.active -= 1
        n = len(tr["calls"])
        print(f"  {ep['episode']} done: task {tr['task_id']:>2} ({tr['source_model']}), {n} calls, "
              f"{ep['t_end'] - ep['t_begin']:.0f}s")


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))], 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes to replay (default: 46 = every W1 task twice; with --traces: every trace once)")
    ap.add_argument("--traces", nargs="+", help="w1_live result folders (our agent) or trace .jsonl files; "
                                                "default: the recorded tau-bench runs")
    ap.add_argument("--tasks", type=int, nargs="+", help="only these task ids (e.g. 4 9 10 23 27)")
    ap.add_argument("--gap-scale", type=float, default=1.0, help="multiply recorded gaps (our traces only)")
    ap.add_argument("--rate", type=float, default=2.0, help="average new episodes per minute (Poisson)")
    ap.add_argument("--think-median", type=float, default=1.5, help="median LLM think time before a call (s)")
    ap.add_argument("--think-sigma", type=float, default=0.5, help="spread of think time (lognormal sigma)")
    ap.add_argument("--user-turn-median", type=float, default=8.0,
                    help="median time for one customer turn (agent reply + customer answer) (s)")
    ap.add_argument("--user-turn-sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=252)
    ap.add_argument("--repeat", type=int, default=1, help="replay every selected trace this many times")
    ap.add_argument("--sequential", action="store_true",
                    help="one episode at a time; under C0/C3 wait until all tools are back at zero in between")
    ap.add_argument("--predictor-url", help="C3: send the agent's hints to this predictor (e.g. http://127.0.0.1:8000); "
                                            "the prewarm controller must be running")
    ap.add_argument("--workflow-id", default="W1", help="workflow id sent in hints")
    ap.add_argument("--label", default="", help="extra text at the end of the result folder name (e.g. eks)")
    a = ap.parse_args()

    episodes = plan(a)
    n_calls = sum(len(e["trace"]["calls"]) for e in episodes)
    span = episodes[-1]["start_offset_s"]
    print(f"W1 replay | {len(episodes)} episodes, {n_calls} tool calls, starts spread over {span / 60:.1f} min "
          f"(rate {a.rate}/min, seed {a.seed})")

    db = "redis (shared, reset once at start)" if shared_db() else "in-memory per pod (no shared DB!)"
    params = {k: v for k, v in vars(a).items()}
    cfg = run_config(sorted({e["trace"]["task_id"] for e in episodes}), db,
                     "W1 airline change/cancel (tau-bench), replay of recorded traces",
                     {"note": "replay: no LLM calls; " + (
                         "traces recorded by our ReAct agent (live runs), with their real gaps"
                         f" x{a.gap_scale}" if a.traces else
                         "traces recorded by gpt-4o and claude-3.5-sonnet (tau-bench), sampled gaps"),
                      "trace_models": sorted({e["trace"]["source_model"] for e in episodes})},
                     {"replay": {**params, "traces_source": a.traces or [str(TRACES.relative_to(HERE.parents[1]))],
                                 "planned_calls": n_calls, "start_span_s": span}})
    if a.predictor_url:   # scale-to-zero tools + predictor + prewarm controller = C3
        cfg["config"] = "C3" if cfg["config"] == "C0" else f"{cfg['config']}+prewarm"
        try:
            with urllib.request.urlopen(a.predictor_url + "/healthz", timeout=5) as r:
                health = json.loads(r.read())
        except Exception as e:
            raise SystemExit(f"Predictor not reachable at {a.predictor_url}: {e}")
        if health.get("crd", {}).get("mode") != "k8s":
            print("WARNING: predictor is not in CRD_MODE=k8s -> no PrewarmRequests, nothing gets prewarmed")
        cfg["prewarm"] = {"predictor_url": a.predictor_url, "workflow_id": a.workflow_id,
                          "predictor": {k: health.get(k) for k in ("model", "theta", "ttl_s")},
                          "crd_mode": health.get("crd", {}).get("mode"),
                          "mechanism": "Option A: controller sends an MCP ping per PrewarmRequest; "
                                       "TTL = Knative idle window",
                          "hints": "sent at the recorded start of every LLM step of the trace"}
    detail = (tasks_tag(cfg["task_ids"]) if a.traces or a.tasks else "taubench") + \
        (f"-x{a.repeat}" if a.repeat > 1 else "") + ("" if a.traces or a.tasks else f"-{len(episodes)}ep")
    out_dir = run_dir(cfg["config"], "replay", detail, a.label)
    print(f"-> {out_dir}")
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"config: {cfg['config']} | window {cfg['knative']['tool_window']} | images {cfg['tool_images']} | db: {db}")

    watcher = PodWatcher()
    watcher.start()
    scale_to_zero = not cfg["config"].startswith("C1")
    if scale_to_zero:
        wait_all_zero(watcher)          # C0 / C3: every run starts with all tools cold
        start_state = "all 14 tools at zero"
    else:
        wait_all_ready()                # C1 (min-scale 1): tools never go to zero; start once all are warm
        start_state = "all 14 tools warm"
    if db.startswith("redis"):
        reset_db()
    print(f"{start_state}, database reset; starting\n")

    rp = Replay(a.predictor_url, a.workflow_id, out_dir / "hints.jsonl" if a.predictor_url else None)
    if a.sequential:                    # one conversation at a time, each starting from the same state
        for k, e in enumerate(episodes):
            if k and scale_to_zero:
                wait_all_zero(watcher)
            if k and db.startswith("redis"):
                reset_db()
            e["start_offset_s"] = 0.0
            rp.episode(e, time.time())
    else:
        t0 = time.time() + 1.0
        threads = [threading.Thread(target=rp.episode, args=(e, t0), daemon=True) for e in episodes]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    time.sleep(3)              # let the watcher see the last pods
    watcher.stop_flag = True

    # ---- label calls: cold/warm and idle gap (needs the full pod history and all calls)
    rows = sorted(rp.rows, key=lambda r: r["t_start"])
    ends: dict[str, list[float]] = {}
    for r in rows:
        t_end = r["t_start"] + r["latency_s"]
        pod = watcher.new_pod_during(r["tool"], r["t_start"], t_end, slack=0.5)
        by_lat = r["latency_s"] > COLD_LATENCY_S
        prev = [e for e in ends.get(r["tool"], []) if e <= r["t_start"]]
        r.update({"latency_ms": round(r["latency_s"] * 1000, 1), "cold": bool(pod) or by_lat,
                  "cold_by_pod": bool(pod), "cold_by_latency": by_lat, "pod": pod or "",
                  "idle_gap_s": round(r["t_start"] - max(prev), 1) if prev else "",
                  "t_start": round(r["t_start"], 3), "planned_gap_s": round(r["planned_gap_s"], 2)})
        ends.setdefault(r["tool"], []).append(t_end)
    with open(out_dir / "calls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CALL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for e in episodes:
            tr = e["trace"]
            mine = [r for r in rows if r["episode"] == e["episode"]]
            f.write(json.dumps({"episode": e["episode"], "task_id": tr["task_id"], "task_set": tr["task_set"],
                                "source_model": tr["source_model"], "trial": tr["trial"], "recorded_reward": tr["reward"],
                                "start_offset_s": e["start_offset_s"], "duration_s": round(e["t_end"] - e["t_begin"], 2),
                                "tool_sequence": [c["tool"] for c in tr["calls"]], "planned_gaps_s": e["gaps"],
                                "calls": len(mine), "cold_calls": sum(r["cold"] for r in mine),
                                "tool_time_s": round(sum(r["latency_s"] for r in mine), 2)}) + "\n")
    pods = watcher.pod_rows()
    with open(out_dir / "pods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pod", "tool", "created", "first_seen", "last_seen"])
        w.writeheader()
        w.writerows(pods)

    # ---- summary
    ok = [r for r in rows if r["transport_ok"]]
    cold = [r["latency_ms"] for r in ok if r["cold"]]
    warm = [r["latency_ms"] for r in ok if not r["cold"]]
    allc = [r["latency_ms"] for r in ok]
    per_tool = {}
    for r in ok:
        d = per_tool.setdefault(r["tool"], {"calls": 0, "cold": 0})
        d["calls"] += 1
        d["cold"] += r["cold"]
    for d in per_tool.values():
        d["cold_share"] = round(d["cold"] / d["calls"], 2)
    warm_p50 = statistics.median(warm) if warm else 0
    summary = {
        "episodes": len(episodes), "tool_calls": len(rows), "transport_failures": len(rows) - len(ok),
        "cold_calls": len(cold), "cold_share": round(len(cold) / len(ok), 3) if ok else None,
        "latency_ms": {"all": {"p50": pct(allc, 50), "p95": pct(allc, 95), "p99": pct(allc, 99), "mean": round(statistics.mean(allc), 1) if allc else None},
                       "cold": {"p50": pct(cold, 50), "p95": pct(cold, 95), "max": max(cold) if cold else None},
                       "warm": {"p50": pct(warm, 50), "p95": pct(warm, 95), "max": max(warm) if warm else None}},
        "cold_start_time_added_s": round(sum(c - warm_p50 for c in cold) / 1000, 1),
        "signals_agree": sum(r["cold_by_pod"] == r["cold_by_latency"] for r in ok),
        "pods_started": len(pods),
        "pod_seconds_seen": round(sum(p["last_seen"] - p["first_seen"] for p in pods), 0),
        "per_tool": dict(sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])),
        "config": cfg["config"], "tool_window": cfg["knative"]["tool_window"], "database": db,
        "replay": params,
        "hints": {"sent": rp.hints_sent, "answered": rp.hints_ok} if a.predictor_url else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k not in ("per_tool", "replay")}, indent=2))
    print("\nper tool (calls / cold share):", ", ".join(f"{k} {v['calls']}/{v['cold_share']}" for k, v in summary["per_tool"].items()))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
