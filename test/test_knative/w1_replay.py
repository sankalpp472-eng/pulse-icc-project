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

--fanout: the recorded agents called tools one at a time, but many runs look up several items in a row
(get_reservation_details for every reservation, search_direct_flight for several dates). With --fanout
such a run of calls is fired at once (asyncio.gather) after one think time, as an agent that batches
independent lookups would. A group = consecutive calls of the same read-only tool, no customer turn in
between, and no call using a value that first appears in the output of an earlier call of the group
(outputs come from re-running the trace on a local copy of the database). The seed gives the same
episodes and start times with or without --fanout.

Every call is labelled cold/warm afterwards (latency > 1 s, or a new pod for that tool appeared
during the call) and gets its idle gap (time since that tool's previous call ended).

Run with the agent's Python, from the project root:
  agent/.venv/Scripts/python test/test_knative/w1_replay.py                       # 46 episodes, 2/min
  agent/.venv/Scripts/python test/test_knative/w1_replay.py --episodes 92 --rate 4
  agent/.venv/Scripts/python test/test_knative/w1_replay.py --fanout --tasks 2 3 28 30 33 34   # parallel lookups
  # replay OUR agent's live runs (w1_live result folders), with their real recorded gaps:
  agent/.venv/Scripts/python test/test_knative/w1_replay.py --traces test/test_knative/results/<date>_C0_W1-live_<...> ...

Output: results/local_runs/ (kind) or results/cloud_runs/ (EKS), <date>_<time>_<config>_W1-replay_<traces>/  (e.g. 2026-10-22_1336_C3_W1-replay_task9-x3)
  run_config.json  replay parameters + Knative/cluster settings (read from the cluster), code version
  calls.csv        one row per tool call
  episodes.jsonl   one row per episode (trace source, planned gaps, timings)
  pods.csv         every tool pod seen (tool, created, first/last seen)
  summary.json     cold share, latency percentiles, per-tool cold share, pod-seconds
"""

import argparse
import asyncio
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

from c0_common import (COLD_LATENCY_S, PodWatcher, peak_pods, reset_db, run_config, run_dir,  # noqa: E402
                       save_predictor_health,
                       shared_db, tasks_tag, wait_all_ready, wait_all_zero)
from tools import READ_ONLY_TOOLS as READ_TOOLS, LocalToolBackend, MCPToolBackend  # noqa: E402

TRACES = HERE / "data" / "w1_tau_traces.jsonl"
CALL_FIELDS = ["episode", "trace", "task_id", "task_set", "source_model", "call_idx", "group", "group_size", "tool", "t_start",
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
    # the signals each live hint carried (C3 live runs save hints.jsonl): replayed hints send the same ones,
    # so the next-action predictor (gbt) sees what it saw live. Per episode, in the order they were sent.
    recorded: dict[str, list] = {}
    if (run_dir / "hints.jsonl").exists():
        for line in open(run_dir / "hints.jsonl", encoding="utf-8"):
            h = json.loads(line)["hint"]
            recorded.setdefault(h.get("episode_id", ""), []).append(h)
        for hs in recorded.values():
            hs.sort(key=lambda h: h.get("ts", 0))
    out = []
    for n, line in enumerate(open(run_dir / "episodes.jsonl", encoding="utf-8")):
        ep = json.loads(line)
        rec_hints = iter(recorded.get(ep.get("episode_id", ""), []))
        # hints: the live agent sent a hint at the START of every LLM step (agent.emit_hint), including
        # steps where it only answered the customer -> keep their offsets within the gap before each call
        calls, known, unknown_turns, turns, hints, sigs = [], 0.0, 0, 0, [], []
        for st in ep.get("steps") or []:
            sigs.append((next(rec_hints, None) or {}).get("signals"))   # this LLM step's hint
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
                              # the agent ran this call at the same time as the previous one (same reply)
                              "with_previous": bool(k and st.get("parallel")),
                              "recorded_gap_s": round(known + (st["think_s"] if k == 0 else 0.05), 3),
                              "unknown_user_turns": unknown_turns if k == 0 else 0,
                              "hint_offsets_s": hints, "hint_signals": sigs if k == 0 else []})
                known, unknown_turns, turns, hints, sigs = 0.0, 0, 0, [], []
        if calls:
            out.append({"task_id": ep["task_id"], "task_set": ep.get("task_set", "?"),
                        "source_model": f"{model} (our agent, {run_dir.name})", "trial": n,
                        "reward": ep.get("reward"), "status": ep.get("status"), "calls": calls})
    return out


def _strings(x) -> set[str]:
    """Every string value (len >= 3) in a tool's arguments: ids, dates, airport codes."""
    if isinstance(x, dict):
        return set().union(*map(_strings, x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*map(_strings, x)) if x else set()
    return {x} if isinstance(x, str) and len(x) >= 3 else set()


def fanout_groups(trace: dict) -> list[list[int]]:
    """Split a trace's calls into groups that can run at once (lists of call indexes, in order).
    Call j joins the open group if it is the same read-only tool, has no customer turn before it,
    has new arguments, and uses no value that first appeared in the output of a call in the group."""
    calls = trace["calls"]
    if not any(a["tool"] == b["tool"] in READ_TOOLS for a, b in zip(calls, calls[1:])):
        return [[i] for i in range(len(calls))]
    backend = LocalToolBackend()        # re-run the trace on a fresh database to see what each call returned
    outputs = [backend.call(c["tool"], c["args"])[0] for c in calls]
    groups: list[list[int]] = []
    for j, c in enumerate(calls):
        g = groups[-1] if groups else None
        if g and c["tool"] == calls[g[0]]["tool"] in READ_TOOLS and not c["user_turns_before"] \
                and all(calls[i]["args"] != c["args"] for i in g):
            known = "\n".join(outputs[:g[0]])           # what the agent had seen before the group
            learned = [v for v in _strings(c["args"]) if v not in known]
            if not any(v in outputs[i] for v in learned for i in g):
                g.append(j)
                continue
        groups.append([j])
    return groups


TAU_DIR = HERE.parents[1] / "predictor" / "data" / "tau-bench" / "historical_trajectories"
HISTORY_N = 10     # conversation messages a hint carries for the LLM speculator (its prompt's LAST_N)
TAU_FILES = {"gpt-4o": "gpt-4o-airline.json", "claude-3.5-sonnet": "sonnet-35-new-airline.json"}


def attach_signals(traces: list[dict]) -> int:
    """Give every call of a tau-bench trace the predictor's signals (predictor/signals.py) for the LLM step
    that made it, computed from the original tau-bench conversation exactly as predictor/build_action_data.py
    does for training. The next-action models (PREDICTOR_MODEL=gbt ...) need them; without them the
    predictor answers with its trajectory-only ensemble. Returns the number of traces matched.
    Also keeps, per call, the state before every LLM step since the previous call (the agent's replies to
    the customer, then the call itself): the fair hint timing sends one hint per LLM step with only what
    was known then (c["hint_states"] = [(signals, history), ...], last = the call's own step)."""
    sys.path.insert(0, str(HERE.parents[1] / "predictor"))
    from build_action_data import tool_calls
    from signals import compute_signals
    convs: dict[tuple, list] = {}
    for model, fname in TAU_FILES.items():
        if not (TAU_DIR / fname).exists():
            raise SystemExit(f"{TAU_DIR / fname} missing: download tau-bench's historical_trajectories "
                             f"(see predictor/build_action_data.py), or pass --no-signals")
        for ep in json.loads((TAU_DIR / fname).read_text(encoding="utf-8")):
            events, per_call, history, pre = [], [], [], []
            for m in ep["traj"]:
                role = m.get("role")
                if role == "user":
                    events.append(("user", m.get("content") or ""))
                elif role == "tool":
                    events.append(("tool", m.get("name") or "", str(m.get("content") or "")))
                elif role == "assistant":
                    calls = tool_calls(m)
                    sig = compute_signals(events)
                    hist = history[-HISTORY_N:]        # what the LLM speculator reads (--hint-context)
                    for n, args in calls:
                        per_call.append(((n, args), (sig, hist, pre)))
                        events.append(("call", n, args))
                    if calls:
                        pre = []
                    else:                              # the agent answers the customer: an LLM step too
                        pre = pre + [(sig, hist)]
                        events.append(("respond",))
                if role in ("user", "tool", "assistant"):   # compact message, as baselines/llm_speculator/steps.py
                    cm = {"role": role, "content": str(m.get("content") or "")[:600]}
                    if role == "tool":
                        cm["name"] = m.get("name") or ""
                    if role == "assistant" and tool_calls(m):
                        cm["calls"] = [[n, a] for n, a in tool_calls(m)]
                    history.append(cm)
            convs.setdefault((model, ep["task_id"], ep.get("trial", 0)), []).append(per_call)
    matched = 0
    for tr in traces:
        want = [(c["tool"], c["args"]) for c in tr["calls"]]
        cands = convs.get((tr["source_model"], tr["task_id"], tr.get("trial", 0)), [])
        # same tools and arguments first; the tools alone if args were edited (fill_trace_args.py)
        hit = next((pc for pc in cands if [k for k, _ in pc] == want), None) or \
            next((pc for pc in cands if [k[0] for k, _ in pc] == [w[0] for w in want]), None)
        if hit:
            for c, (_, (sig, hist, before)) in zip(tr["calls"], hit):
                c["signals"], c["history"] = sig, hist
                n = c["user_turns_before"]             # one replayed customer turn per agent reply
                before = (before[:1] * (n - len(before)) + before)[-n:] if n else []
                c["hint_states"] = before + [(sig, hist)]
            matched += 1
    return matched


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
    a.signals_matched = None
    if a.predictor_url and not a.no_signals and not a.traces:     # tau-bench traces: signals for the hints
        a.signals_matched = f"{attach_signals(traces)} of {len(traces)} traces"
        print(f"hint signals: {a.signals_matched} matched to their tau-bench conversation")
    traces = traces * a.repeat

    def gaps_for(tr):
        """Gap before every call, and its parts: (agent think time, [customer turns]). The draw order is
        fixed, so a seed gives the same gaps whatever the hint timing."""
        gaps, parts = [], []
        for c in tr["calls"]:
            if "recorded_gap_s" in c:      # our agent: real timing, scaled; sample only missing customer turns
                g = c["recorded_gap_s"] * a.gap_scale
                turns = c.get("unknown_user_turns", 0)
            else:                          # tau-bench recordings carry no timing: sample think time
                g = lognormal(rng, a.think_median, a.think_sigma, 0.2, 15.0)
                turns = c["user_turns_before"]
            us = [lognormal(rng, a.user_turn_median, a.user_turn_sigma, 1.0, 60.0) for _ in range(turns)]
            gaps.append(g + sum(us))
            parts.append((g, us))
        return gaps, parts

    def hints_for(tr, parts):
        """When to send hints within the gap before each call (seconds from the gap's start) and what each
        carries: [[(offset, signals, history), ...] per call].
          our agent's runs: at the recorded start of every LLM step, with the signals it sent live
          tau-bench, fair:  before every LLM step -- each agent reply to the customer (at the start of its
                            customer turn) and the call itself (after the last turn) -- with only what was
                            known at that step, so the customer's reply time is lead time the predictor
                            earns by predicting before the reply
          tau-bench, legacy (runs before 28 Oct 2026): one hint at the gap's start, carrying the call
                            step's signals, which already include the customer's reply"""
        out = []
        for c, (_, us) in zip(tr["calls"], parts):
            if "hint_offsets_s" in c:
                rec = c.get("hint_signals") or []
                out.append([(o * a.gap_scale, rec[h] if h < len(rec) and rec[h] is not None else c.get("signals"),
                             None) for h, o in enumerate(c["hint_offsets_s"])])
            elif getattr(a, "hint_timing", "fair") == "fair" and "hint_states" in c:
                offs = [sum(us[:k]) for k in range(len(us) + 1)]
                out.append([(o, sg, hs) for o, (sg, hs) in zip(offs, c["hint_states"])])
            else:
                out.append([(0.0, c.get("signals"), c.get("history"))])
        return out

    cache: dict[int, list[list[int]]] = {}

    def groups_for(tr):
        """Calls fired together. Uses no randomness, so --fanout leaves the seeded plan unchanged.
        Our agent's runs: the calls it really ran at once (one reply) always stay together."""
        if any(c.get("with_previous") for c in tr["calls"]):
            out = []
            for i, c in enumerate(tr["calls"]):
                if c.get("with_previous"):
                    out[-1].append(i)
                else:
                    out.append([i])
            return out
        if not a.fanout:
            return [[i] for i in range(len(tr["calls"]))]
        if id(tr) not in cache:
            cache[id(tr)] = fanout_groups(tr)
        return cache[id(tr)]

    if a.episodes is None and a.traces:    # our own traces: replay every one once, in a seeded order
        order = traces[:]
        rng.shuffle(order)
        episodes, t = [], 0.0
        for tr in order:
            t += rng.expovariate(a.rate / 60.0)
            gaps, parts = gaps_for(tr)
            episodes.append({"episode": f"ep{len(episodes) + 1:03d}", "start_offset_s": round(t, 2), "trace": tr,
                             "gaps": [round(g, 2) for g in gaps], "hints": hints_for(tr, parts),
                             "groups": groups_for(tr)})
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
            gaps, parts = gaps_for(tr)
            t += rng.expovariate(a.rate / 60.0)
            episodes.append({"episode": f"ep{len(episodes) + 1:03d}", "start_offset_s": round(t, 2), "trace": tr,
                             "gaps": [round(g, 2) for g in gaps], "hints": hints_for(tr, parts),
                             "groups": groups_for(tr)})
    return episodes


class Replay:
    def __init__(self, predictor_url: str | None = None, workflow_id: str = "W1", hint_log: Path | None = None,
                 hint_context: bool = False, hint_timeout: float = 1.0):
        self.rows, self.lock, self.active = [], threading.Lock(), 0
        self.predictor_url, self.workflow_id, self.hint_log = predictor_url, workflow_id, hint_log
        self.hint_context, self.hint_timeout = hint_context, hint_timeout
        self.hints_sent = self.hints_ok = 0

    def send_hint(self, episode: str, trajectory: list[str], signals: dict | None = None,
                  history: list | None = None) -> None:
        """What agent.emit_hint does: POST the trajectory (+ signals, if known) to the predictor in the
        background (never waits). With --hint-context also the conversation so far (LLM speculator)."""
        if not self.predictor_url:
            return
        hint = {"workflow_id": self.workflow_id, "episode_id": episode, "trajectory": trajectory[-8:],
                "current_tool": trajectory[-1] if trajectory else "START", "ts": time.time()}
        if signals is not None:
            hint["signals"] = signals
        if self.hint_context and history is not None:
            hint["history"] = history

        def post():
            rec, t0 = {"hint": hint}, time.perf_counter()
            try:
                req = urllib.request.Request(self.predictor_url + "/predict", data=json.dumps(hint).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=self.hint_timeout) as r:
                    ans = json.loads(r.read())
                rec.update(ok=True, prewarm=ans["prewarm"], top3=ans["ranked"][:3], theta=ans["theta"],
                           model=ans.get("model"))
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

        async def fire(idx):               # one group: all its calls at once
            return await asyncio.gather(*(backend.acall(tr["calls"][i]["tool"], tr["calls"][i]["args"]) for i in idx))

        for k, idx in enumerate(ep["groups"]):
            i = idx[0]                      # the group waits for its first call's gap; the others' gaps are skipped
            gap = ep["gaps"][i]
            gap_start = time.time()
            for off, sig, hist in ep["hints"][i]:   # one hint per LLM step in this gap (see plan.hints_for)
                time.sleep(max(0.0, gap_start + off - time.time()))
                self.send_hint(ep["episode"], trajectory, sig, hist)
            time.sleep(max(0.0, gap_start + gap - time.time()))
            with self.lock:
                concurrent = self.active
            results = asyncio.run(fire(idx))
            with self.lock:
                for j, (out, latency, t_start) in zip(idx, results):
                    call = tr["calls"][j]
                    self.rows.append({"episode": ep["episode"], "trace": f"{tr['source_model']}/task{tr['task_id']}/trial{tr['trial']}",
                                      "task_id": tr["task_id"], "task_set": tr["task_set"], "source_model": tr["source_model"],
                                      "call_idx": j, "group": k, "group_size": len(idx), "tool": call["tool"],
                                      "t_start": t_start, "latency_s": latency,
                                      "planned_gap_s": gap if j == i else 0.0, "user_turns_before": call["user_turns_before"],
                                      "concurrent_episodes": concurrent,
                                      "transport_ok": "MCP call" not in out, "tool_error": out.startswith("Error")})
            trajectory += [tr["calls"][j]["tool"] for j in idx]
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
    ap.add_argument("--fanout", action="store_true",
                    help="fire independent runs of the same read-only tool at once (asyncio.gather), see top of file")
    ap.add_argument("--workflow-id", default="W1", help="workflow id sent in hints")
    ap.add_argument("--hint-timing", choices=["fair", "legacy"], default="fair",
                    help="tau-bench traces: fair = a hint before every LLM step with only what was known then "
                         "(default); legacy = one hint at the gap start with the call step's signals (runs "
                         "before 28 Oct 2026: it knew the customer's reply before the reply time had passed)")
    ap.add_argument("--hint-context", action="store_true",
                    help="C3 with the LLM speculator (baselines/llm_speculator/service.py): hints also carry the "
                         "last 10 messages of the recorded conversation")
    ap.add_argument("--hint-timeout", type=float, default=1.0,
                    help="seconds to wait for the predictor's answer (logging only; prewarming happens anyway)")
    ap.add_argument("--no-signals", action="store_true",
                    help="C3: send trajectory-only hints (the predictor then uses its ensemble, not gbt)")
    ap.add_argument("--label", default="", help="extra text at the end of the result folder name (e.g. eks)")
    a = ap.parse_args()

    episodes = plan(a)
    n_calls = sum(len(e["trace"]["calls"]) for e in episodes)
    span = episodes[-1]["start_offset_s"]
    print(f"W1 replay | {len(episodes)} episodes, {n_calls} tool calls, starts spread over {span / 60:.1f} min "
          f"(rate {a.rate}/min, seed {a.seed})")
    multi = [g for e in episodes for g in e["groups"] if len(g) > 1]
    fanout = {"enabled": a.fanout, "groups": len(multi), "calls_in_groups": sum(map(len, multi)),
              "max_width": max(map(len, multi), default=1)}
    if multi:                           # --fanout, or our agent's recorded parallel calls
        print(f"fan-out | {fanout['groups']} concurrent groups, {fanout['calls_in_groups']} of {n_calls} calls "
              f"in them, widest {fanout['max_width']}")

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
                                 "planned_calls": n_calls, "start_span_s": span, "fanout": fanout}})
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
                          "mechanism": "Option A: controller sends k = desiredReplicas MCP pings per PrewarmRequest "
                                       "(one if the tool already has k pods); TTL = Knative idle window",
                          "hints": "sent at the recorded start of every LLM step of the trace",
                          "hint_signals": a.signals_matched, "hint_context": a.hint_context,
                          "hint_timing": a.hint_timing,
                          "hint_timeout_s": a.hint_timeout,
                          "health": {k: health.get(k) for k in ("stats", "keys", "action_models") if k in health}}
    detail = (tasks_tag(cfg["task_ids"]) if a.traces or a.tasks else "taubench") + \
        (f"-x{a.repeat}" if a.repeat > 1 else "") + ("" if a.traces else f"-{len(episodes)}ep") + \
        ("-fanout" if a.fanout else "")
    out_dir = run_dir(cfg["config"], "replay", detail, a.label)
    print(f"-> {out_dir}")
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"config: {cfg['config']} | window {cfg['knative']['tool_window']} | images {cfg['tool_images']} | db: {db}")

    watcher = PodWatcher()
    watcher.start()
    scale_to_zero = not cfg["config"].startswith("C1")
    if scale_to_zero:
        wait_all_zero(watcher)          # C0 / C3: every run starts with all tools cold
        watcher.forget()                # pods from before the run (e.g. an old revision) are not part of it
        start_state = "all 14 tools at zero"
    else:
        wait_all_ready()                # C1 (min-scale 1): tools never go to zero; start once all are warm
        start_state = "all 14 tools warm"
    if db.startswith("redis"):
        reset_db()
    print(f"{start_state}, database reset; starting\n")

    rp = Replay(a.predictor_url, a.workflow_id, out_dir / "hints.jsonl" if a.predictor_url else None,
                a.hint_context, a.hint_timeout)
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
        # a new pod counts only if the tool had none before; else it may be extra capacity for parallel calls
        pod = not watcher.had_pod_before(r["tool"], r["t_start"]) and \
            watcher.new_pod_during(r["tool"], r["t_start"], t_end, slack=0.5)
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
                                "groups": e["groups"],
                                "calls": len(mine), "cold_calls": sum(r["cold"] for r in mine),
                                "tool_time_s": round(sum(r["latency_s"] for r in mine), 2)}) + "\n")
    pods = watcher.pod_rows()
    with open(out_dir / "pods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pod", "tool", "created", "first_seen", "last_seen", "deleting_seen"])
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
        "peak_pods_per_tool": peak_pods(pods),
        "per_tool": dict(sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])),
        "config": cfg["config"], "tool_window": cfg["knative"]["tool_window"], "database": db,
        "replay": params,
        "fanout": {**fanout, "cold_calls_in_groups": sum(r["cold"] for r in ok if r["group_size"] > 1)},
        "hints": {"sent": rp.hints_sent, "answered": rp.hints_ok} if a.predictor_url else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_predictor_health(a.predictor_url, out_dir)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k not in ("per_tool", "replay")}, indent=2))
    print("\nper tool (calls / cold share):", ", ".join(f"{k} {v['calls']}/{v['cold_share']}" for k, v in summary["per_tool"].items()))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
