"""
Capacity simulation: how prewarming behaves when the tool nodes fill up. No cluster, Docker or LLM: recorded
tau-bench episodes (the same plan as pulse/experiments/replay.py: traces, think times, customer turns, fan-out
groups, hint timing) run on the simulated cluster (pulse/ui/sim_cluster.py) with a pod capacity, on a virtual
clock, against the real predictor called in-process (pulse/predictor/service.py, CRD writes off).

Configurations (every one sees the same episodes, hints and predictions):
  C0        scale to zero, Knative window 60 s, no prewarming
  C1        one pod per tool always warm (min-scale 1), window 60 s
  C3-ping   the ping controller (Option A): each predicted tool gets its k pods at once; window 60 s
  C3-lease  the lease controller (pulse/controller/leases.py): admission control -- a prediction starts only the new
            pods that fit next to a reserve of free pods (fewer, or none for a cold tool); window 60 s

Per configuration and capacity: tool-call latency p50 / p95 / p99 / max, cold share, pods that had to wait for a
slot, pod-seconds (cost), and the lease decisions.

Capacity = pod slots for TOOL pods (the Knative and system pods on the tool nodes already taken out). On EKS one
m7i-flex.large worker held about 15: its CPU left after the system pods, at 75m per tool pod (50m tool + 25m
queue-proxy). A scaled-down pod keeps its slot for --grace seconds (Knative's scale-to-zero grace period, 30 s).

  python -m pulse.experiments.sim_capacity                          # capacities 0 (no limit), 24, 15 pods
  python -m pulse.experiments.sim_capacity --capacity 15 --rate 16 --episodes 60   # the EKS pressure run
  ... --configs C0 C3-ping --refresh always never value --shed-p 0 0.6 --windows 60 per-tool   # pod lifetime levers

Output: results/sim_runs/<date>_<time>_capacity[_label]/  summary.csv, summary.json, calls.csv
"""

import argparse
import csv
import heapq
import itertools
import json
import os
import statistics
import sys
import time as _time
import types

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capacity", type=int, nargs="+", default=[0, 24, 15],
                    help="pod slots for tool pods (0 = no limit; one EKS worker ~ 15)")
    ap.add_argument("--configs", nargs="+", default=["C0", "C1", "C3-ping", "C3-lease"])
    ap.add_argument("--predictor", default="lightgbm", help="lightgbm | markov | pulse_v3 (speculator: needs an LLM)")
    ap.add_argument("--prewarm-policy", choices=["adaptive", "topk"], default="adaptive",
                    help="adaptive: top 3 candidates, each warmed only if its learned hit rate x cold start is worth "
                         "a slot now (pulse/controller/adaptive.py); topk: exactly --top-k tools")
    ap.add_argument("--top-k", type=int, default=2, help="topk: tools per step")
    ap.add_argument("--p-min", type=float, default=0.05, help="every candidate needs p >= this")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--rate", type=float, default=4.0, help="new episodes per minute (Poisson)")
    ap.add_argument("--seed", type=int, default=252)
    ap.add_argument("--think-median", type=float, default=1.5)
    ap.add_argument("--think-sigma", type=float, default=0.5)
    ap.add_argument("--user-turn-median", type=float, default=8.0)
    ap.add_argument("--user-turn-sigma", type=float, default=0.5)
    ap.add_argument("--no-fanout", action="store_true", help="run every call alone (no parallel groups)")
    ap.add_argument("--window", type=float, default=60.0, help="the tools' Knative idle window (s)")
    ap.add_argument("--grace", type=float, default=30.0, help="seconds a scaled-down pod still holds its slot")
    ap.add_argument("--cold-median", type=float, default=1.56,
                    help="scale the measured per-tool cold starts to this median (s; 0 = as measured). 1.56 = cold "
                         "calls inside the EKS cluster (18 Nov 2026); data/coldstart/ measures from the laptop, so "
                         "its times include the internet round trip")
    ap.add_argument("--autoscaler", choices=["kpa", "window"], default="kpa",
                    help="kpa: Knative's KPA (concurrency, panic mode); window: each pod lives window s after its "
                         "last call (the UI demo's simpler model)")
    ap.add_argument("--reserve", type=int, default=4, help="C3-lease: free pod slots prewarms never take")
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--no-adapt-k", action="store_true", help="C3-lease: k as the predictor asks (no fan-out cap)")
    ap.add_argument("--k-policy", nargs="+", default=["rule"], choices=["pred", "one", "rule", "adaptive", "oracle"],
                    help="how many pods a prediction asks for (C3): rule (default, what the controllers do) = "
                         "min(k, ceil(k * call time / --k-budget)); adaptive = rule with the budget scaled by the "
                         "tool's cold start and how full the nodes are (controller K_ADAPT=1, pulse/controller/"
                         "adaptive.py); pred = the predictor's pending-ID k; one = 1; oracle = the calls the next step "
                         "really makes to that tool. Several = a sweep")
    ap.add_argument("--k-budget", type=float, default=0.3,
                    help="rule: queueing (s) one pod may add to a fan-out before another pod is worth it")
    ap.add_argument("--warm-s", type=float, default=0.04, help="warm call time (s); 0.04 = the tau-bench tools")
    ap.add_argument("--ttl-step", type=int, default=20, help="predictor's TTL for a next-step prediction (s)")
    ap.add_argument("--ttl-ahead", type=int, default=75, help="predictor's TTL for a lookahead prediction (s)")
    ap.add_argument("--refresh", nargs="+", default=["always"], choices=["always", "never", "value"],
                    help="C3-ping: does a prediction ping a tool that already has its pods (pulse/controller/"
                         "refresh.py)? always (as before) | never | value. Several = a sweep")
    ap.add_argument("--refresh-free-high", type=int, default=6, help="value: free slots at which a refresh is free")
    ap.add_argument("--refresh-p-max", type=float, default=0.8, help="value: the threshold with no free slot")
    ap.add_argument("--shed-p", nargs="+", type=float, default=[0.0],
                    help="C3-ping: while pods wait for a slot, drop predictions below this (0 = off). Several = a sweep")
    ap.add_argument("--windows", nargs="+", default=["60"], choices=["60", "per-tool"],
                    help="the tools' idle windows: 60 (--window for all) or per-tool (data/tool_windows.json, "
                         "pulse.experiments.tool_windows). Several = a sweep (applies to every configuration)")
    ap.add_argument("--tick", type=float, default=0.25, help="simulation step for the autoscaler (s)")
    ap.add_argument("--label", default="")
    return ap.parse_args()


A = parse_args()
os.environ["PREDICTOR"], os.environ["CRD_MODE"] = A.predictor, "off"
os.environ["PREWARM_POLICY"], os.environ["TOP_K"], os.environ["P_MIN"] = A.prewarm_policy, str(A.top_k), str(A.p_min)
os.environ.setdefault("LOOKAHEAD", "0")
os.environ["TTL_STEP_S"], os.environ["TTL_AHEAD_S"] = str(A.ttl_step), str(A.ttl_ahead)

from pulse import paths  # noqa: E402
import pulse.ui.sim_cluster as sim_cluster  # noqa: E402
from pulse.controller.leases import LeaseBook, LeaseConfig, rule_k  # noqa: E402
from pulse.controller.refresh import RefreshConfig, decide  # noqa: E402
from pulse.controller.adaptive import REF_COLD_S, budget, worth_warming  # noqa: E402
from pulse.experiments import tool_windows  # noqa: E402
from pulse.experiments import replay  # noqa: E402
from pulse.predictor import client  # noqa: E402

sim_cluster.WARM_S = A.warm_s       # the simulator's warm call time (its busy-pod queueing uses it too)
WARM_S = A.warm_s


class Clock:
    t = 0.0


CLOCK = Clock()
sim_cluster.time = types.SimpleNamespace(time=lambda: CLOCK.t)     # the simulator runs on virtual time


# ---------------------------------------------------------------------- predictions (computed once, shared)
class Predictions:
    def __init__(self):
        import pulse.predictor.service as svc
        from pulse.predictor.contract import PredictRequest
        self.svc, self.Req = svc, PredictRequest
        self.wanted = set(svc.PRED.inputs)
        self.cache: dict[tuple, list] = {}

    def get(self, key: tuple, episode: str, trajectory: list[str], signals, history) -> list:
        """[(tool, p, k, ttl_s, lookahead, hit rate or None), ...] the predictor's prewarm list for this hint.
        Computed once and shared by every configuration: the adaptive policy's hit rates are what the predictor's
        calibrator had learned by then, in the first configuration's order of hints."""
        if key not in self.cache:
            hint = client.make_hint(self.wanted, episode, trajectory, None, domain=replay.DOMAIN,
                                    signals=signals, history=history)
            out = self.svc.predict(self.Req(**hint))
            ahead = {a.tool for a in out.lookahead}
            self.cache[key] = [(a.tool, a.p, a.k or 1, a.ttl_s or out.ttl_s, a.tool in ahead, a.hit)
                               for a in out.prewarm]
        return self.cache[key]


# ---------------------------------------------------------------------- one configuration
class Run:
    def __init__(self, config: str, capacity: int, plan: list[dict], preds: Predictions, tools: list[str],
                 k_policy: str = "pred", refresh: RefreshConfig | None = None, windows: str = "60"):
        self.config, self.capacity, self.plan, self.preds = config, capacity, plan, preds
        self.k_policy = k_policy
        self.refresh, self.windows = refresh or RefreshConfig(), windows
        self.skips = {"refresh": 0, "pressure": 0, "topk": 0}
        lease = config == "C3-lease"
        CLOCK.t = 0.0                    # before the simulator exists: C1's always-warm pods are ready at t = 0
        per_tool = tool_windows.load(replay.DOMAIN) if windows == "per-tool" else {}
        if windows == "per-tool" and not per_tool:
            raise SystemExit(f"no per-tool windows for {replay.DOMAIN} in {tool_windows.OUT}: "
                             "python -m pulse.experiments.tool_windows <C0 runs> --domain " + replay.DOMAIN)
        self.sim = sim_cluster.SimCluster(tools, window_s=A.window, min_scale=1 if config == "C1" else 0,
                                          capacity=capacity or None, grace_s=A.grace, seed=A.seed,
                                          autoscaler=A.autoscaler, windows=per_tool)
        if A.cold_median:
            scale = A.cold_median / statistics.median(self.sim.cold.values())
            self.sim.cold = {t: v * scale for t, v in self.sim.cold.items()}
        self.book = LeaseBook(LeaseConfig(reserve_pods=A.reserve, k_max=A.k_max, adapt_k=not A.no_adapt_k,
                                          window_s=A.window)) if lease else None
        self.q: list = []
        self.seq = itertools.count()
        self.rows: list[dict] = []
        self.waiting: list[dict] = []               # calls on a Pending pod
        self.open_eps = 0

    def at(self, t: float, fn, *args):
        heapq.heappush(self.q, (t, next(self.seq), fn, args))

    # ---- the controller
    def hint(self, ep: dict, i: int, j: int, signals, history, trajectory: list[str]):
        if self.config not in ("C3-ping", "C3-lease"):
            return
        items = self.preds.get((ep["episode"], i, j), ep["episode"], trajectory, signals, history)
        group = next((g for g in ep["groups"] if g[0] == i), [i])     # the calls the next step really makes
        for tool, p, k, ttl, ahead, hit in items:
            if tool not in self.sim.pods:
                continue
            if hit is not None:                          # adaptive top-k: worth a slot now? (adaptive.py)
                free = self.sim.free_pods() if self.capacity else 10 ** 6
                if not worth_warming(hit, self.sim.cold.get(tool, REF_COLD_S), free, self.sim.pending_pods())[0]:
                    self.skips["topk"] += 1
                    continue
            k = self.choose_k(tool, k, ep, group)
            if self.config == "C3-ping":                 # refresh / skip / shed: pulse/controller/refresh.py
                cap = (self.sim.free_pods(), self.sim.pending_pods()) if self.refresh.needs_capacity() else (0, 0)
                action, why = decide(p, len(self.sim.pods[tool]), k, self.refresh, *cap)
                if action == "skip":
                    self.skips["pressure" if why.startswith("pressure") else "refresh"] += 1
                    continue
                self.sim.prewarm(tool, k, p)
                continue
            have = len(self.sim.pods[tool])          # ready, starting or waiting for a slot
            d = self.book.request(tool, ep["episode"], k, ttl, CLOCK.t, self.sim.free_pods(), have)
            if d.action != "reject":                 # the controller's pings: start the new pods, refresh the window
                self.sim.prewarm(tool, max(1, d.k), p)

    def choose_k(self, tool: str, k_pred: int, ep: dict, group: list[int]) -> int:
        k_pred = max(1, int(k_pred or 1))
        if self.k_policy == "one":
            return 1
        if self.k_policy == "rule":       # the smallest k whose remaining queueing fits the budget (as the controllers)
            return rule_k(k_pred, A.warm_s, A.k_budget)
        if self.k_policy == "adaptive":   # K_ADAPT=1: the budget follows the tool's cold start and the free slots
            free = self.sim.free_pods() if self.capacity else 10 ** 6
            return rule_k(k_pred, A.warm_s, budget(A.k_budget, self.sim.cold.get(tool, REF_COLD_S), free))
        if self.k_policy == "oracle":
            return max(1, sum(ep["trace"]["calls"][j]["tool"] == tool for j in group))
        return k_pred

    def tick(self):
        if self.book:
            self.book.expire(CLOCK.t)
            for t in self.book.due_pings(CLOCK.t):  # a lease that outlives the window: one extending ping
                self.sim.keepalive(t)
                self.book.pinged(t, CLOCK.t)
        self.sim.tick(CLOCK.t)
        for c in list(self.waiting):                # a Pending pod got a slot and started
            if c["pod"]["ready_at"] != sim_cluster.INF:
                self.waiting.remove(c)
                self.at(max(CLOCK.t, c["pod"]["ready_at"]) + WARM_S, self.done, c)
        if self.open_eps or self.waiting or self.q:      # episodes running, calls waiting, or more to come
            self.at(CLOCK.t + A.tick, self.tick)

    # ---- an episode
    def start(self, ep: dict):
        self.open_eps += 1
        ep["_k"], ep["_traj"] = 0, []
        self.gap(ep)

    def gap(self, ep: dict):
        k = ep["_k"]
        if k >= len(ep["groups"]):
            self.open_eps -= 1
            return
        i = ep["groups"][k][0]
        t0 = CLOCK.t
        for j, (off, sig, hist) in enumerate(ep["hints"][i]):
            self.at(t0 + off, self.hint, ep, i, j, sig, hist, list(ep["_traj"]))
        self.at(t0 + ep["gaps"][i], self.fire, ep)

    def fire(self, ep: dict):
        idx = ep["groups"][ep["_k"]]
        tr = ep["trace"]
        ep["_left"] = len(idx)
        if self.book:
            widths: dict[str, int] = {}
            for j in idx:
                widths[tr["calls"][j]["tool"]] = widths.get(tr["calls"][j]["tool"], 0) + 1
            for t, w in widths.items():
                self.book.observe_fanout(t, w)
        for j in idx:
            tool = tr["calls"][j]["tool"]
            wait, cold, pod = self.sim.acquire(tool)
            c = {"ep": ep, "j": j, "tool": tool, "t": CLOCK.t, "cold": cold, "pod": pod, "group": len(idx),
                 "pending": wait == sim_cluster.INF}
            if c["pending"]:
                self.waiting.append(c)
            else:
                self.at(CLOCK.t + wait + WARM_S, self.done, c)

    def done(self, c: dict):
        self.sim.release(c["pod"])
        lat = CLOCK.t - c["t"]
        self.rows.append({"config": self.config, "capacity": self.capacity or "none", "episode": c["ep"]["episode"],
                          "call_idx": c["j"], "tool": c["tool"], "group_size": c["group"], "t_start": round(c["t"], 3),
                          "latency_s": round(lat, 4), "cold": c["cold"], "waited_for_slot": c["pending"]})
        ep = c["ep"]
        ep["_left"] -= 1
        if ep["_left"] == 0:
            ep["_traj"] += [ep["trace"]["calls"][j]["tool"] for j in ep["groups"][ep["_k"]]]
            ep["_k"] += 1
            self.gap(ep)

    def run(self) -> dict:
        eps = [dict(e) for e in self.plan]
        for e in eps:
            self.at(e["start_offset_s"], self.start, e)
        self.at(0.0, self.tick)
        while self.q:
            t, _, fn, args = heapq.heappop(self.q)
            CLOCK.t = max(CLOCK.t, t)
            fn(*args)
        return self.summary()

    def summary(self) -> dict:
        lat = sorted(r["latency_s"] for r in self.rows)
        n = len(lat)

        def q(p):
            return round(lat[min(n - 1, int(round(p / 100 * (n - 1))))], 3) if n else None
        pod_s = self.sim.live_pod_seconds() + self.sim.pending_pod_seconds()
        ping = self.config == "C3-ping"
        out = {"config": self.config, "k_policy": self.k_policy if self.config.startswith("C3") else "-",
               "windows": self.windows, "refresh": self.refresh.policy if ping else "-",
               "shed_p": self.refresh.shed_p if ping else "-",
               "skipped_refresh": self.skips["refresh"] if ping else "-",
               "skipped_pressure": self.skips["pressure"] if ping else "-",
               "skipped_topk": self.skips["topk"] if self.config.startswith("C3") else "-",
               "capacity": self.capacity or "none", "calls": n,
               # the runners' rule (experiments/common.py COLD_LATENCY_S): a call slower than 1 s was cold
               "cold_share": round(sum(r["latency_s"] > 1.0 for r in self.rows) / n, 3) if n else None,
               "met_starting_pod": round(sum(r["cold"] for r in self.rows) / n, 3) if n else None,
               "waited_for_slot": sum(r["waited_for_slot"] for r in self.rows),
               "p50_s": q(50), "p95_s": q(95), "p99_s": q(99), "max_s": round(lat[-1], 3) if n else None,
               "mean_s": round(statistics.mean(lat), 3) if n else None,
               "pod_seconds": round(pod_s, 1), "pods_pending_starts": self.sim.pending_starts,
               "makespan_s": round(CLOCK.t, 1)}
        if self.book:
            out.update({f"lease_{k}": v for k, v in self.book.stats.items()})
        return out


# ---------------------------------------------------------------------- main
def main():
    tau_ok = all((replay.TAU_DIR / f).exists() for f in replay.TAU_FILES.values())
    plan_args = types.SimpleNamespace(
        seed=A.seed, traces=None, tasks=None, repeat=1, predictor_url="in-process", no_signals=not tau_ok,
        think_median=A.think_median, think_sigma=A.think_sigma, user_turn_median=A.user_turn_median,
        user_turn_sigma=A.user_turn_sigma, gap_scale=1.0, hint_timing="fair", fanout=not A.no_fanout,
        episodes=A.episodes, rate=A.rate)
    plan = replay.plan(plan_args)
    if not tau_ok:
        print("tau-bench conversations not downloaded: hints carry the trajectory only "
              "(python -m pulse.setup.download_data for signals)")
    tools = sorted({c["tool"] for e in plan for c in e["trace"]["calls"]} |
                   set(sim_cluster.SimCluster([]).cold))
    preds = Predictions()
    stamp = _time.strftime("%Y-%m-%d_%H%M")
    out = paths.RESULTS / "sim_runs" / (f"{stamp}_capacity" + (f"_{A.label}" if A.label else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(plan)} episodes ({sum(len(e['trace']['calls']) for e in plan)} calls), rate {A.rate}/min, "
          f"predictor {A.predictor} {A.prewarm_policy}"
          f"{f' top-{A.top_k}' if A.prewarm_policy == 'topk' else ''} | {len(tools)} tools | -> {out}")
    summaries, rows = [], []
    for cap, win, cfg in itertools.product(A.capacity, A.windows, A.configs):
        ping = cfg == "C3-ping"
        for kp, pol, shed in itertools.product(A.k_policy if cfg.startswith("C3") else ["-"],
                                               A.refresh if ping else ["always"], A.shed_p if ping else [0.0]):
            rc = RefreshConfig(pol, A.refresh_free_high, A.refresh_p_max, shed)
            r = Run(cfg, cap, plan, preds, tools, kp, rc, win)
            s = r.run()
            summaries.append(s)
            rows += [dict(x, k_policy=kp, windows=win, refresh=pol if ping else "-", shed_p=shed if ping else "-")
                     for x in r.rows]
            extra = f" refresh {pol:<6} shed {shed:<4g} skipped {s['skipped_refresh']}+{s['skipped_pressure']}" \
                if ping else ""
            print(f"  cap {str(s['capacity']):>4}  win {win:<8} {cfg:<9} k {kp:<6}{extra}  p50 {s['p50_s']:>6} "
                  f"p95 {s['p95_s']:>6} p99 {s['p99_s']:>6}  max {s['max_s']:>7}  cold {s['cold_share']:>5}  "
                  f"waited {s['waited_for_slot']:>4}  pod-s {s['pod_seconds']:>8}")
    keys = list(dict.fromkeys(k for s in summaries for k in s))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summaries)
    with open(out / "calls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"args": vars(A), "summaries": summaries}, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
