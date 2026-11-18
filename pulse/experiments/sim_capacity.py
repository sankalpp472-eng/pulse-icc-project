"""
Capacity simulation: how prewarming behaves when the tool nodes fill up. No cluster, Docker or LLM: recorded
tau-bench episodes (the same plan as pulse/experiments/replay.py: traces, think times, customer turns, fan-out
groups, hint timing) run on the simulated cluster (pulse/ui/sim_cluster.py) with a pod capacity, on a virtual
clock, against the real predictor called in-process (pulse/predictor/service.py, CRD writes off).

Configurations (every one sees the same episodes, hints and predictions):
  C0        scale to zero, Knative window 60 s, no prewarming
  C1        one pod per tool always warm (min-scale 1), window 60 s
  C3-ping   today's controller (Option A): each predicted tool gets k pods at once; window 60 s is the TTL
  C3-lease  the lease controller (pulse/controller/leases.py): per-claim TTL, k lowered under pressure, a reserve
            of free pods for on-demand calls, lowest-value leases evicted; short window (--lease-window, 6 s)

Per configuration and capacity: tool-call latency p50 / p95 / p99 / max, cold share, pods that had to wait for a
slot, pod-seconds (cost), and the lease decisions.

  python -m pulse.experiments.sim_capacity                          # capacities 0 (no limit), 40, 24, 16 pods
  python -m pulse.experiments.sim_capacity --capacity 20 --rate 8 --episodes 80 --predictor markov

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
    ap.add_argument("--capacity", type=int, nargs="+", default=[0, 40, 24, 16],
                    help="pod slots on the tool nodes (0 = no limit)")
    ap.add_argument("--configs", nargs="+", default=["C0", "C1", "C3-ping", "C3-lease"])
    ap.add_argument("--predictor", default="lightgbm", help="lightgbm | markov | pulse_v3 (speculator: needs an LLM)")
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--rate", type=float, default=4.0, help="new episodes per minute (Poisson)")
    ap.add_argument("--seed", type=int, default=252)
    ap.add_argument("--think-median", type=float, default=1.5)
    ap.add_argument("--think-sigma", type=float, default=0.5)
    ap.add_argument("--user-turn-median", type=float, default=8.0)
    ap.add_argument("--user-turn-sigma", type=float, default=0.5)
    ap.add_argument("--no-fanout", action="store_true", help="run every call alone (no parallel groups)")
    ap.add_argument("--window", type=float, default=60.0, help="Knative idle window for C0 / C1 / C3-ping (s)")
    ap.add_argument("--lease-window", type=float, default=6.0, help="Knative idle window for C3-lease (s)")
    ap.add_argument("--grace", type=float, default=10.0, help="seconds a scaled-down pod still holds its slot")
    ap.add_argument("--reserve", type=int, default=4, help="C3-lease: free pods prewarms never take")
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--idle-ttl", type=float, default=60.0, help="C3-lease: idle warmth while there is room (s)")
    ap.add_argument("--idle-free", type=int, default=8, help="C3-lease: free pods above the reserve for idle warmth")
    ap.add_argument("--no-idle", action="store_true", help="C3-lease without idle warmth")
    ap.add_argument("--no-supersede", action="store_true", help="C3-lease: a newer hint never ends a claim early")
    ap.add_argument("--no-adapt-k", action="store_true", help="C3-lease: k as the predictor asks (no fan-out cap)")
    ap.add_argument("--ttl-step", type=int, default=20, help="predictor's TTL for a next-step prediction (s)")
    ap.add_argument("--ttl-ahead", type=int, default=75, help="predictor's TTL for a lookahead prediction (s)")
    ap.add_argument("--tick", type=float, default=0.25, help="simulation step for the autoscaler (s)")
    ap.add_argument("--label", default="")
    return ap.parse_args()


A = parse_args()
os.environ["PREDICTOR"], os.environ["THETA"], os.environ["CRD_MODE"] = A.predictor, str(A.theta), "off"
os.environ.setdefault("LOOKAHEAD", "0")
os.environ["TTL_STEP_S"], os.environ["TTL_AHEAD_S"] = str(A.ttl_step), str(A.ttl_ahead)

from pulse import paths  # noqa: E402
import pulse.ui.sim_cluster as sim_cluster  # noqa: E402
from pulse.controller.leases import LeaseBook, LeaseConfig  # noqa: E402
from pulse.experiments import replay  # noqa: E402
from pulse.predictor import client  # noqa: E402

WARM_S = sim_cluster.WARM_S


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
        """[(tool, p, k, ttl_s, lookahead), ...] the predictor's prewarm list for this hint."""
        if key not in self.cache:
            hint = client.make_hint(self.wanted, episode, trajectory, None, domain=replay.DOMAIN,
                                    signals=signals, history=history)
            out = self.svc.predict(self.Req(**hint))
            ahead = {a.tool for a in out.lookahead}
            self.cache[key] = [(a.tool, a.p, a.k or 1, a.ttl_s or out.ttl_s, a.tool in ahead) for a in out.prewarm]
        return self.cache[key]


# ---------------------------------------------------------------------- one configuration
class Run:
    def __init__(self, config: str, capacity: int, plan: list[dict], preds: Predictions, tools: list[str]):
        self.config, self.capacity, self.plan, self.preds = config, capacity, plan, preds
        lease = config == "C3-lease"
        CLOCK.t = 0.0                    # before the simulator exists: C1's always-warm pods are ready at t = 0
        self.sim = sim_cluster.SimCluster(tools, window_s=A.lease_window if lease else A.window,
                                          min_scale=1 if config == "C1" else 0,
                                          capacity=capacity or None, grace_s=A.grace, seed=A.seed)
        self.book = LeaseBook(LeaseConfig(reserve_pods=A.reserve, k_max=A.k_max, idle_ttl_s=A.idle_ttl,
                                          idle_free_pods=A.idle_free, adapt_k=not A.no_adapt_k,
                                          supersede_s=1e9 if A.no_supersede else 1.0),
                              cold_s=self.sim.cold) if lease else None
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
        for tool, p, k, ttl, ahead in items:
            if tool not in self.sim.pods:
                continue
            if self.config == "C3-ping":
                self.sim.prewarm(tool, k, p)
                continue
            d = self.book.request(tool, ep["episode"], p, k, ttl, CLOCK.t, self.sim.free_pods(), lookahead=ahead,
                                  busy=self.sim.busy_tools())
            running = sum(1 for x in self.sim.pods[tool] if not x.get("pending"))
            if d.action != "reject" and (d.extra > 0 or running == 0):
                self.sim.prewarm(tool, d.k, p)

    def tick(self):
        if self.book:
            self.book.expire(CLOCK.t, None if A.no_idle else self.sim.free_pods())
            for t in self.book.active():
                self.sim.keepalive(t)
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
            if self.book:
                self.book.end_episode(ep["episode"])
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
        if self.book and not A.no_idle:                # a real call: keep the tool warm a while if there is room
            self.book.adopt(c["tool"], CLOCK.t, self.sim.free_pods())
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
        out = {"config": self.config, "capacity": self.capacity or "none", "calls": n,
               "cold_share": round(sum(r["cold"] for r in self.rows) / n, 3) if n else None,
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
          f"predictor {A.predictor} θ {A.theta} | {len(tools)} tools | -> {out}")
    summaries, rows = [], []
    for cap in A.capacity:
        for cfg in A.configs:
            r = Run(cfg, cap, plan, preds, tools)
            s = r.run()
            summaries.append(s)
            rows += r.rows
            print(f"  cap {str(s['capacity']):>4}  {cfg:<9} p50 {s['p50_s']:>6} p95 {s['p95_s']:>6} p99 {s['p99_s']:>6}"
                  f"  max {s['max_s']:>6}  cold {s['cold_share']:>5}  waited {s['waited_for_slot']:>4}"
                  f"  pod-s {s['pod_seconds']:>8}")
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
