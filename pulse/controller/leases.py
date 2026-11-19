"""
The lease policy of the prewarm controller (CONTROLLER_MODE=lease): admission control for prewarms. Pure logic with
no Kubernetes or clock of its own, so the real controller (prewarm_controller.py) and the simulator
(pulse/ui/sim_cluster.py, pulse/experiments/sim_capacity.py) make exactly the same decisions.

The tools keep Knative's 60 s idle window: a warm pod stays warm 60 s after its last call or ping, whatever the
controller does, and a pod that already exists is the cheapest one to use. What hurts p99 when the tool nodes are
full is the prewarms themselves: k pods started ahead of time take the slots that real calls then wait for (the ping
controller's p99 went from 1.9 s to 24 s on EKS with one worker; a 6 s window that dropped warm pods quickly made it
far worse, 156 s, by restarting pods all the time). So the lease controller decides how many NEW pods a prediction
may start, and nothing else:

  1. k      min(predicted k, k_max, learned cap). The learned cap is the tool's observed fan-out width (exponential
            average of how many calls one step really made; only the simulator sees calls), so a predictor that
            over-asks stops wasting pods.
  2. Admit  the new pods (k minus the pods the tool already has) fit in the free pod slots minus the reserve. The
            reserve is never used by prewarms: it is the room on-demand calls need.
  3. Reduce otherwise start only as many as fit; a tool that is already warm just gets its window refreshed.
  4. Reject a cold tool with no room left: its first real call starts it, as without prewarming.

The lease itself is the predictor's TTL for the claim (next step ~ think time, lookahead ~ customer reply), clamped
to [ttl_min, ttl_max]. The 60 s window already covers most leases; for one that outlives it (a long lookahead) the
controller sends one extending ping shortly before the window would run out (due_pings()).
"""

import csv
import math
from dataclasses import dataclass, field

from pulse import paths


def measured_cold_s() -> dict[str, float]:
    """Per-tool cold-start seconds from the newest EKS cold/warm check (data/coldstart/*_tools-cold-warm.csv);
    {} if there is none (every tool then counts as 1.6 s). Here, not in the UI package: the controller's image has
    no pulse/ui."""
    for f in reversed(sorted(paths.COLDSTART.glob("*_tools-cold-warm.csv"))):
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
            return {r["tool"]: float(r["cold_ms"]) / 1000 for r in rows if r.get("cold_ms")}
        except (OSError, KeyError, ValueError):
            continue
    return {}


def rule_k(k: int, warm_s: float, budget_s: float = 0.3) -> int:
    """How many pods a prediction of k parallel calls is worth: the smallest number whose queueing fits the budget.
    Each tool pod serves one call at a time and Knative's activator queues the rest on the first pod to free, so k
    calls on m warm pods finish after about ceil(k/m) calls' time. One more pod is worth its cold start, its slot
    and its 60 s of idling only if it removes more than budget_s of that queueing:

        m = min(k, ceil(k * warm_s / budget_s))

    Fast tools (the tau-bench tools: ~45 ms a call) get 1 pod for any k up to 6; a 0.5 s tool gets k. In the
    capacity simulator this matched the best fixed policy both ways: 1 pod for fast calls (same latency as k > 1 with
    room, ~20 % fewer pod-seconds, and none of the ping controller's tail when full), k for slow calls."""
    k = max(1, int(k or 1))
    if not warm_s or warm_s <= 0 or not budget_s or budget_s <= 0:
        return k
    return max(1, min(k, math.ceil(k * warm_s / budget_s - 1e-9)))


@dataclass
class LeaseConfig:
    ttl_min_s: float = 5.0          # shortest lease
    ttl_max_s: float = 120.0        # longest lease (a lookahead over a slow customer reply)
    k_max: int = 5                  # most pods one prediction may ask for (the CRD allows 0..5)
    reserve_pods: int = 4           # free pod slots prewarms never take: room for on-demand calls
    adapt_k: bool = True            # cap k at the tool's observed fan-out width (needs observe_fanout calls)
    k_alpha: float = 0.3            # weight of the newest observation in that average
    window_s: float = 60.0          # the tools' Knative idle window
    extend_margin_s: float = 5.0    # an extending ping goes this long before the window would run out


@dataclass
class Lease:
    tool: str
    expires: float = 0.0
    last_ping: float = 0.0          # the controller's last ping (the window counts from it at least)
    claims: dict = field(default_factory=dict)      # episode -> expiry


@dataclass
class Decision:
    action: str                 # admit | reduced | reject
    k: int = 0                  # the pods the tool should have (reduced: fewer than asked)
    new: int = 0                # pods this decision starts (k minus the pods the tool already has)
    reason: str = ""


class LeaseBook:
    def __init__(self, cfg: LeaseConfig | None = None):
        self.cfg = cfg or LeaseConfig()
        self.leases: dict[str, Lease] = {}
        self.width: dict[str, float] = {}           # tool -> average fan-out width seen
        self.stats = {"requests": 0, "admitted": 0, "reduced": 0, "rejected": 0, "refreshed": 0,
                      "pods_asked": 0, "pods_started": 0, "extended": 0, "expired": 0}

    # ------------------------------------------------------------------ adaptive k
    def observe_fanout(self, tool: str, width: int):
        """One step called `tool` `width` times at once."""
        if width < 1:
            return
        a = self.cfg.k_alpha
        self.width[tool] = width if tool not in self.width else (1 - a) * self.width[tool] + a * width

    def k_cap(self, tool: str) -> int:
        if not self.cfg.adapt_k or tool not in self.width:
            return self.cfg.k_max
        return max(1, min(self.cfg.k_max, math.ceil(self.width[tool] - 1e-9)))

    # ------------------------------------------------------------------ requests
    def request(self, tool: str, episode: str, k: int, ttl_s: float, now: float, free_pods: int,
                have_pods: int) -> Decision:
        """One prediction ("episode will probably call tool soon, k at once"). free_pods = free pod slots on the
        tool nodes now; have_pods = the tool's pods now (ready or starting)."""
        cfg = self.cfg
        self.stats["requests"] += 1
        want = max(1, min(int(k or 1), cfg.k_max, self.k_cap(tool)))
        need = max(0, want - have_pods)
        room = max(0, free_pods - cfg.reserve_pods)
        self.stats["pods_asked"] += need
        if need == 0:
            d = Decision("admit", have_pods, 0, "warm: refresh the window")
            self.stats["refreshed"] += 1
        elif need <= room:
            d = Decision("admit", want, need, "fits")
            self.stats["admitted"] += 1
        elif room >= 1:
            d = Decision("reduced", have_pods + room, room, f"k {want} -> {have_pods + room}: little room")
            self.stats["reduced"] += 1
        elif have_pods >= 1:
            d = Decision("reduced", have_pods, 0, f"k {want} -> {have_pods}: no room, refresh only")
            self.stats["reduced"] += 1
        else:
            self.stats["rejected"] += 1
            return Decision("reject", 0, 0, f"no room: {free_pods} free, reserve {cfg.reserve_pods}")
        self.stats["pods_started"] += d.new
        ttl = min(cfg.ttl_max_s, max(cfg.ttl_min_s, float(ttl_s or cfg.ttl_min_s)))
        lease = self.leases.setdefault(tool, Lease(tool))
        lease.claims[episode] = max(lease.claims.get(episode, 0.0), now + ttl)
        lease.expires = max(lease.claims.values())
        lease.last_ping = now                      # the decision's ping (or warm-up) restarts the window
        return d

    # ------------------------------------------------------------------ the lease clock
    def due_pings(self, now: float) -> list[str]:
        """Tools whose lease outlives the window counted from the last ping, and whose window runs out within the
        margin: send each one extending ping (then call pinged())."""
        cfg = self.cfg
        return [t for t, l in self.leases.items()
                if l.expires > l.last_ping + cfg.window_s and now >= l.last_ping + cfg.window_s - cfg.extend_margin_s]

    def pinged(self, tool: str, now: float):
        if tool in self.leases:
            self.leases[tool].last_ping = now
            self.stats["extended"] += 1

    def expire(self, now: float) -> list[str]:
        """Drop leases past their expiry (the pods stay until Knative's window ends); returns those tools."""
        gone = [t for t, l in self.leases.items() if l.expires <= now]
        for t in gone:
            del self.leases[t]
        self.stats["expired"] += len(gone)
        return gone

    def active(self) -> dict[str, float]:
        """tool -> lease expiry."""
        return {t: l.expires for t, l in self.leases.items()}
