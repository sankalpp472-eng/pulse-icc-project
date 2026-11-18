"""
The lease policy of the prewarm controller: which prewarms to grant, with how many pods (k), for how long (TTL),
and which warm tools to give up when the tool nodes are full. Pure logic with no Kubernetes or clock of its own,
so the real controller (prewarm_controller.py, CONTROLLER_MODE=lease) and the simulator (pulse/ui/sim_cluster.py,
pulse/experiments/sim_capacity.py) make exactly the same decisions.

A lease = "keep this tool warm until expires". The controller keeps a leased tool alive with keep-alive pings; with
a short Knative idle window (~6 s) the tool scales down soon after its lease ends or is evicted. One lease per tool,
holding one claim per episode that asked for it:

  p        1 - prod(1 - p_claim): the chance at least one of those episodes calls the tool soon
  k        the largest claim's k, capped by what the tool's calls really fan out to (adaptive k, below)
  expires  the latest claim's expiry

Request handling (request()):
  1. TTL      the predictor's TTL for the claim (next step ~ think time, lookahead ~ customer reply), clamped to
              [ttl_min, ttl_max]. A claim ends early when the same episode's next hint no longer predicts the tool
              (supersede): the agent moved on.
  2. k        min(predicted k, k_max, learned cap): the learned cap is the tool's observed fan-out width
              (exponential average of how many calls one step really made), so a predictor that over-asks stops
              wasting pods. Under pressure k drops to 1 before anything is rejected.
  3. Admit    if the extra pods fit in free capacity minus the reserve. The reserve (pods) is never used by
              prewarms: it is the room on-demand calls (mispredictions, C0-style cold starts) need, which keeps
              their p99 from exploding when the cluster is full.
  4. Evict    otherwise, evict the lowest-value idle leases whose value is below the new claim's, until the extra
              pods fit; evicted tools stop getting keep-alive pings. Value = p * cold-start seconds saved / pods
              held, decayed by how far off the expiry is: the expected cold-start time saved per pod held.
  5. Reject   if even that does not make room (or nothing has a lower value).

Idle warmth (expire(..., free_pods), adopt()): with a short window a tool goes cold seconds after its last call,
where a 60 s window kept it warm for the next episode. So while free pods exceed reserve + idle_free_pods, a lease
that ends, or a tool a real call just used, keeps a low-value idle lease for idle_ttl_s. Idle leases are evicted
first and all shed the moment room gets tight: warmth when it is free, none when it would cost p99.
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


@dataclass
class LeaseConfig:
    ttl_min_s: float = 5.0          # shortest lease
    ttl_max_s: float = 120.0        # longest lease (a lookahead over a slow customer reply)
    k_max: int = 5                  # most pods one lease may hold (the CRD allows 0..5)
    reserve_pods: int = 4           # free pods prewarms never take: room for on-demand calls
    adapt_k: bool = True            # cap k at the tool's observed fan-out width
    k_alpha: float = 0.3            # weight of the newest observation in that average
    supersede_s: float = 1.0        # a claim older than this, from the same episode, is from an earlier hint
    decay_s: float = 60.0           # value halves for every decay_s of remaining lease (far-off calls count less)
    # idle warmth: while there is plenty of room, a tool stays warm idle_ttl_s after its lease ends or after a real
    # call (what a long Knative window gave for free), as a low-value lease that is evicted first and dropped as
    # soon as free pods fall below reserve + idle_free_pods
    idle_ttl_s: float = 60.0        # = the 60 s window the ping controller relies on
    idle_free_pods: int = 8
    idle_p: float = 0.05


IDLE = "_idle"                      # the episode name of an idle-warmth claim


@dataclass
class Claim:
    episode: str
    p: float
    k: int
    expires: float
    at: float
    lookahead: bool = False


@dataclass
class Lease:
    tool: str
    claims: dict = field(default_factory=dict)      # episode -> Claim
    since: float = 0.0

    @property
    def p(self) -> float:
        q = 1.0
        for c in self.claims.values():
            q *= 1.0 - min(1.0, max(0.0, c.p))
        return 1.0 - q

    @property
    def k(self) -> int:
        return max((c.k for c in self.claims.values()), default=0)

    @property
    def expires(self) -> float:
        return max((c.expires for c in self.claims.values()), default=0.0)


@dataclass
class Decision:
    action: str                 # admit | reduced | reject
    k: int = 0                  # pods granted (reduced: lowered under pressure)
    extra: int = 0              # pods the grant adds on top of what the tool's lease held
    evict: list = field(default_factory=list)       # tools whose leases were evicted to make room
    reason: str = ""
    ended: list = field(default_factory=list)       # tools whose lease ended: this episode's hint moved on


class LeaseBook:
    def __init__(self, cfg: LeaseConfig | None = None, cold_s: dict | None = None, default_cold_s: float = 1.6):
        self.cfg = cfg or LeaseConfig()
        self.cold_s, self.default_cold_s = dict(cold_s or {}), default_cold_s
        self.leases: dict[str, Lease] = {}
        self.width: dict[str, float] = {}           # tool -> average fan-out width seen
        self.stats = {"requests": 0, "admitted": 0, "reduced": 0, "rejected": 0, "evicted": 0,
                      "expired": 0, "superseded": 0, "idled": 0, "adopted": 0, "shed": 0}

    # ------------------------------------------------------------------ value
    def _value(self, tool: str, p: float, k: int, expires: float, now: float) -> float:
        """Expected cold-start seconds saved per pod held, discounted for a call that is far off."""
        left = max(0.0, expires - now)
        return p * self.cold_s.get(tool, self.default_cold_s) / max(1, k) * 0.5 ** (left / self.cfg.decay_s)

    def value(self, lease: Lease, now: float) -> float:
        return self._value(lease.tool, lease.p, lease.k, lease.expires, now)

    # ------------------------------------------------------------------ adaptive k
    def observe_fanout(self, tool: str, width: int):
        """One step called `tool` `width` times at once (the simulator sees calls; the real controller sees the
        most pods the tool reached during a lease)."""
        if width < 1:
            return
        a = self.cfg.k_alpha
        self.width[tool] = width if tool not in self.width else (1 - a) * self.width[tool] + a * width

    def k_cap(self, tool: str) -> int:
        if not self.cfg.adapt_k or tool not in self.width:
            return self.cfg.k_max
        return max(1, min(self.cfg.k_max, math.ceil(self.width[tool] - 1e-9)))

    # ------------------------------------------------------------------ requests
    def request(self, tool: str, episode: str, p: float, k: int, ttl_s: float, now: float, free_pods: int,
                lookahead: bool = False, busy: set | None = None) -> Decision:
        """One prediction ("episode will probably call tool"), with the tool nodes' free pod slots right now.
        busy = tools serving a call now (never evicted; the real controller cannot see this and passes None)."""
        cfg = self.cfg
        self.stats["requests"] += 1
        ended = self.supersede(episode, keep={tool}, now=now, free_pods=free_pods)
        ttl = min(cfg.ttl_max_s, max(cfg.ttl_min_s, float(ttl_s or cfg.ttl_min_s)))
        want = max(1, min(int(k or 1), cfg.k_max, self.k_cap(tool)))
        cur = self.leases.get(tool)
        held = cur.k if cur else 0
        room = free_pods - cfg.reserve_pods

        def grant(kk: int, action: str, evicted=(), reason=""):
            lease = self.leases.setdefault(tool, Lease(tool, since=now))
            lease.was_idle = False                             # a real prediction: it may idle again after
            old = lease.claims.get(episode)
            lease.claims[episode] = Claim(episode, p, kk, max(now + ttl, old.expires if old else 0.0), now,
                                          lookahead or (old.lookahead if old else False))
            self.stats["admitted" if action == "admit" else "reduced"] += 1
            return Decision(action, kk, max(0, kk - held), list(evicted), reason, ended)

        if want <= held:                                       # a refresh: no new pods
            return grant(held, "admit", reason="refresh")
        if want - held <= room:
            return grant(want, "admit", reason="fits")
        if held >= 1:                                          # keep what the tool holds, no extra pods
            return grant(held, "reduced", reason=f"k {want} -> {held}: little room")
        if room >= 1:                                          # degrade: one pod instead of k
            return grant(1, "reduced", reason=f"k {want} -> 1: little room")
        # not even one pod fits: evict lower-value idle leases
        need = 1 - room
        mine = self._value(tool, p, 1, now + ttl, now)
        busy = busy or set()
        victims, freed = [], 0
        for v in sorted((l for t, l in self.leases.items() if t != tool and t not in busy),
                        key=lambda l: self.value(l, now)):
            if freed >= need or self.value(v, now) >= mine:
                break
            victims.append(v.tool)
            freed += v.k
        if freed >= need:
            for t in victims:
                self.evict(t)
            return grant(1, "reduced" if want > 1 else "admit", victims,
                         reason=f"evicted {len(victims)} lower-value lease(s)")
        self.stats["rejected"] += 1
        return Decision("reject", 0, 0, [], f"full: {free_pods} free, reserve {cfg.reserve_pods}, "
                                            f"nothing cheaper to evict", ended)

    def supersede(self, episode: str, keep: set, now: float, free_pods: int | None = None) -> list[str]:
        """The episode's newer hint no longer predicts these tools: drop its older next-step claims on them
        (lookahead claims wait for the customer's reply, so they stay until their TTL). While there is plenty of
        room the tool keeps idle warmth instead (the agent may still call it a step later)."""
        gone = []
        roomy = self._roomy(free_pods)
        for t, lease in list(self.leases.items()):
            c = lease.claims.get(episode)
            if c and t not in keep and not c.lookahead and now - c.at > self.cfg.supersede_s:
                del lease.claims[episode]
                self.stats["superseded"] += 1
                if not lease.claims:
                    if roomy and not getattr(lease, "was_idle", False):
                        lease.claims[IDLE] = self._idle_claim(now)
                        lease.was_idle = True
                        self.stats["idled"] += 1
                        continue
                    del self.leases[t]
                    gone.append(t)
        return gone

    def end_episode(self, episode: str) -> list[str]:
        gone = []
        for t, lease in list(self.leases.items()):
            if lease.claims.pop(episode, None) and not lease.claims:
                del self.leases[t]
                gone.append(t)
        return gone

    def evict(self, tool: str):
        if self.leases.pop(tool, None):
            self.stats["evicted"] += 1

    def _roomy(self, free_pods: int | None) -> bool:
        return free_pods is not None and free_pods - self.cfg.reserve_pods >= self.cfg.idle_free_pods

    def _idle_claim(self, now: float) -> Claim:
        return Claim(IDLE, self.cfg.idle_p, 1, now + self.cfg.idle_ttl_s, now)

    def expire(self, now: float, free_pods: int | None = None) -> list[str]:
        """Drop claims past their expiry; returns the tools whose lease ended. With free_pods: a lease that ends
        while there is plenty of room becomes an idle-warmth lease instead, and idle leases are shed at once when
        room gets tight."""
        gone = []
        roomy = self._roomy(free_pods)
        for t, lease in list(self.leases.items()):
            for e, c in list(lease.claims.items()):
                if c.expires <= now or (e == IDLE and free_pods is not None and not roomy):
                    if e == IDLE and c.expires > now:
                        self.stats["shed"] += 1
                    del lease.claims[e]
            if not lease.claims:
                if roomy and not getattr(lease, "was_idle", False):
                    lease.claims[IDLE] = self._idle_claim(now)        # keep it warm a while: there is room
                    lease.was_idle = True
                    self.stats["idled"] += 1
                    continue
                del self.leases[t]
                self.stats["expired"] += 1
                gone.append(t)
        return gone

    def adopt(self, tool: str, now: float, free_pods: int) -> bool:
        """A real call just used a tool no lease holds: keep it warm idle_ttl_s if there is plenty of room (the
        simulator sees calls; the real controller sees a new pod Knative started for one)."""
        if tool in self.leases or not self._roomy(free_pods):
            return False
        lease = self.leases[tool] = Lease(tool, since=now)
        lease.claims[IDLE] = self._idle_claim(now)
        lease.was_idle = True
        self.stats["adopted"] += 1
        return True

    def active(self) -> dict[str, int]:
        """tool -> pods its lease holds: what the keep-alive loop keeps warm."""
        return {t: l.k for t, l in self.leases.items()}

    def held_pods(self) -> int:
        return sum(l.k for l in self.leases.values())
