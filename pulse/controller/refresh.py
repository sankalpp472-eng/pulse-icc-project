"""
How long prewarmed pods stay alive (ping controller), shared by the controller and the simulator.

In ping mode a tool's pod lives until Knative's idle window (autoscaling.knative.dev/window) passes with no
request, and ANY request starts the window again: a real call, or a prewarm ping. So a wrong prediction that
pings a warm tool keeps a pod nobody uses alive for another window, and under contention that pod holds a slot
real calls then wait for. Three levers, each off by default (= the behaviour before them):

  1. REFRESH_POLICY=never   a prewarm never pings a tool that already has its k pods: only real calls extend them
  2. REFRESH_POLICY=value   it does only when the prediction is worth the slot: p >= a threshold that is 0 with
                            plenty of free pod slots (REFRESH_FREE_HIGH or more) and rises to REFRESH_P_MAX when
                            none is free. (always = every prediction pings, as before.)
  3. per-tool windows       each tool's idle window from how soon it is really called again (tool_windows below,
                            pulse/experiments/tool_windows.py -> data/tool_windows.json; TOOL_WINDOWS=per-tool):
                            rarely reused tools give their slot back sooner, frequently reused ones keep 60 s
  4. SHED_P (pressure)      while tool pods wait for a slot (Pending), a prediction below SHED_P neither starts
                            new pods nor refreshes warm ones, so low-value pods age out within their window.
                            This is the eviction Knative allows: its autoscaler sets each tool's pod count from
                            the traffic in the window, so a deleted idle pod is simply recreated; a pod can only be
                            freed by not sending it traffic. 0 = off.
"""

import math
import os
from dataclasses import dataclass


@dataclass
class RefreshConfig:
    policy: str = "always"        # always | never | value
    free_high: int = 6            # free pod slots at or above which a refresh costs nothing (value policy)
    p_max: float = 0.8            # the threshold when no slot is free (value policy)
    shed_p: float = 0.0           # under pressure (pods waiting for a slot): predictions below this are dropped

    @classmethod
    def from_env(cls) -> "RefreshConfig":
        return cls(policy=os.getenv("REFRESH_POLICY", "always"), free_high=int(os.getenv("REFRESH_FREE_HIGH", "6")),
                   p_max=float(os.getenv("REFRESH_P_MAX", "0.8")), shed_p=float(os.getenv("SHED_P", "0")))

    def needs_capacity(self) -> bool:
        """Whether decide() reads free / pending pod counts (the controller then asks the cluster)."""
        return self.policy == "value" or self.shed_p > 0


def refresh_threshold(free: int, cfg: RefreshConfig) -> float:
    """value policy: the probability a refresh needs, 0 with free_high or more free slots, p_max with none."""
    if cfg.free_high <= 0:
        return 0.0
    return cfg.p_max * max(0.0, 1.0 - max(0, free) / cfg.free_high)


def decide(p: float, have: int, k: int, cfg: RefreshConfig, free: int = 0, pending: int = 0) -> tuple[str, str]:
    """What a prediction (probability p, k pods wanted) does to a tool that has `have` pods.
    -> (action, reason): action "start" (new pods: k pings), "refresh" (one ping: the window starts again) or "skip"."""
    p = float(p or 0.0)
    warm = have >= max(1, k)
    if pending > 0 and cfg.shed_p > 0 and p < cfg.shed_p:
        return "skip", f"pressure: {pending} pod(s) waiting for a slot and p {p:.2f} < {cfg.shed_p:g}"
    if not warm:
        return "start", f"{have} of {k} pod(s)"
    if cfg.policy == "never":
        return "skip", "warm: only real calls extend its pods (REFRESH_POLICY=never)"
    if cfg.policy == "value":
        th = refresh_threshold(free, cfg)
        if p < th:
            return "skip", f"warm: p {p:.2f} < {th:.2f} with {free} free slot(s)"
    return "refresh", "warm: the window starts again"


def tool_windows(gaps: dict[str, list[float]], q: float = 0.8, lo: int = 20, hi: int = 60) -> dict[str, int]:
    """Per-tool idle windows from the gaps between a tool's successive calls (seconds): the q-quantile of the
    gaps, clamped to [lo, hi] -- long enough that most re-uses find the pod still there, no longer.
    A tool whose re-uses mostly come later than hi gets lo (keeping it would rarely pay; prewarms cover it);
    a tool never called twice gets lo too."""
    out = {}
    for tool, g in gaps.items():
        g = sorted(x for x in g if x >= 0)
        if not g:
            out[tool] = lo
            continue
        v = g[min(len(g) - 1, max(0, math.ceil(q * len(g)) - 1))]
        out[tool] = lo if v > hi else int(min(hi, max(lo, math.ceil(v))))
    return out
