"""
A simulated Knative cluster for demos without a real one (no Docker / kind / EKS needed).

Each of the 14 tools has pods, like a Knative Service with min-scale 0, containerConcurrency 1:
  - a call takes a free ready pod (warm: ~40 ms), waits for a starting pod, or starts a new pod
    (cold: the tool's measured cold-start time, from data/coldstart/*_tools-cold-warm.csv on EKS)
  - a pod with no traffic for `window_s` (Knative's idle window, 60 s) is removed: scale to zero
  - prewarm(tool, k) starts pods until the tool has k, exactly what the controller's k pings do
The tool itself runs in-process on the airline database (LocalToolBackend), so results are real.
"""

import csv
import itertools
import random
import threading
import time

from pulse import paths
WARM_S = 0.04            # warm call overhead (measured in-cluster: ~40 ms)
DEFAULT_COLD_S = 1.6


def measured_cold_s() -> dict[str, float]:
    """Per-tool cold-start seconds from the newest EKS cold/warm check (fallback: 1.6 s for all)."""
    files = sorted(paths.COLDSTART.glob("*_tools-cold-warm.csv"))
    for f in reversed(files):
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
            return {r["tool"]: float(r["cold_ms"]) / 1000 for r in rows if r.get("cold_ms")}
        except (OSError, KeyError, ValueError):
            continue
    return {}


class SimCluster:
    def __init__(self, tools: list[str], window_s: float = 60.0, max_scale: int = 10, min_scale: int = 0):
        self.tools, self.window_s, self.max_scale = list(tools), window_s, max_scale
        self.cold = {t: DEFAULT_COLD_S for t in tools} | measured_cold_s()
        self.lock = threading.Lock()
        self.pods: dict[str, list[dict]] = {t: [] for t in tools}
        self.ids = itertools.count(1)
        self.rng = random.Random(252)
        self.events: list[dict] = []          # prewarms / scale-downs, read by the UI
        self.pod_seconds = 0.0
        self.set_min_scale(min_scale)

    # ------------------------------------------------------------------ state
    def set_min_scale(self, n: int):
        """0 = scale to zero (C0 / C3), 1 = always warm (C1)."""
        with self.lock:
            now = time.time()
            if getattr(self, "min_scale", n) != n:
                self.events.append({"time": now, "event": "min_scale", "tool": "", "source": "sim", "min_scale": n})
            self.min_scale = n
            for t in self.tools:
                while len(self.pods[t]) < n:
                    self.pods[t].append(self._new_pod(t, now, ready_now=True))

    def _new_pod(self, tool: str, now: float, ready_now: bool = False) -> dict:
        cold = 0.0 if ready_now else self.cold.get(tool, DEFAULT_COLD_S) * self.rng.uniform(0.9, 1.15)
        return {"id": next(self.ids), "created": now, "ready_at": now + cold, "last_used": now + cold, "busy": 0}

    def _reap(self, now: float):
        """Knative's autoscaler: remove pods idle longer than the window (keep min_scale)."""
        for t, pods in self.pods.items():
            for p in sorted(pods, key=lambda p: p["last_used"]):          # longest idle first
                if len(pods) <= self.min_scale:
                    break
                if not p["busy"] and now - max(p["last_used"], p["ready_at"]) > self.window_s:
                    pods.remove(p)
                    self.pod_seconds += now - p["created"]
                    self.events.append({"time": now, "event": "scaled_down", "tool": t, "source": "sim",
                                        "pods_after": len(pods)})

    def snapshot(self) -> list[dict]:
        now = time.time()
        with self.lock:
            self._reap(now)
            return [{"tool": t, "ready": sum(p["ready_at"] <= now for p in ps),
                     "starting": sum(p["ready_at"] > now for p in ps),
                     "busy": sum(p["busy"] > 0 for p in ps)} for t, ps in self.pods.items()]

    def live_pod_seconds(self) -> float:
        now = time.time()
        with self.lock:
            return self.pod_seconds + sum(now - p["created"] for ps in self.pods.values() for p in ps)

    # ------------------------------------------------------------------ the controller
    def prewarm(self, tool: str, k: int = 1, probability: float | None = None) -> dict | None:
        """Start pods until the tool has k (what the controller's k concurrent pings make Knative do)."""
        if tool not in self.pods:
            return None
        now = time.time()
        with self.lock:
            self._reap(now)
            pods = self.pods[tool]
            before = len(pods)
            for p in pods:                 # a ping is traffic: the idle window starts again
                p["last_used"] = max(p["last_used"], now)
            while len(pods) < min(max(1, k), self.max_scale):
                pods.append(self._new_pod(tool, now))
            ev = {"time": now, "event": "warmed", "tool": tool, "source": "sim", "was_cold": before == 0,
                  "desired_replicas": k, "pods_before": before, "pods_after": len(pods), "probability": probability}
            self.events.append(ev)
            return ev

    # ------------------------------------------------------------------ a tool call
    def acquire(self, tool: str) -> tuple[float, bool, dict]:
        """Reserve a pod for one call. Returns (seconds to wait before it can serve, cold?, pod); release(pod) after.
        cold = no ready pod was free, so the call waited for a pod to start (fully or partly)."""
        now = time.time()
        with self.lock:
            self._reap(now)
            pods = self.pods.setdefault(tool, [])
            free = [p for p in pods if not p["busy"]]
            ready = [p for p in free if p["ready_at"] <= now]
            if ready:
                pod, wait, cold = ready[0], 0.0, False
            elif free:                                            # a prewarmed pod still starting
                pod = min(free, key=lambda p: p["ready_at"])
                wait, cold = pod["ready_at"] - now, True
            elif len(pods) < self.max_scale:                      # the activator starts a new pod
                pod = self._new_pod(tool, now)
                pods.append(pod)
                wait, cold = pod["ready_at"] - now, True
            else:                                                 # all busy at max scale: queue on the soonest
                pod = min(pods, key=lambda p: p["ready_at"])
                wait, cold = max(0.0, pod["ready_at"] - now) + WARM_S, False
            pod["busy"] += 1
            return wait, cold, pod

    def release(self, pod: dict):
        with self.lock:
            pod["busy"] = max(0, pod["busy"] - 1)
            pod["last_used"] = time.time()
