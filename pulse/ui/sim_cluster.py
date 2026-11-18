"""
A simulated Knative cluster for demos without a real one (no Docker / kind / EKS needed).

Each of the 14 tools has pods, like a Knative Service with min-scale 0, containerConcurrency 1:
  - a call takes a free ready pod (warm: ~40 ms), waits for a starting pod, or starts a new pod
    (cold: the tool's measured cold-start time, from data/coldstart/*_tools-cold-warm.csv on EKS)
  - a pod with no traffic for `window_s` (Knative's idle window, 60 s) is removed: scale to zero
  - prewarm(tool, k) starts pods until the tool has k, exactly what the controller's k pings do
  - capacity (optional): at most this many pods run at once on the tool nodes. A pod that does not fit waits as
    Pending until one scales down and its slot frees (grace_s later: Knative's termination), then cold-starts.
    keepalive(tool) is the lease controller's ping: traffic that keeps the tool's pods from scaling down.
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


from pulse.controller.leases import measured_cold_s  # noqa: E402  (per-tool cold starts measured on EKS)


INF = float("inf")


class SimCluster:
    def __init__(self, tools: list[str], window_s: float = 60.0, max_scale: int = 10, min_scale: int = 0,
                 capacity: int | None = None, grace_s: float = 0.0, seed: int = 252):
        self.tools, self.window_s, self.max_scale = list(tools), window_s, max_scale
        self.capacity, self.grace_s = capacity, grace_s
        self.cold = {t: DEFAULT_COLD_S for t in tools} | measured_cold_s()
        self.lock = threading.Lock()
        self.pods: dict[str, list[dict]] = {t: [] for t in tools}
        self.ids = itertools.count(1)
        self.rng = random.Random(seed)
        self.events: list[dict] = []          # prewarms / scale-downs, read by the UI
        self.pod_seconds = 0.0
        self.terminating: list[float] = []    # when the slots of scaled-down pods free (capacity only)
        self.pending_starts = 0               # pods that had to wait for a slot (capacity only)
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

    def _running(self, now: float) -> int:
        """Slots in use: started pods, plus scaled-down pods still terminating."""
        return sum(1 for ps in self.pods.values() for p in ps if not p.get("pending")) + \
            sum(1 for t in self.terminating if t > now)

    def _new_pod(self, tool: str, now: float, ready_now: bool = False) -> dict:
        if self.capacity is not None and not ready_now and self._running(now) >= self.capacity:
            self.pending_starts += 1           # no slot: Pending until one frees (_promote)
            return {"id": next(self.ids), "created": now, "queued": now, "pending": True, "tool": tool,
                    "ready_at": INF, "last_used": now, "busy": 0}
        cold = 0.0 if ready_now else self.cold.get(tool, DEFAULT_COLD_S) * self.rng.uniform(0.9, 1.15)
        return {"id": next(self.ids), "created": now, "ready_at": now + cold, "last_used": now + cold, "busy": 0}

    def _promote(self, now: float):
        """Pending pods take the slots that freed, oldest first, and start then (a cold start from that moment)."""
        if self.capacity is None:
            return
        freed = sorted(t for t in self.terminating if t <= now)
        self.terminating = [t for t in self.terminating if t > now]
        pending = sorted((p for ps in self.pods.values() for p in ps if p.get("pending")), key=lambda p: p["queued"])
        room = self.capacity - self._running(now)
        for p in pending:
            if room <= 0:
                break
            start = max(p["queued"], freed.pop(0) if freed else p["queued"])
            p.update(pending=False, created=start,
                     ready_at=start + self.cold.get(p["tool"], DEFAULT_COLD_S) * self.rng.uniform(0.9, 1.15))
            p["last_used"] = p["ready_at"]
            room -= 1

    def _reap(self, now: float):
        """Knative's autoscaler: remove pods idle longer than the window (keep min_scale)."""
        for t, pods in self.pods.items():
            for p in sorted(pods, key=lambda p: p["last_used"]):          # longest idle first
                if len(pods) <= self.min_scale:
                    break
                if not p["busy"] and not p.get("pending") and now - max(p["last_used"], p["ready_at"]) > self.window_s:
                    pods.remove(p)
                    # the pod ended when its idle window ran out, not when this check noticed it (the checks follow
                    # the UI's polling), so its pod time stops there: exact, like the per-episode figure
                    end = max(p["last_used"], p["ready_at"]) + self.window_s
                    self.pod_seconds += end - p["created"]
                    if self.capacity is not None:
                        self.terminating.append(end + self.grace_s)
                    self.events.append({"time": end, "event": "scaled_down", "tool": t, "source": "sim",
                                        "pods_after": len(pods)})
        self._promote(now)

    def tick(self, now: float | None = None):
        """Advance the autoscaler (scale-downs, pending pods starting): the batch simulator calls it as time moves."""
        with self.lock:
            self._reap(time.time() if now is None else now)

    def free_pods(self) -> int:
        """Free pod slots now (a large number when there is no capacity limit)."""
        if self.capacity is None:
            return 10 ** 6
        now = time.time()
        with self.lock:
            self._reap(now)
            pending = sum(1 for ps in self.pods.values() for p in ps if p.get("pending"))
            return self.capacity - self._running(now) - pending

    def busy_tools(self) -> set:
        with self.lock:
            return {t for t, ps in self.pods.items() if any(p["busy"] for p in ps)}

    def keepalive(self, tool: str):
        """The lease controller's keep-alive ping: traffic, so the tool's started pods do not scale down."""
        now = time.time()
        with self.lock:
            for p in self.pods.get(tool, []):
                if not p.get("pending"):
                    p["last_used"] = max(p["last_used"], now)

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
            return self.pod_seconds + sum(max(0.0, now - p["created"]) for ps in self.pods.values() for p in ps
                                          if not p.get("pending"))

    def pending_pod_seconds(self) -> float:
        """Pod time still to come if no traffic arrives: every pod that will scale down lives on until its idle
        window has passed (the same rule as _reap). Added to an episode's pod-seconds, it gives the pod time the
        episode caused. The min_scale pods each tool keeps (C1) are not counted: they never scale down."""
        now = time.time()
        with self.lock:
            total = 0.0
            for pods in self.pods.values():
                # _reap removes the longest-idle pods first and keeps the min_scale most recently used ones
                ends = sorted((max(p["last_used"], p["ready_at"]) + self.window_s for p in pods
                               if not p.get("pending")), reverse=True)
                total += sum(max(0.0, end - now) for end in ends[self.min_scale:])
            return total

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
        cold = no ready pod was free, so the call waited for a pod to start (fully or partly).
        With a capacity limit the pod may be Pending: the wait is then inf until a slot frees (pod["ready_at"]
        becomes known); the batch simulator watches it."""
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
                wait, cold = max(0.0, pod["ready_at"] - now) + WARM_S, pod.get("pending", False)
            pod["busy"] += 1
            return wait, cold, pod

    def release(self, pod: dict):
        with self.lock:
            pod["busy"] = max(0, pod["busy"] - 1)
            pod["last_used"] = time.time()
