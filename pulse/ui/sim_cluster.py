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
  - autoscaler: "window" (default, the UI demo): each pod lives until it has been idle for window_s.
    "kpa" (the capacity simulator): Knative's KPA with its defaults. Every 2 s each tool is sized by its average
    concurrency (requests in flight, waiting ones included) over the stable window (window_s, 60 s), at a target
    of 0.7 per pod (containerConcurrency 1 x 70 % utilization); a burst over the 6 s panic window that needs at
    least twice the ready pods puts the tool in panic mode for one stable window: it scales up to the burst and
    never down. Scale-down removes idle pods, at most half of them per step; a tool with no request for the whole
    stable window goes to zero. A call that finds no free pod waits for the first pod to free (the activator).
The tool itself runs in-process on the airline database (LocalToolBackend), so results are real.
"""

import itertools
import math
import random
import threading
import time

WARM_S = 0.04            # warm call overhead (measured in-cluster: ~40 ms)
DEFAULT_COLD_S = 1.6
PING_S = 0.005           # a warm-up / keep-alive ping on a ready pod
KPA_TARGET = 0.7         # requests in flight per pod the KPA aims for (containerConcurrency 1 x 70 %)
KPA_PANIC_WINDOW_S = 6.0
KPA_PANIC_THRESHOLD = 2.0
KPA_TICK_S = 2.0
KPA_MAX_SCALE_DOWN_RATE = 2.0


from pulse.controller.leases import measured_cold_s  # noqa: E402  (per-tool cold starts measured on EKS)


INF = float("inf")


class SimCluster:
    def __init__(self, tools: list[str], window_s: float = 60.0, max_scale: int = 10, min_scale: int = 0,
                 capacity: int | None = None, grace_s: float = 0.0, seed: int = 252, autoscaler: str = "window",
                 windows: dict[str, float] | None = None):
        self.tools, self.window_s, self.max_scale = list(tools), window_s, max_scale
        self.windows = dict(windows or {})    # per-tool idle windows (refresh.py lever 3); others use window_s
        self.capacity, self.grace_s = capacity, grace_s
        self.kpa = autoscaler == "kpa"
        self.reqs: dict[str, list[list]] = {t: [] for t in tools}   # kpa: [start, end or None] per request
        self.panic_until: dict[str, float] = {}
        self.last_eval = -INF
        self.cold = {t: DEFAULT_COLD_S for t in tools} | measured_cold_s()
        self.lock = threading.Lock()
        self.pods: dict[str, list[dict]] = {t: [] for t in tools}
        self.ids = itertools.count(1)
        self.rng = random.Random(seed)
        self.events: list[dict] = []          # prewarms / scale-downs, read by the UI
        self.pod_seconds = 0.0
        self.terminating: list[float] = []    # when the slots of scaled-down pods free (capacity only)
        self.pending_starts = 0               # pods that had to wait for a slot (capacity only)
        self.slot_waits: list[float] = []     # how long each of them waited (the learned top-k slot price reads it)
        self.set_min_scale(min_scale)

    # ------------------------------------------------------------------ state
    def win(self, tool: str) -> float:
        """The tool's idle window (Knative's autoscaling.knative.dev/window on its service)."""
        return float(self.windows.get(tool, self.window_s))

    def pending_pods(self) -> int:
        """Pods waiting for a slot (capacity only)."""
        with self.lock:
            return sum(1 for ps in self.pods.values() for p in ps if p.get("pending"))

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
            self.slot_waits.append(start - p["queued"])
            p.update(pending=False, created=start,
                     ready_at=start + self.cold.get(p["tool"], DEFAULT_COLD_S) * self.rng.uniform(0.9, 1.15))
            p["last_used"] = p["ready_at"]
            room -= 1

    # ------------------------------------------------------------------ Knative's KPA (autoscaler="kpa")
    def _avg(self, tool: str, now: float, w: float) -> float:
        """Average requests in flight for `tool` over the last w seconds."""
        s = 0.0
        for a, b in self.reqs.get(tool, []):
            o = min(now if b is None else b, now) - max(a, now - w)
            if o > 0:
                s += o
        return s / w

    def _kpa(self, now: float):
        """One autoscaler step per KPA_TICK_S: size every tool from its concurrency, as Knative's KPA does."""
        if now - self.last_eval < KPA_TICK_S:
            return
        self.last_eval = now
        for t, pods in self.pods.items():
            reqs = self.reqs.setdefault(t, [])
            w = self.win(t)
            reqs[:] = [r for r in reqs if r[1] is None or r[1] > now - w]                 # the stable window
            traffic = bool(reqs)
            ready = sum(1 for p in pods if not p.get("pending") and p["ready_at"] <= now)
            want_stable = math.ceil(self._avg(t, now, w) / KPA_TARGET - 1e-9)
            want_panic = math.ceil(self._avg(t, now, KPA_PANIC_WINDOW_S) / KPA_TARGET - 1e-9)
            if want_panic >= KPA_PANIC_THRESHOLD * max(1, ready):
                self.panic_until[t] = now + w
            if self.panic_until.get(t, -INF) > now:                 # panic: up to the burst, never down
                desired = max(want_panic, len(pods))
            else:
                desired = want_stable if traffic else 0
            busy = sum(1 for p in pods if p["busy"])
            desired = min(self.max_scale, max(desired, self.min_scale, busy, 1 if traffic and pods else 0))
            while len(pods) < desired:
                pods.append(self._new_pod(t, now))
            if len(pods) > desired:
                floor = math.ceil(len(pods) / KPA_MAX_SCALE_DOWN_RATE) if desired > 0 else 0
                goal = max(desired, floor)
                # Kubernetes' ReplicaSet order: Pending pods first, then not-ready (still starting), then the ready
                # pod that became ready most recently -- the longest-warm pod is the one kept
                idle = sorted((p for p in pods if not p["busy"]),
                              key=lambda p: (not p.get("pending"), p["ready_at"] <= now, -p["ready_at"]))
                for p in idle:
                    if len(pods) <= goal:
                        break
                    pods.remove(p)
                    if not p.get("pending"):
                        self.pod_seconds += now - p["created"]
                        if self.capacity is not None:
                            self.terminating.append(now + self.grace_s)
                        self.events.append({"time": now, "event": "scaled_down", "tool": t, "source": "sim",
                                            "pods_after": len(pods)})

    def _request(self, tool: str, start: float, end: float | None) -> list:
        r = [start, end]
        if self.kpa:
            self.reqs.setdefault(tool, []).append(r)
        return r

    def _reap(self, now: float):
        """Knative's autoscaler: KPA mode, or remove pods idle longer than the window (keep min_scale)."""
        if self.kpa:
            self._kpa(now)
            self._promote(now)
            return
        for t, pods in self.pods.items():
            for p in sorted(pods, key=lambda p: p["last_used"]):          # longest idle first
                if len(pods) <= self.min_scale:
                    break
                if not p["busy"] and not p.get("pending") and now - max(p["last_used"], p["ready_at"]) > self.win(t):
                    pods.remove(p)
                    # the pod ended when its idle window ran out, not when this check noticed it (the checks follow
                    # the UI's polling), so its pod time stops there: exact, like the per-episode figure
                    end = max(p["last_used"], p["ready_at"]) + self.win(t)
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
            self._request(tool, now, now + PING_S)
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
            if self.kpa:     # every pod lives until its tool has had no request for the stable window
                for t, pods in self.pods.items():
                    last = max((now if b is None else b for _, b in self.reqs.get(t, [])), default=now)
                    running = [p for p in pods if not p.get("pending")]
                    total += max(0.0, last + self.win(t) - now) * max(0, len(running) - self.min_scale)
                return total
            for t, pods in self.pods.items():
                # _reap removes the longest-idle pods first and keeps the min_scale most recently used ones
                ends = sorted((max(p["last_used"], p["ready_at"]) + self.win(t) for p in pods
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
            # kpa: the k pings are k requests in flight, each until its pod answers (a cold start for a new pod)
            for p in sorted(pods, key=lambda p: p["ready_at"])[:max(1, k)]:
                end = now + self.cold.get(tool, DEFAULT_COLD_S) if p.get("pending") else max(now, p["ready_at"])
                self._request(tool, now, end + PING_S)
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
            ready = [p for p in pods if not p["busy"] and not p.get("pending") and p["ready_at"] <= now]
            if ready:
                pod, start = ready[0], now
            else:
                # Knative's activator holds the request and hands it to whichever pod is free first: a busy one
                # finishing its call (~WARM_S), or one still starting. With no free pod at all, the autoscaler
                # adds one (Pending if the nodes are full) -- the request does not wait for that pod in particular.
                # kpa: only the activator's scale-from-zero here; more pods come from the next autoscaler step
                if (not pods if self.kpa else not any(not p["busy"] for p in pods)) and len(pods) < self.max_scale:
                    pods.append(self._new_pod(tool, now))
                pod = min(pods, key=lambda p: self._available(p, now))
                start = self._available(pod, now)
            pod["busy"] += 1
            if start != INF:
                pod["busy_until"] = start + WARM_S
            pod.setdefault("open", []).append(self._request(tool, now, None))     # in flight until release
            cold = pod.get("pending", False) or pod["ready_at"] > now    # it waited for a pod to start
            return (INF if start == INF else start - now), cold, pod

    @staticmethod
    def _available(p: dict, now: float) -> float:
        """When pod p can take one more call: after it has started and finished the calls already on it."""
        if p.get("pending"):
            return INF
        return max(now, p["ready_at"], p.get("busy_until", 0.0) if p["busy"] else 0.0)

    def release(self, pod: dict):
        with self.lock:
            pod["busy"] = max(0, pod["busy"] - 1)
            pod["last_used"] = time.time()
            if pod.get("open"):
                pod["open"].pop(0)[1] = pod["last_used"]
