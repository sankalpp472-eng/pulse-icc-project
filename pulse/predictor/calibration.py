"""
How often a prewarm candidate is really the next call, learned while the predictor runs (adaptive top-k).

Every hint of an episode tells what the PREVIOUS step did: the tools added to the trajectory since the last hint
(none = the agent answered the customer). So each candidate the predictor offered gets its outcome one hint later,
and the calibrator counts hits per probability bin. Its estimate for a candidate of probability p is

    hit(p) = (hits in p's bin + M * p) / (candidates in p's bin + M)

i.e. the predictor's own p until the bin has data, then what really happened: a predictor that is overconfident
(pulse_v3 put a tool above 0.3 on almost every step and 70 % of those went unused) gets its rates pulled down.
The controller weighs hit(p) x the tool's cold start against the price of a pod slot (pulse/controller/adaptive.py).

An episode's last step is never seen (no next hint), so episodes idle for SESSION_TIMEOUT_S are just dropped.
"""

import os
import threading
import time

BINS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0001)
PRIOR_N = float(os.getenv("CALIB_PRIOR_N", "10"))
SESSION_TIMEOUT_S = float(os.getenv("SESSION_TIMEOUT_S", "300"))


def appended(prev: list, cur: list) -> list:
    """Tools added to a trajectory window since the last hint (hints carry the last 8 tools)."""
    for s in range(len(prev) + 1):
        if prev[s:] == cur[:len(prev) - s]:
            return cur[len(prev) - s:]
    return cur


def bin_of(p: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= p < BINS[i + 1]:
            return i
    return len(BINS) - 2


class Calibrator:
    def __init__(self, prior_n: float = PRIOR_N, timeout_s: float = SESSION_TIMEOUT_S):
        self.prior_n, self.timeout_s = prior_n, timeout_s
        self.hits = [0] * (len(BINS) - 1)
        self.n = [0] * (len(BINS) - 1)
        self.episodes: dict[str, dict] = {}
        self.lock = threading.Lock()

    def rate(self, p: float) -> float:
        b = bin_of(p)
        with self.lock:
            return (self.hits[b] + self.prior_n * p) / (self.n[b] + self.prior_n)

    def observe(self, episode: str, trajectory: list[str], candidates: list[tuple[str, float]],
                now: float | None = None) -> None:
        """One hint: score the episode's previous candidates against what that step did, remember these."""
        now = time.time() if now is None else now
        eid = episode or "default"
        with self.lock:
            for e, ep in list(self.episodes.items()):
                if now - ep["t"] > self.timeout_s:
                    self.episodes.pop(e)
            prev = self.episodes.get(eid)
            if prev is not None:
                called = set(appended(prev["traj"], list(trajectory)))
                for tool, p in prev["cands"]:
                    b = bin_of(p)
                    self.n[b] += 1
                    self.hits[b] += tool in called
            self.episodes[eid] = {"t": now, "traj": list(trajectory), "cands": list(candidates)}

    def stats(self) -> dict:
        with self.lock:
            return {"bins": [{"p": f"{BINS[i]:g}-{min(1, BINS[i + 1]):g}", "candidates": self.n[i],
                              "hit_rate": round(self.hits[i] / self.n[i], 3) if self.n[i] else None}
                             for i in range(len(BINS) - 1)], "open_episodes": len(self.episodes)}
