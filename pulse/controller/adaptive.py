"""
The k rule's two numbers, learned while the controller runs (K_ADAPT=1) instead of fixed settings.

  pods = min(k, ceil(k * warm_s / budget_s))          (leases.rule_k)

warm_s, a tool's warm call time: the median of its warm calls (pulse_tool_call_seconds{temperature="warm"}, which
  the runners export) read from Prometheus every WARM_POLL_S (PROM_URL, e.g. http://prometheus-operated.monitoring:9090).
  Until a tool has WARM_MIN_CALLS warm calls there, the configured TOOL_WARM_S / TOOL_WARM_S_JSON is used.
  No PROM_URL (no monitoring): the configured values throughout.

budget_s, the queueing one pod may add before another pod is worth it: K_BUDGET_S scaled by
  - the tool's cold start, learned from the controller's own pings to a tool at zero pods (a running average,
    seeded with REF_COLD_S): a pod that takes longer to start is less likely to be ready for the burst it is for
    (factor cold / REF_COLD_S, clamped to 0.5-3);
  - how full the tool nodes are: 1 + fullness, fullness = 1 - free slots / K_FREE_HIGH (0 with K_FREE_HIGH (6) or
    more free slots, 1 with none): on a full node an extra pod takes a slot a real call may wait for.
  So with room and a 1.5 s cold start it is K_BUDGET_S exactly, and up to 2 x 3 x K_BUDGET_S when full and slow to start.

The top-k of the adaptive prewarm policy (pulse/predictor/policy.py) is decided here too, one candidate at a time:
  warm it  iff  hitRate x the tool's cold start  >=  (1 - hitRate) x the price of a pod slot,
  price = TOPK_PRICE_S x (1 + (TOPK_FULL_X - 1) x fullness), fullness 1 while tool pods wait for a slot.
Only a wrong prewarm costs a slot: a right one starts the pod the call would have started a moment later. hitRate is
how often a candidate of its probability was really the next call (pulse/predictor/calibration.py); the cold start is
the same learned one as above. With room (price 0.075 s) a 1.5 s tool is warmed from a 5 % chance; with the nodes full
(0.75 s) from 33 %. So a step warms its top 3 when slots are free and only its likely ones when not. (Charging the
whole price, not just the miss's share, turned away the best guess on a full node: p99 77 s vs top-1's 1.8 s in the
simulator, retail at 15 slots.)

Learned slot price (TOPK_PRICE=learned): the full-node part of the price is measured instead of TOPK_FULL_X:
  price = TOPK_PRICE_S + fullness x W,
W = how long a tool pod waited for a node slot (created -> scheduled), a running average over the pods that waited
at least WAIT_MIN_S, capped at WAIT_MAX_S, seeded with TOPK_PRICE_S x (TOPK_FULL_X - 1) so that before any pod has
waited the price is exactly the fixed one. A wrong prewarm on a full node costs the call that needed its slot about
what pods are waiting now: with pods waiting ~10 s a 1.5 s tool is warmed only from 87 %, at ~30 s from 95 %.
TOPK_PRICE_S stays a setting: what a slot costs when nobody waits is a preference (pod-seconds vs latency).
"""

import json
import math
import os
import threading
import urllib.parse
import urllib.request

COLD_LATENCY_S = 1.0          # the runners' rule: a call slower than this was cold (experiments/common.py)
REF_COLD_S = float(os.getenv("REF_COLD_S", "1.5"))
WARM_MIN_CALLS = int(os.getenv("WARM_MIN_CALLS", "5"))
K_FREE_HIGH = int(os.getenv("K_FREE_HIGH", "6"))
TOPK_PRICE_S = float(os.getenv("TOPK_PRICE_S", "0.075"))
TOPK_FULL_X = float(os.getenv("TOPK_FULL_X", "10"))
WAIT_MIN_S = float(os.getenv("WAIT_MIN_S", "1"))      # a shorter wait is scheduling, not a queue for a slot
WAIT_MAX_S = float(os.getenv("WAIT_MAX_S", "60"))
WAIT_SEED_S = TOPK_PRICE_S * (TOPK_FULL_X - 1)        # the learned price starts as the fixed one
WARM_QUERY = ('histogram_quantile(0.5, sum by (le, tool) (rate(pulse_tool_call_seconds_bucket{temperature="warm"}[10m])))'
              ' and on (tool) sum by (tool) (increase(pulse_tool_calls_total{temperature="warm"}[10m])) >= %d')


def budget(base_s: float, cold_s: float, free: int, free_high: int = K_FREE_HIGH, ref_cold_s: float = REF_COLD_S) -> float:
    """The queueing budget for one tool now: base x cold-start factor x (1 + fullness)."""
    cold_f = min(3.0, max(0.5, (cold_s or ref_cold_s) / ref_cold_s))
    full = max(0.0, 1.0 - max(0, free) / free_high) if free_high > 0 else 0.0
    return base_s * cold_f * (1.0 + full)


def slot_price(free: int, pending: int = 0, price_s: float = TOPK_PRICE_S, full_x: float = TOPK_FULL_X,
               free_high: int = K_FREE_HIGH, wait_s: float | None = None) -> float:
    """What a pod slot is worth now, in seconds of cold start: low with room, full_x times that when full;
    with wait_s (the learned slot wait) price_s + fullness x wait_s instead."""
    full = 1.0 if pending > 0 else (max(0.0, 1.0 - max(0, free) / free_high) if free_high > 0 else 0.0)
    if wait_s is not None:
        return price_s + full * max(0.0, wait_s)
    return price_s * (1.0 + (full_x - 1.0) * full)


def worth_warming(hit_rate: float, cold_s: float, free: int, pending: int = 0, **kw) -> tuple[bool, str]:
    """Adaptive top-k, one candidate: is its expected saving worth the slot it may waste?
    A RIGHT prewarm takes no extra slot (the call would start that pod a moment later anyway) and saves the cold
    start; only a WRONG one holds a slot for nothing. So: hit x cold start >= (1 - hit) x slot price."""
    hit = min(1.0, max(0.0, float(hit_rate)))
    price = slot_price(free, pending, **kw)
    gain, waste = hit * float(cold_s), (1.0 - hit) * price
    return gain >= waste, (f"top-k: hit {hit:.2f} x cold {cold_s:.2f} s = {gain:.3f} s vs (1 - hit) x slot "
                           f"{price:.3f} s = {waste:.3f} s")


def parse_prom(result: dict) -> dict[str, float]:
    """{tool: seconds} from a Prometheus instant-query answer (NaN / missing values dropped)."""
    out = {}
    for r in (result.get("data") or {}).get("result") or []:
        tool = (r.get("metric") or {}).get("tool")
        try:
            v = float(r["value"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if tool and math.isfinite(v) and 0 < v < COLD_LATENCY_S:
            out[tool] = v
    return out


class Estimates:
    """Per-tool warm call time and cold start, learned at runtime, with the configured values as the fallback."""

    def __init__(self, warm_default: float, warm_config: dict[str, float] | None = None, alpha: float = 0.3):
        self.warm_default, self.warm_config = warm_default, dict(warm_config or {})
        self.alpha = alpha
        self.warm_learned: dict[str, float] = {}
        self.cold_learned: dict[str, float] = {}
        self.wait_learned: float | None = None      # slot wait (learned slot price)
        self.wait_seen: dict[str, None] = {}        # pod uids already counted, oldest first (bounded)
        self.lock = threading.Lock()

    def warm(self, tool: str) -> float:
        with self.lock:
            return self.warm_learned.get(tool, self.warm_config.get(tool, self.warm_default))

    def cold(self, tool: str) -> float:
        with self.lock:
            return self.cold_learned.get(tool, REF_COLD_S)

    def set_warm(self, measured: dict[str, float]) -> None:
        """Replace the learned warm times with the latest Prometheus medians (tools with too few calls keep theirs)."""
        with self.lock:
            self.warm_learned.update(measured)

    def observe_cold(self, tool: str, seconds: float) -> None:
        """A ping to a tool at zero pods took `seconds`: its cold start (running average)."""
        if seconds <= 0:
            return
        with self.lock:
            old = self.cold_learned.get(tool)
            self.cold_learned[tool] = seconds if old is None else (1 - self.alpha) * old + self.alpha * seconds

    def wait(self) -> float:
        """How long a tool pod waits for a node slot now (the seed until one has waited)."""
        with self.lock:
            return WAIT_SEED_S if self.wait_learned is None else self.wait_learned

    def observe_wait(self, uid: str, seconds: float) -> bool:
        """Tool pod `uid` waited `seconds` between created and scheduled. Each pod counts once; waits under
        WAIT_MIN_S are ignored. True if it was counted."""
        with self.lock:
            if uid in self.wait_seen:
                return False
            self.wait_seen[uid] = None
            if len(self.wait_seen) > 5000:
                self.wait_seen.pop(next(iter(self.wait_seen)))
            if seconds < WAIT_MIN_S:
                return False
            s = min(WAIT_MAX_S, seconds)
            old = self.wait_learned
            self.wait_learned = s if old is None else (1 - self.alpha) * old + self.alpha * s
            return True

    def snapshot(self) -> dict:
        with self.lock:
            return {"warm_s": {t: round(v, 4) for t, v in sorted(self.warm_learned.items())},
                    "cold_s": {t: round(v, 3) for t, v in sorted(self.cold_learned.items())},
                    "slot_wait_s": None if self.wait_learned is None else round(self.wait_learned, 2)}


def query_warm(prom_url: str, min_calls: int = WARM_MIN_CALLS, timeout: float = 5.0) -> dict[str, float]:
    """The warm-call medians per tool from Prometheus ({} if it cannot be reached)."""
    url = prom_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": WARM_QUERY % min_calls})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return parse_prom(json.loads(r.read()))
    except Exception:
        return {}
