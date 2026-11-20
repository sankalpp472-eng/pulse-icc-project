"""
Prewarm controller, Option A: "warm-up request + Knative's idle window as the TTL".

Watches PrewarmRequest objects (infra/prewarm/prewarmrequest-crd.yaml), written by the predictor
(pulse/predictor/crd_writer.py with CRD_MODE=k8s). For every NEW or REFRESHED request (spec.requestedAt
newer than the last one handled) it sends the tool's Knative service MCP `ping`s through the
ingress -- the same path the agent's tool calls take:

    tool at zero  -> Knative's activator starts a pod (a cold start, but before the agent needs it)
    tool warm     -> the ping just counts as activity, so the idle window starts again

spec.desiredReplicas = k (the predictor's fan-out width): if the tool has fewer than k pods, k pings go
out at once. With containerConcurrency 1 (pulse/agent/gen_services.py) each needs its own pod, so
Knative starts k pods for the coming burst of k parallel calls. Otherwise one ping.

It never evicts anything: Knative scales the tool back to zero after its idle window
(autoscaling.knative.dev/window) without traffic, so that window IS the TTL. A sweeper marks
objects Evicted once their tool is back at zero pods, for the record (NFR-8 / E7).

CONTROLLER_MODE=lease instead (pulse/controller/leases.py): admission control for prewarms, with the same 60 s
window. The controller reads the tool nodes' free pod slots (allocatable pods, and CPU in tool pods of TOOL_CPU_M,
minus what runs there and the tool pods waiting for a slot) and lets a prediction start only the NEW pods that fit
next to a reserve kept for on-demand calls: all of them, fewer (k lowered), or none (a cold tool is rejected; a
warm one just gets its window refreshed). status.phase / grantedReplicas / decision record it. A lease lasts
spec.ttlSeconds; one that outlives the window gets one extending ping before the window runs out.

Run from the project root, with the agent's Python (needs `kubernetes`), next to the predictor:

    python -m pulse.controller.prewarm_controller

or inside the cluster as the pulse-controller Deployment (infra/cloud/), where it uses the pod's
service account and calls the tools at http://{svc}.default.svc.cluster.local/mcp.

Settings (environment):
  KUBE_CONTEXT       cluster to watch      (default: current kubectl context; ignored in the cluster)
  CRD_NAMESPACE      namespace             (default: default)
  MCP_URL_TEMPLATE   ingress URL for a tool (default http://127.0.0.1:31080/mcp; EKS: set by infra/eks/use_eks.sh)
  MCP_HOST_TEMPLATE  Host header           (default {svc}.default.example.com; empty = no Host header)
  MAX_REQUEST_AGE_S  ignore requests older than this, e.g. left over from an earlier run (default 30)
  CONTROLLER_LOG     one JSON line per action (default results/controller.jsonl)
  CONTROLLER_METRICS_PORT  Prometheus /metrics on this port (default 9102; 0 = off), see infra/monitoring/
  CONTROLLER_MODE    ping (default, Option A) | lease
  both modes, how many pods a prediction of k parallel calls gets (leases.rule_k):
  K_RULE             1 (default): min(k, ceil(k * the tool's warm call time / K_BUDGET_S)); 0 = k as predicted
  K_BUDGET_S         queueing one pod may add before another pod is worth it (default 0.3)
  TOOL_WARM_S        a warm call's time in seconds, every tool (default 0.05: the tau-bench tools take ~45 ms)
  TOOL_WARM_S_JSON   per-tool times overriding it, e.g. {"search_onestop_flight": 0.4}
  K_ADAPT            1: learn both at runtime (pulse/controller/adaptive.py): warm times from Prometheus (PROM_URL,
                     every WARM_POLL_S 60; the settings above until a tool has WARM_MIN_CALLS 5 warm calls), the
                     budget scaled by the tool's cold start (from this controller's pings to tools at zero pods,
                     vs REF_COLD_S 1.5) and by how full the tool nodes are (K_FREE_HIGH 6 free slots = not full)
  adaptive top-k (both modes; the predictor's PREWARM_POLICY=adaptive, its candidates carry spec.hitRate):
  TOPK_PRICE_S       a pod slot's price with room, in seconds of cold start (default 0.075: a 1.5 s tool from a 5 %
                     chance); a candidate is warmed only if hitRate x the tool's cold start reaches (1 - hitRate) x
                     the price (only a wrong prewarm wastes a slot)
  TOPK_FULL_X        the price's multiple when the tool nodes are full or pods wait for a slot (default 10)
  TOPK_PRICE         fixed (default: the two settings above) | learned: price = TOPK_PRICE_S + fullness x how long
                     tool pods are waiting for a node slot (measured here from the pods' created -> scheduled times;
                     seeded so it starts as the fixed price; adaptive.py)
  ping mode, how long prewarmed pods live (pulse/controller/refresh.py; every one off by default):
  REFRESH_POLICY     always (default: every prediction pings) | never (a warm tool is not pinged: only real calls
                     extend its pods) | value (pinged only if p >= a threshold rising from 0 at REFRESH_FREE_HIGH
                     free slots to REFRESH_P_MAX at none; defaults 6, 0.8)
  SHED_P             while tool pods wait for a slot, predictions below this neither start nor refresh pods (0 = off)
  per-tool idle windows: set on the Knative services (TOOL_WINDOWS=per-tool in run_cloud.sh / the UI), not here
  lease mode:
  LEASE_RESERVE_PODS free pod slots prewarms never take (default 4)
  LEASE_K_MAX        most pods one prediction may ask for (default 5)
  LEASE_TTL_MIN_S, LEASE_TTL_MAX_S   clamp for spec.ttlSeconds (default 5, 120)
  TOOL_WINDOW_S      the tools' Knative idle window (default 60)
  TOOL_NODES         label selector of the nodes the tools run on (default pulse/role=worker; none found = every
                     node without a taint, e.g. kind); cordoned nodes count as full
  TOOL_CPU_M         CPU a tool pod requests in millicores, Knative's queue-proxy sidecar included, to turn free CPU
                     into pod slots (default 75 = the tool's 50 + queue-proxy's 25)
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from prometheus_client import Counter, Gauge, Histogram, start_http_server

GROUP, VERSION, PLURAL = "pulse.iisc.ac.in", "v1alpha1", "prewarmrequests"
from pulse import paths  # noqa: E402
from pulse.controller.leases import LeaseBook, LeaseConfig, rule_k  # noqa: E402
from pulse.controller.refresh import RefreshConfig, decide  # noqa: E402
from pulse.controller.adaptive import K_FREE_HIGH, Estimates, budget, query_warm, worth_warming  # noqa: E402

MAX_REPLICAS = 5     # desiredReplicas upper bound: keep equal to `maximum` in infra/prewarm/prewarmrequest-crd.yaml and K_CAP in predictor/action_model.py
NS = os.getenv("CRD_NAMESPACE", "default")
URL_TPL = os.getenv("MCP_URL_TEMPLATE", "http://127.0.0.1:31080/mcp")
HOST_TPL = os.getenv("MCP_HOST_TEMPLATE", "{svc}.default.example.com")
MAX_AGE_S = float(os.getenv("MAX_REQUEST_AGE_S", "30"))
LOG = Path(os.getenv("CONTROLLER_LOG") or paths.RESULTS / "controller.jsonl")
PING = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
METRICS_PORT = int(os.getenv("CONTROLLER_METRICS_PORT", "9102"))
MODE = os.getenv("CONTROLLER_MODE", "ping")             # ping | lease
LEASE_TICK_S = 2.0                                       # how often the lease clock runs
K_RULE = os.getenv("K_RULE", "1") != "0"
K_BUDGET_S = float(os.getenv("K_BUDGET_S", "0.3"))
TOOL_WARM_S = float(os.getenv("TOOL_WARM_S", "0.05"))
TOOL_WARM = json.loads(os.getenv("TOOL_WARM_S_JSON") or "{}")
REFRESH = RefreshConfig.from_env()                       # ping mode: refresh / skip / shed (refresh.py)
K_ADAPT = os.getenv("K_ADAPT", "0") == "1"               # learn the k rule's warm time and budget (adaptive.py)
PROM_URL = os.getenv("PROM_URL", "")
WARM_POLL_S = float(os.getenv("WARM_POLL_S", "60"))
TOPK_PRICE = os.getenv("TOPK_PRICE", "fixed")            # fixed | learned (adaptive.py)
if TOPK_PRICE not in ("fixed", "learned"):
    raise SystemExit(f"TOPK_PRICE must be fixed or learned, not {TOPK_PRICE!r}")
ESTIMATES = Estimates(TOOL_WARM_S, TOOL_WARM)


def price_kw() -> dict:
    """worth_warming's extra arguments: the learned slot wait with TOPK_PRICE=learned, none otherwise."""
    return {"wait_s": ESTIMATES.wait()} if TOPK_PRICE == "learned" else {}


def slot_wait_s(pod) -> float | None:
    """How long a scheduled pod waited for a node: PodScheduled time - created (None if not scheduled yet)."""
    for c in (pod.status and pod.status.conditions) or []:
        if c.type == "PodScheduled" and c.status == "True" and c.last_transition_time and pod.metadata.creation_timestamp:
            return (c.last_transition_time - pod.metadata.creation_timestamp).total_seconds()
    return None


def k_inputs(tool: str, free: int | None = None) -> tuple[float, float]:
    """(warm call time, queueing budget) for the k rule: the settings, or with K_ADAPT=1 what was learned at
    runtime (adaptive.py: warm medians from Prometheus, the tool's measured cold start, how full the nodes are)."""
    if not K_ADAPT:
        return float(TOOL_WARM.get(tool, TOOL_WARM_S)), K_BUDGET_S
    return ESTIMATES.warm(tool), budget(K_BUDGET_S, ESTIMATES.cold(tool), K_FREE_HIGH if free is None else free)


def pods_for(tool: str, k_predicted, free: int | None = None) -> int:
    """The pods a prediction gets: the predictor's fan-out k (at most MAX_REPLICAS, #34), cut by the k rule
    (fast tools: 1)."""
    k = min(MAX_REPLICAS, max(1, int(k_predicted or 1)))
    return rule_k(k, *k_inputs(tool, free)) if K_RULE else k
TOOL_NODES = os.getenv("TOOL_NODES", "pulse/role=worker")
TOOL_CPU_M = int(os.getenv("TOOL_CPU_M", "75"))         # 50m tool + 25m Knative queue-proxy (measured on EKS)
LEASE_CFG = LeaseConfig(reserve_pods=int(os.getenv("LEASE_RESERVE_PODS", "4")),
                        k_max=int(os.getenv("LEASE_K_MAX", "5")),
                        ttl_min_s=float(os.getenv("LEASE_TTL_MIN_S", "5")),
                        ttl_max_s=float(os.getenv("LEASE_TTL_MAX_S", "120")),
                        window_s=float(os.getenv("TOOL_WINDOW_S", "60")),
                        adapt_k=False)      # the controller cannot see calls, only pods: k varies with room instead

# ---- Prometheus metrics (FR-11): what the controller did, as counters the dashboards can rate()
WARMUPS = Counter("pulse_controller_warmups_total", "Warm-ups done (PrewarmRequests acted on)", ["tool", "cold"])
PINGS = Counter("pulse_controller_pings_total", "Warm-up MCP pings sent", ["tool"])
PING_SECONDS = Histogram("pulse_controller_ping_seconds", "Warm-up ping round trip (about a cold start if cold)",
                         ["cold"], buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 60))
FAILURES = Counter("pulse_controller_failures_total", "Warm-ups that failed", ["tool"])
EVICTIONS = Counter("pulse_controller_evictions_total", "Tools seen back at zero pods after a warm-up", ["tool"])
WARM_SECONDS = Histogram("pulse_controller_warm_seconds", "Time from a warm-up's readyAt to the tool's scale-down",
                         buckets=(30, 60, 75, 90, 120, 180, 300, 600))
LEASES = Counter("pulse_controller_lease_decisions_total", "Lease mode: what happened to a lease",
                 ["decision"])     # admit | reduced | reject | extended | expired
FREE_PODS = Gauge("pulse_controller_free_pods", "Lease mode: free pod slots on the tool nodes")
LEASED = Gauge("pulse_controller_leases", "Lease mode: live leases")
NEW_PODS = Counter("pulse_controller_lease_pods_total", "Lease mode: new pods asked for and started by prewarms",
                   ["kind"])       # asked | started
SKIPS = Counter("pulse_controller_skips_total", "Predictions not acted on (refresh.py, adaptive.py)",
                ["why"])           # refresh (a warm tool not worth refreshing) | pressure (shed while pods wait) |
                                   # topk (an adaptive top-k candidate not worth a slot now)


def parse_cpu_m(q: str) -> int:
    """Kubernetes CPU quantity -> millicores ('1930m' -> 1930, '2' -> 2000)."""
    q = str(q)
    return int(float(q[:-1])) if q.endswith("m") else int(float(q) * 1000)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Controller:
    def __init__(self):
        if os.getenv("KUBERNETES_SERVICE_HOST"):       # running in the cluster (infra/cloud/)
            config.load_incluster_config()
        else:                                          # running on the laptop
            config.load_kube_config(context=os.getenv("KUBE_CONTEXT") or None)
        self.crd = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.in_flight: set[str] = set()
        self.lock = threading.Lock()
        LOG.parent.mkdir(parents=True, exist_ok=True)
        # lease mode
        self.book = LeaseBook(LEASE_CFG)
        self.objects: dict[str, tuple[str, str]] = {}         # tool -> (PrewarmRequest name, Knative service)
        self._free: tuple[float, int] = (0.0, 0)              # (when read, free pod slots), cached for 2 s
        self._pending = 0                                     # tool pods waiting for a slot, read with _free

    # ------------------------------------------------------------------ helpers
    def log(self, **rec):
        rec = {"time": now_iso(), **rec}
        print(json.dumps(rec))
        with self.lock, open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def pods(self, svc: str) -> int:
        """Ready + starting pods of the service (Knative labels every pod with its service)."""
        items = self.core.list_namespaced_pod(NS, label_selector=f"serving.knative.dev/service={svc}").items
        return sum(1 for p in items if p.metadata.deletion_timestamp is None and p.status.phase in ("Running", "Pending"))

    def window_s(self, svc: str) -> int | None:
        try:
            ks = self.crd.get_namespaced_custom_object("serving.knative.dev", "v1", NS, "services", svc)
            w = ks["spec"]["template"]["metadata"].get("annotations", {}).get("autoscaling.knative.dev/window", "60s")
            return int(float(w.rstrip("s")))
        except (ApiException, KeyError, ValueError):
            return None

    def set_status(self, name: str, **status):
        try:
            self.crd.patch_namespaced_custom_object_status(GROUP, VERSION, NS, PLURAL, name, {"status": status})
        except ApiException as e:
            self.log(event="status_patch_failed", object=name, error=f"{e.status} {e.reason}")

    # ------------------------------------------------------------------ the warm-up
    def ping(self, svc: str, tool: str) -> tuple[int, int]:
        """One MCP ping through the ingress. Returns (HTTP code, ms)."""
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if HOST_TPL:                   # empty in the cluster: the URL itself names the service
            headers["Host"] = HOST_TPL.format(svc=svc, tool=tool)
        req = urllib.request.Request(URL_TPL.format(svc=svc, tool=tool), data=PING, method="POST", headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code              # any answer FROM THE POD (even 4xx) proves it is up ...
        return code, int((time.perf_counter() - t0) * 1000)

    def skip(self, name: str, spec: dict, requested: str, why: str, kind: str, **extra):
        """A prediction not acted on: status Skipped, one log line, the skips metric."""
        SKIPS.labels(kind).inc()
        self.set_status(name, phase="Skipped", lastRequestedAt=requested, message=why[:300])
        self.log(event="skipped", why=kind, tool=spec["tool"], service=spec["service"],
                 probability=spec.get("probability"), rank=spec.get("rank"), hit_rate=spec.get("hitRate"),
                 episode=spec.get("episodeId"), reason=why, **extra)

    def warm(self, obj: dict, k: int | None = None):
        name, spec = obj["metadata"]["name"], obj["spec"]
        svc, requested = spec["service"], spec["requestedAt"]
        try:
            before = self.pods(svc)
            cold = before == 0
            # desiredReplicas = the predictor's fan-out width k. With containerConcurrency 1 every in-flight
            # request needs its own pod, so k pings at once make Knative start k pods. A tool that already
            # has k pods gets one ping, which just restarts its idle window. Lease mode passes the granted k.
            # Either way at most MAX_REPLICAS, the CRD schema's cap, enforced here too (#34).
            topk = spec.get("hitRate") is not None    # an adaptive top-k candidate (pulse/predictor/policy.py)
            room = self.capacity() if (REFRESH.needs_capacity() or K_ADAPT or topk) and MODE == "ping" else None
            k = min(MAX_REPLICAS, max(1, int(k))) if k else \
                pods_for(spec["tool"], spec.get("desiredReplicas"), room and room[0])
            if MODE == "ping":
                if topk:                              # adaptive.py: worth a pod slot now?
                    ok, why = worth_warming(spec["hitRate"], ESTIMATES.cold(spec["tool"]), *room, **price_kw())
                    if not ok:
                        return self.skip(name, spec, requested, why, "topk", pods_before=before, free_pods=room[0],
                                         pending_pods=room[1])
                free, pending = room if REFRESH.needs_capacity() else (0, 0)
                action, why = decide(spec.get("probability") or 0.0, before, k, REFRESH, free, pending)
                if action == "skip":                  # refresh.py: not worth refreshing, or shed under pressure
                    return self.skip(name, spec, requested, why, "pressure" if why.startswith("pressure") else
                                     "refresh", pods_before=before, desired_replicas=k, free_pods=free,
                                     pending_pods=pending, refresh_policy=REFRESH.policy)
            n = k if before < k else 1
            ttl = self.window_s(svc)
            sent = now_iso()
            self.set_status(name, phase="Warming", lastRequestedAt=requested, lastPingAt=sent,
                            wasCold=cold, effectiveTtlSeconds=ttl, message=f"{n} warm-up ping(s) sent")
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(lambda _: self.ping(svc, spec["tool"]), range(n)))
            codes = [c for c, _ in results]
            ms = max(m for _, m in results)
            if cold:                                  # a ping to a tool at zero pods: its cold start (adaptive.py)
                ESTIMATES.observe_cold(spec["tool"], ms / 1000)
            bad = [c for c in codes if c in (404, 502, 503, 504)]   # ... but these come from the ingress/activator
            if bad:
                raise RuntimeError(f"ingress answered HTTP {bad[0]} to {len(bad)} of {n} pings")
            ready = now_iso()
            expires = iso(parse_iso(ready) + ttl) if ttl else None
            if MODE == "lease":                       # the lease, not the window, decides when it goes cold
                with self.lock:
                    lease = self.book.leases.get(spec["tool"])
                expires = iso(lease.expires) if lease else expires
            self.set_status(name, phase="Ready", readyAt=ready, pingLatencyMs=ms, expiresAt=expires,
                            evictedAt=None, message=f"{n} ping(s) answered (HTTP {codes[0]})")
            WARMUPS.labels(spec["tool"], str(cold).lower()).inc()
            PINGS.labels(spec["tool"]).inc(n)
            PING_SECONDS.labels(str(cold).lower()).observe(ms / 1000)
            self.log(event="warmed", tool=spec["tool"], service=svc, was_cold=cold, ping_ms=ms, http=codes[0],
                     desired_replicas=k, predicted_k=spec.get("desiredReplicas"), pings=n, pods_before=before,
                     pods_after=self.pods(svc),
                     probability=spec.get("probability"), episode=spec.get("episodeId"), requested_at=requested,
                     effective_ttl_s=ttl, **dict(zip(("k_warm_s", "k_budget_s"),
                                                     (round(x, 4) for x in k_inputs(spec["tool"],
                                                                                   room and room[0])))),
                     **({"slot_wait_s": round(ESTIMATES.wait(), 2)} if TOPK_PRICE == "learned" else {}))
        except Exception as e:
            FAILURES.labels(spec.get("tool", "")).inc()
            self.set_status(name, phase="Failed", message=f"{type(e).__name__}: {e}"[:300])
            self.log(event="failed", tool=spec.get("tool"), service=svc, error=f"{type(e).__name__}: {e}")
        finally:
            with self.lock:
                self.in_flight.discard(svc)

    def handle(self, obj: dict):
        spec, status = obj.get("spec", {}), obj.get("status") or {}
        requested = spec.get("requestedAt")
        if not requested or requested == status.get("lastRequestedAt"):
            return                                  # nothing new (e.g. our own status update)
        age = time.time() - parse_iso(requested)
        if age > MAX_AGE_S:
            return                                  # stale: from before this controller started
        if MODE == "lease":
            return self.lease(obj)
        svc = spec["service"]
        with self.lock:
            if svc in self.in_flight:               # a ping is already warming this tool
                return
            self.in_flight.add(svc)
        self.pool.submit(self.warm, obj)

    # ------------------------------------------------------------------ lease mode
    def free_pods(self) -> int:
        """Free pod slots on the tool nodes: allocatable pods and CPU (in tool pods of TOOL_CPU_M, queue-proxy
        included) minus what runs there, and minus the tool pods waiting for a slot (they get the next free one).
        Cached for 2 s."""
        when, val = self._free
        if time.time() - when < 2:
            return val
        nodes = self.core.list_node(label_selector=TOOL_NODES).items
        if not nodes:                               # no labelled tool nodes (kind): every untainted node
            nodes = [n for n in self.core.list_node().items if not n.spec.taints]
        nodes = [n for n in nodes if not n.spec.unschedulable]     # a cordoned node takes no new pods
        names = {n.metadata.name for n in nodes}
        pods_alloc = sum(int(n.status.allocatable.get("pods", "110")) for n in nodes)
        cpu_alloc = sum(parse_cpu_m(n.status.allocatable.get("cpu", "0")) for n in nodes)
        used_pods = used_cpu = pending = 0
        for p in self.core.list_pod_for_all_namespaces(field_selector="status.phase!=Succeeded,status.phase!=Failed").items:
            tool_pod = "serving.knative.dev/service" in (p.metadata.labels or {})
            pending_tool = tool_pod and not p.spec.node_name
            pending += pending_tool
            if tool_pod and TOPK_PRICE == "learned" and p.spec.node_name:
                waited = slot_wait_s(p)
                if waited is not None and ESTIMATES.observe_wait(p.metadata.uid, waited):
                    self.log(event="slot_wait", pod=p.metadata.name, wait_s=round(waited, 2),
                             slot_wait_s=round(ESTIMATES.wait(), 2))
            if p.spec.node_name in names or pending_tool:
                used_pods += 1
                used_cpu += sum(parse_cpu_m(((c.resources and c.resources.requests) or {}).get("cpu", "0"))
                                for c in p.spec.containers)
        free = max(0, min(pods_alloc - used_pods, (cpu_alloc - used_cpu) // max(1, TOOL_CPU_M)))
        self._free = (time.time(), free)
        self._pending = pending
        FREE_PODS.set(free)
        return free

    def capacity(self) -> tuple[int, int]:
        """(free pod slots, tool pods waiting for a slot), from the same 2 s cache."""
        free = self.free_pods()
        return free, self._pending

    def lease(self, obj: dict):
        """One request in lease mode: ask the lease book how many new pods may start, record it, warm the tool."""
        name, spec = obj["metadata"]["name"], obj["spec"]
        tool, svc, requested = spec["tool"], spec["service"], spec["requestedAt"]
        self.objects[tool] = (name, svc)
        free, have = self.free_pods(), self.pods(svc)
        if spec.get("hitRate") is not None:         # adaptive top-k: is this candidate worth a slot now?
            ok, why = worth_warming(spec["hitRate"], ESTIMATES.cold(tool), free, self._pending, **price_kw())
            if not ok:
                return self.skip(name, spec, requested, why, "topk", had=have, free_pods=free,
                                 pending_pods=self._pending)
        want = pods_for(tool, spec.get("desiredReplicas"), free)
        with self.lock:
            d = self.book.request(tool, spec.get("episodeId") or name, want,
                                  float(spec.get("ttlSeconds") or 0), time.time(), free, have)
            lease = self.book.leases.get(tool)
            start = d.action != "reject" and svc not in self.in_flight
            if start:
                self.in_flight.add(svc)
            LEASED.set(len(self.book.leases))
        LEASES.labels(d.action).inc()
        NEW_PODS.labels("asked").inc(max(0, min(want, LEASE_CFG.k_max) - have))
        NEW_PODS.labels("started").inc(d.new)
        if d.new:                                   # pods are about to start: the cached free count is stale
            self._free = (0.0, 0)
        self.log(event="leased" if d.action != "reject" else "rejected", tool=tool, service=svc, action=d.action,
                 granted=d.k, new_pods=d.new, had=have, asked=spec.get("desiredReplicas"), ttl_s=spec.get("ttlSeconds"),
                 free_pods=free, reason=d.reason, probability=spec.get("probability"), episode=spec.get("episodeId"))
        if d.action == "reject":
            self.set_status(name, phase="Rejected", lastRequestedAt=requested, grantedReplicas=0, decision=d.reason,
                            message="no room for this prewarm: the first real call starts the tool")
            return
        self.set_status(name, grantedReplicas=d.k, leaseExpiresAt=iso(lease.expires), decision=f"{d.action}: {d.reason}")
        if start:                                   # k pings if the tool has fewer than k pods, else one (refresh)
            self.pool.submit(self.warm, obj, max(1, d.k))
        else:                                       # being warmed right now: note that this one was seen
            self.set_status(name, lastRequestedAt=requested)

    def lease_clock(self):
        """Lease mode: end expired leases; send the one extending ping a lease needs when it outlives the window."""
        while True:
            time.sleep(LEASE_TICK_S)
            try:
                now = time.time()
                with self.lock:
                    ended = self.book.expire(now)
                    due = [(t, self.objects.get(t, (None, None))) for t in self.book.due_pings(now)]
                    for t, _ in due:
                        self.book.pinged(t, now)
                    LEASED.set(len(self.book.leases))
                for t in ended:
                    name = self.objects.get(t, (None, None))[0]
                    LEASES.labels("expired").inc()
                    if name:
                        self.set_status(name, phase="Expired", decision="TTL passed: Knative's window takes over")
                for t, (name, svc) in due:
                    LEASES.labels("extended").inc()
                    self.pool.submit(self._extend, svc, t)
            except Exception as e:      # keep the clock running even if one round fails
                print(f"lease clock error: {type(e).__name__}: {e}")

    def _extend(self, svc: str, tool: str):
        try:
            code, ms = self.ping(svc, tool)
            self.log(event="extended", tool=tool, service=svc, http=code, ping_ms=ms)
        except Exception as e:
            self.log(event="extend_failed", tool=tool, service=svc, error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ loops
    def warm_poll(self):
        """K_ADAPT with PROM_URL: refresh the tools' warm call times from Prometheus every WARM_POLL_S."""
        while True:
            got = query_warm(PROM_URL)
            if got:
                ESTIMATES.set_warm(got)
                self.log(event="k_inputs", source="prometheus", **ESTIMATES.snapshot())
            time.sleep(WARM_POLL_S)

    def sweep(self):
        """Mark Ready objects Evicted once Knative has scaled their tool back to zero."""
        while True:
            time.sleep(5)
            try:
                for obj in self.crd.list_namespaced_custom_object(GROUP, VERSION, NS, PLURAL)["items"]:
                    st = obj.get("status") or {}
                    if st.get("phase") == "Ready" and self.pods(obj["spec"]["service"]) == 0:
                        at = now_iso()
                        self.set_status(obj["metadata"]["name"], phase="Evicted", evictedAt=at,
                                        message="Knative scaled the tool to zero (idle window passed)")
                        ready = parse_iso(st.get("readyAt"))
                        EVICTIONS.labels(obj["spec"]["tool"]).inc()
                        if ready:
                            WARM_SECONDS.observe(parse_iso(at) - ready)
                        self.log(event="evicted", tool=obj["spec"]["tool"], service=obj["spec"]["service"],
                                 warm_for_s=round(parse_iso(at) - ready, 1) if ready else None,
                                 effective_ttl_s=st.get("effectiveTtlSeconds"))
            except Exception as e:   # keep sweeping even if one round fails
                print(f"sweep error: {type(e).__name__}: {e}")

    def run(self):
        ctx = "in-cluster" if os.getenv("KUBERNETES_SERVICE_HOST") else config.list_kube_config_contexts()[1]["name"]
        from pulse.net import bind_host
        mode = "lease" if MODE == "lease" else "Option A"
        print(f"prewarm controller ({mode}) | context {ctx} | namespace {NS} | ingress {URL_TPL} | log {LOG}")
        if MODE == "lease":
            print(f"leases: reserve {LEASE_CFG.reserve_pods} pods, k max {LEASE_CFG.k_max}, "
                  f"TTL {LEASE_CFG.ttl_min_s:g}-{LEASE_CFG.ttl_max_s:g} s over a {LEASE_CFG.window_s:g} s window, "
                  f"tool nodes {TOOL_NODES}, {TOOL_CPU_M}m per tool pod")
        if METRICS_PORT:
            try:
                start_http_server(METRICS_PORT, addr=bind_host())     # this machine only on a laptop (pulse/net.py)
                print(f"metrics: http://{bind_host()}:{METRICS_PORT}/metrics")
            except OSError as e:     # e.g. a second controller on the same laptop: run without metrics
                print(f"metrics port {METRICS_PORT} not available ({e}); no /metrics")
        # both modes record when Knative scaled a tool back to zero; lease mode also runs the lease clock
        threading.Thread(target=self.sweep, daemon=True).start()
        if TOPK_PRICE == "learned":
            print(f"top-k slot price: learned (seed wait {ESTIMATES.wait():.3f} s, from the tool pods' slot waits)")
        if K_ADAPT:
            print(f"k rule: adaptive (cold starts from the pings; warm times "
                  f"{'from ' + PROM_URL if PROM_URL else 'as configured: no PROM_URL'})")
            if PROM_URL:
                threading.Thread(target=self.warm_poll, daemon=True).start()
        if MODE == "lease":
            threading.Thread(target=self.lease_clock, daemon=True).start()
        while True:      # the watch times out every few minutes; just start it again
            try:
                for ev in watch.Watch().stream(self.crd.list_namespaced_custom_object, GROUP, VERSION, NS,
                                               PLURAL, timeout_seconds=300):
                    if ev["type"] in ("ADDED", "MODIFIED"):
                        self.handle(ev["object"])
            except ApiException as e:
                if e.status == 404:
                    raise SystemExit("PrewarmRequest CRD not installed: kubectl apply -f infra/prewarm/prewarmrequest-crd.yaml")
                print(f"watch error {e.status}; retrying in 2 s")
                time.sleep(2)
            except Exception as e:
                print(f"watch error: {type(e).__name__}: {e}; retrying in 2 s")
                time.sleep(2)


if __name__ == "__main__":
    Controller().run()
