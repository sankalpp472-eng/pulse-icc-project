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

CONTROLLER_MODE=lease instead (pulse/controller/leases.py; give the tools a short idle window, ~6 s):
every request asks for a lease of spec.ttlSeconds with k pods. The controller reads the tool nodes' free pod
slots (allocatable pods and CPU minus what runs there), admits the lease if it fits next to a reserve of free pods
kept for on-demand calls, lowers k, evicts lower-value leases, or rejects it (status.phase, status.decision).
A keep-alive loop pings every leased tool every KEEPALIVE_S; when a lease expires, is superseded by the episode's
next hint, or is evicted, the pings stop and Knative scales the tool down within its short window.

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
  lease mode:
  KEEPALIVE_S        seconds between keep-alive pings to a leased tool (default 2; keep it under the idle window)
  LEASE_RESERVE_PODS free pods prewarms never take (default 4)
  LEASE_K_MAX        most pods one lease holds (default 5)
  LEASE_TTL_MIN_S, LEASE_TTL_MAX_S   clamp for spec.ttlSeconds (default 5, 120)
  TOOL_NODES         label selector of the nodes the tools run on (default pulse/role=worker; none found = every
                     node without a taint, e.g. kind)
  TOOL_CPU_M         a tool pod's CPU request in millicores, to turn free CPU into pod slots (default 50)
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
from pulse.controller.leases import LeaseBook, LeaseConfig  # noqa: E402
NS = os.getenv("CRD_NAMESPACE", "default")
URL_TPL = os.getenv("MCP_URL_TEMPLATE", "http://127.0.0.1:31080/mcp")
HOST_TPL = os.getenv("MCP_HOST_TEMPLATE", "{svc}.default.example.com")
MAX_AGE_S = float(os.getenv("MAX_REQUEST_AGE_S", "30"))
LOG = Path(os.getenv("CONTROLLER_LOG") or paths.RESULTS / "controller.jsonl")
PING = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
METRICS_PORT = int(os.getenv("CONTROLLER_METRICS_PORT", "9102"))
MODE = os.getenv("CONTROLLER_MODE", "ping")             # ping | lease
KEEPALIVE_S = float(os.getenv("KEEPALIVE_S", "2"))
TOOL_NODES = os.getenv("TOOL_NODES", "pulse/role=worker")
TOOL_CPU_M = int(os.getenv("TOOL_CPU_M", "50"))
LEASE_CFG = LeaseConfig(reserve_pods=int(os.getenv("LEASE_RESERVE_PODS", "4")),
                        k_max=int(os.getenv("LEASE_K_MAX", "5")),
                        ttl_min_s=float(os.getenv("LEASE_TTL_MIN_S", "5")),
                        ttl_max_s=float(os.getenv("LEASE_TTL_MAX_S", "120")),
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
                 ["decision"])     # admit | reduced | reject | evicted | expired | superseded
FREE_PODS = Gauge("pulse_controller_free_pods", "Lease mode: free pod slots on the tool nodes")
HELD_PODS = Gauge("pulse_controller_leased_pods", "Lease mode: pods the live leases hold")


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
        from pulse.ui.sim_cluster import measured_cold_s     # the cold-start times a lease's value is weighed by
        self.book = LeaseBook(LEASE_CFG, cold_s=measured_cold_s())
        self.objects: dict[str, tuple[str, str]] = {}         # tool -> (PrewarmRequest name, Knative service)
        self._free: tuple[float, int] = (0.0, 0)              # (when read, free pod slots), cached for 2 s

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

    def warm(self, obj: dict, k: int | None = None):
        name, spec = obj["metadata"]["name"], obj["spec"]
        svc, requested = spec["service"], spec["requestedAt"]
        try:
            before = self.pods(svc)
            cold = before == 0
            # desiredReplicas = the predictor's fan-out width k. With containerConcurrency 1 every in-flight
            # request needs its own pod, so k pings at once make Knative start k pods. A tool that already
            # has k pods gets one ping, which just restarts its idle window. Lease mode passes the granted k.
            k = max(1, int(k or spec.get("desiredReplicas") or 1))
            n = k if before < k else 1
            ttl = self.window_s(svc)
            sent = now_iso()
            self.set_status(name, phase="Warming", lastRequestedAt=requested, lastPingAt=sent,
                            wasCold=cold, effectiveTtlSeconds=ttl, message=f"{n} warm-up ping(s) sent")
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(lambda _: self.ping(svc, spec["tool"]), range(n)))
            codes = [c for c, _ in results]
            ms = max(m for _, m in results)
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
                     desired_replicas=k, pings=n, pods_before=before, pods_after=self.pods(svc),
                     probability=spec.get("probability"), episode=spec.get("episodeId"), requested_at=requested,
                     effective_ttl_s=ttl)
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
        """Free pod slots on the tool nodes: allocatable pods and CPU (in tool-pod requests) minus what runs there,
        counting unscheduled tool pods too (they are about to take room). Cached for 2 s."""
        when, val = self._free
        if time.time() - when < 2:
            return val
        nodes = self.core.list_node(label_selector=TOOL_NODES).items
        if not nodes:                               # no labelled tool nodes (kind): every untainted node
            nodes = [n for n in self.core.list_node().items if not n.spec.taints]
        names = {n.metadata.name for n in nodes}
        pods_alloc = sum(int(n.status.allocatable.get("pods", "110")) for n in nodes)
        cpu_alloc = sum(parse_cpu_m(n.status.allocatable.get("cpu", "0")) for n in nodes)
        used_pods = used_cpu = 0
        for p in self.core.list_pod_for_all_namespaces(field_selector="status.phase!=Succeeded,status.phase!=Failed").items:
            pending_tool = not p.spec.node_name and "serving.knative.dev/service" in (p.metadata.labels or {})
            if p.spec.node_name in names or pending_tool:
                used_pods += 1
                used_cpu += sum(parse_cpu_m(((c.resources and c.resources.requests) or {}).get("cpu", "0"))
                                for c in p.spec.containers)
        free = max(0, min(pods_alloc - used_pods, (cpu_alloc - used_cpu) // max(1, TOOL_CPU_M)))
        self._free = (time.time(), free)
        FREE_PODS.set(free)
        return free

    def lease(self, obj: dict):
        """One request in lease mode: ask the lease book, record the decision, warm the tool if it needs pods."""
        name, spec = obj["metadata"]["name"], obj["spec"]
        tool, svc, requested = spec["tool"], spec["service"], spec["requestedAt"]
        self.objects[tool] = (name, svc)
        free = self.free_pods()
        cold = self.pods(svc) == 0
        with self.lock:
            d = self.book.request(tool, spec.get("episodeId") or name, float(spec.get("probability") or 0),
                                  int(spec.get("desiredReplicas") or 1), float(spec.get("ttlSeconds") or 0),
                                  time.time(), free, lookahead=bool(spec.get("lookahead")))
            lease = self.book.leases.get(tool)
            held = self.book.held_pods()
            start = d.action != "reject" and (d.extra > 0 or cold) and svc not in self.in_flight
            if start:
                self.in_flight.add(svc)
        HELD_PODS.set(held)
        LEASES.labels(d.action).inc()
        for t in d.evict:
            self.end_lease(t, "Evicted", f"evicted to make room for {tool}: lower value", "evicted")
        for t in d.ended:
            self.end_lease(t, "Expired", "superseded: the episode's next hint no longer predicts it", "superseded")
        if d.action == "reject":
            self.set_status(name, phase="Rejected", lastRequestedAt=requested, grantedReplicas=0, decision=d.reason,
                            message="no room for this prewarm")
            self.log(event="rejected", tool=tool, service=svc, probability=spec.get("probability"),
                     free_pods=free, reason=d.reason, episode=spec.get("episodeId"))
            return
        self.set_status(name, grantedReplicas=d.k, leaseExpiresAt=iso(lease.expires), decision=f"{d.action}: {d.reason}")
        self.log(event="leased", tool=tool, service=svc, action=d.action, granted=d.k, extra=d.extra,
                 asked=spec.get("desiredReplicas"), ttl_s=spec.get("ttlSeconds"), free_pods=free, held_pods=held,
                 evicted=d.evict, superseded=d.ended, probability=spec.get("probability"), episode=spec.get("episodeId"))
        if start:
            self.pool.submit(self.warm, obj, d.k)
        else:                                       # warm already, or being warmed: note that this one was seen
            self.set_status(name, lastRequestedAt=requested)

    def end_lease(self, tool: str, phase: str, why: str, kind: str):
        name, svc = self.objects.get(tool, (None, None))
        LEASES.labels(kind).inc()
        if name:
            self.set_status(name, phase=phase, evictedAt=now_iso(), decision=why)
        self.log(event=kind, tool=tool, service=svc, reason=why)

    def keepalive(self):
        """Lease mode: end expired leases, ping every live one so Knative's short window never runs out on it."""
        while True:
            time.sleep(KEEPALIVE_S)
            try:
                free = self.free_pods()
                self.adopt_new_pods(free)
                with self.lock:
                    ended = self.book.expire(time.time(), free)    # idle warmth while there is room, shed when not
                    live = [(t, self.objects.get(t, (None, None))[1]) for t in self.book.active()]
                    live = [(t, s) for t, s in live if s and s not in self.in_flight]
                    HELD_PODS.set(self.book.held_pods())
                for t in ended:
                    self.end_lease(t, "Expired", "TTL passed", "expired")
                for t, svc in live:
                    self.pool.submit(self._keep, svc, t)
            except Exception as e:      # keep the leases alive even if one round fails
                print(f"keepalive error: {type(e).__name__}: {e}")

    def adopt_new_pods(self, free: int):
        """A tool pod no lease asked for appeared: a real call made Knative start it. Give the tool idle warmth if
        there is room (what the 60 s window did), so the next episode's call finds it warm."""
        items = self.core.list_namespaced_pod(NS, label_selector="serving.knative.dev/service").items
        seen = self.__dict__.setdefault("_seen_pods", set())
        for p in items:
            name, svc = p.metadata.name, p.metadata.labels.get("serving.knative.dev/service", "")
            if name in seen or p.metadata.deletion_timestamp is not None:
                continue
            seen.add(name)
            tool = svc.removeprefix("retail-").replace("-", "_")
            with self.lock:
                if tool in self.book.leases or not self.book.adopt(tool, time.time(), free):
                    continue
                self.objects.setdefault(tool, (None, svc))      # no PrewarmRequest of its own to update
            LEASES.labels("adopted").inc()
            self.log(event="adopted", tool=tool, service=svc, pod=name, free_pods=free)
        live = {p.metadata.name for p in items}
        seen &= live                                    # forget pods that are gone

    def _keep(self, svc: str, tool: str):
        try:
            self.ping(svc, tool)
        except Exception as e:
            self.log(event="keepalive_failed", tool=tool, service=svc, error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ loops
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
                  f"TTL {LEASE_CFG.ttl_min_s:g}-{LEASE_CFG.ttl_max_s:g} s, keep-alive every {KEEPALIVE_S:g} s, "
                  f"tool nodes {TOOL_NODES}")
        if METRICS_PORT:
            try:
                start_http_server(METRICS_PORT, addr=bind_host())     # this machine only on a laptop (pulse/net.py)
                print(f"metrics: http://{bind_host()}:{METRICS_PORT}/metrics")
            except OSError as e:     # e.g. a second controller on the same laptop: run without metrics
                print(f"metrics port {METRICS_PORT} not available ({e}); no /metrics")
        # lease mode ends leases itself; Option A only records when Knative scaled a tool back to zero
        threading.Thread(target=self.keepalive if MODE == "lease" else self.sweep, daemon=True).start()
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
