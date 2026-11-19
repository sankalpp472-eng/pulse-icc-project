"""
Prewarm controller: "warm-up request + keep-alive for the TTL, then Knative's idle window scales it down".

Watches PrewarmRequest objects (infra/prewarm/prewarmrequest-crd.yaml), written by the predictor
(pulse/predictor/crd_writer.py with CRD_MODE=k8s). For every NEW or REFRESHED request (spec.requestedAt
newer than the last one handled) it sends the tool's Knative service MCP `ping`s through the
ingress -- the same path the agent's tool calls take:

    tool at zero  -> Knative's activator starts a pod (a cold start, but before the agent needs it)
    tool warm     -> the ping just counts as activity, so the idle window starts again

spec.desiredReplicas = k (the predictor's fan-out width): if the tool has fewer than k pods, k pings go
out at once. With containerConcurrency 1 (pulse/agent/gen_services.py) each needs its own pod, so
Knative starts k pods for the coming burst of k parallel calls. Otherwise one ping.

TTL (FR-7, issue #37): the prewarmed tool stays warm for spec.ttlSeconds after the warm-up answered (readyAt),
then is released. Knative scales a tool to zero once it has had no request for its idle window W
(autoscaling.knative.dev/window), so:

    TTL > W   the controller keeps the tool alive with keep-alive pings (one every W/2) until readyAt + TTL - W.
              The last one starts the final idle window, so the pod scales down at about readyAt + TTL.
    TTL <= W  no keep-alive is needed or possible: the pod lives for the window, about readyAt + W.
              Knative does not scale a tool down earlier than its window (a deleted pod is started again),
              so W is a floor. To sweep TTLs below the default 60 s (E7: 30/60/120 s), deploy the tools
              with a window at or below the smallest TTL (WINDOW=20s bash infra/knative/deploy.sh 0).

So the effective TTL is max(TTL, W), shown as status.effectiveTtlSeconds, next to status.requestedTtlSeconds
and status.windowSeconds. A new prediction for the tool (new requestedAt) starts the TTL again. Real tool calls
are not affected: they keep the tool warm as usual, the TTL only bounds how long a PREDICTION keeps it warm.
The keep-alive holds one pod; extra fan-out pods (desiredReplicas k > 1) follow Knative's own scale-down.

The controller never deletes pods. A sweeper marks objects Evicted once their tool is back at zero pods, with
how long it stayed warm (warm_for_s), for NFR-8 / E7.

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
  TTL_KEEPALIVE      1 = honour spec.ttlSeconds with keep-alive pings (default); 0 = the window alone (old behaviour)
  CONTROLLER_LOG     one JSON line per action (default results/controller.jsonl)
  CONTROLLER_METRICS_PORT  Prometheus /metrics on this port (default 9102; 0 = off), see infra/monitoring/
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
from prometheus_client import Counter, Histogram, start_http_server

GROUP, VERSION, PLURAL = "pulse.iisc.ac.in", "v1alpha1", "prewarmrequests"
from pulse import paths  # noqa: E402
NS = os.getenv("CRD_NAMESPACE", "default")
URL_TPL = os.getenv("MCP_URL_TEMPLATE", "http://127.0.0.1:31080/mcp")
HOST_TPL = os.getenv("MCP_HOST_TEMPLATE", "{svc}.default.example.com")
MAX_AGE_S = float(os.getenv("MAX_REQUEST_AGE_S", "30"))
LOG = Path(os.getenv("CONTROLLER_LOG") or paths.RESULTS / "controller.jsonl")
PING = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
METRICS_PORT = int(os.getenv("CONTROLLER_METRICS_PORT", "9102"))
TTL_KEEPALIVE = os.getenv("TTL_KEEPALIVE", "1") != "0"

# ---- Prometheus metrics (FR-11): what the controller did, as counters the dashboards can rate()
WARMUPS = Counter("pulse_controller_warmups_total", "Warm-ups done (PrewarmRequests acted on)", ["tool", "cold"])
PINGS = Counter("pulse_controller_pings_total", "Warm-up MCP pings sent", ["tool"])
PING_SECONDS = Histogram("pulse_controller_ping_seconds", "Warm-up ping round trip (about a cold start if cold)",
                         ["cold"], buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 60))
FAILURES = Counter("pulse_controller_failures_total", "Warm-ups that failed", ["tool"])
EVICTIONS = Counter("pulse_controller_evictions_total", "Tools seen back at zero pods after a warm-up", ["tool"])
KEEPALIVES = Counter("pulse_controller_keepalive_pings_total", "Keep-alive pings sent to hold a tool for its TTL",
                     ["tool"])
WARM_SECONDS = Histogram("pulse_controller_warm_seconds", "Time from a warm-up's readyAt to the tool's scale-down",
                         buckets=(30, 60, 75, 90, 120, 180, 300, 600))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def to_iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def effective_ttl(ttl_s: int | None, window_s: int | None, keepalive: bool = True) -> int | None:
    """How long a prewarmed tool stays warm after its warm-up: max(TTL, window) with keep-alive, else the window."""
    if not keepalive or not ttl_s:
        return window_s
    return max(int(ttl_s), window_s or 0)


def keepalive_due(hold: dict, now: float) -> tuple[bool, bool]:
    """(ping now?, is it the last one?) for a tool held until hold["until"] (= readyAt + TTL).

    The last ping goes out at until - window, so the window that follows ends at about `until`; before that,
    one ping every window/2 keeps Knative from seeing a whole window without traffic."""
    last_at = hold["until"] - hold["window"]
    due = min(hold["last_ping"] + hold["window"] / 2, last_at)
    if now < due:
        return False, False
    return True, due >= last_at


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
        self.holds: dict[str, dict] = {}       # svc -> tool held warm until its TTL ends (keep-alive pings)
        self.lock = threading.Lock()
        LOG.parent.mkdir(parents=True, exist_ok=True)

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

    def warm(self, obj: dict):
        name, spec = obj["metadata"]["name"], obj["spec"]
        svc, requested = spec["service"], spec["requestedAt"]
        try:
            before = self.pods(svc)
            cold = before == 0
            # desiredReplicas = the predictor's fan-out width k. With containerConcurrency 1 every in-flight
            # request needs its own pod, so k pings at once make Knative start k pods. A tool that already
            # has k pods gets one ping, which just restarts its idle window.
            k = max(1, int(spec.get("desiredReplicas") or 1))
            n = k if before < k else 1
            window = self.window_s(svc)
            want = int(spec.get("ttlSeconds") or 0) or None
            ttl = effective_ttl(want, window, TTL_KEEPALIVE)
            sent = now_iso()
            self.set_status(name, phase="Warming", lastRequestedAt=requested, lastPingAt=sent, wasCold=cold,
                            requestedTtlSeconds=want, windowSeconds=window, effectiveTtlSeconds=ttl,
                            message=f"{n} warm-up ping(s) sent")
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(lambda _: self.ping(svc, spec["tool"]), range(n)))
            codes = [c for c, _ in results]
            ms = max(m for _, m in results)
            bad = [c for c in codes if c in (404, 502, 503, 504)]   # ... but these come from the ingress/activator
            if bad:
                raise RuntimeError(f"ingress answered HTTP {bad[0]} to {len(bad)} of {n} pings")
            ready_t = time.time()
            ready = to_iso(ready_t)
            expires = to_iso(ready_t + ttl) if ttl else None
            held = bool(window) and ttl is not None and ttl > window      # keep-alive pings needed
            with self.lock:                     # a refreshed prediction starts the TTL again
                if held:
                    self.holds[svc] = {"name": name, "tool": spec["tool"], "until": ready_t + ttl,
                                       "window": window, "last_ping": ready_t, "pings": 0}
                else:
                    self.holds.pop(svc, None)
            note = ""
            if want and window and want < window:
                note = f"; TTL {want} s is below the {window} s idle window, so the pod stays {window} s"
            elif held:
                note = f"; held for TTL {ttl} s with keep-alive pings (window {window} s)"
            self.set_status(name, phase="Ready", readyAt=ready, pingLatencyMs=ms, expiresAt=expires,
                            evictedAt=None, message=f"{n} ping(s) answered (HTTP {codes[0]}){note}")
            WARMUPS.labels(spec["tool"], str(cold).lower()).inc()
            PINGS.labels(spec["tool"]).inc(n)
            PING_SECONDS.labels(str(cold).lower()).observe(ms / 1000)
            self.log(event="warmed", tool=spec["tool"], service=svc, was_cold=cold, ping_ms=ms, http=codes[0],
                     desired_replicas=k, pings=n, pods_before=before, pods_after=self.pods(svc),
                     probability=spec.get("probability"), episode=spec.get("episodeId"), requested_at=requested,
                     effective_ttl_s=ttl, requested_ttl_s=want, window_s=window, held=held)
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
        svc = spec["service"]
        with self.lock:
            if svc in self.in_flight:               # a ping is already warming this tool
                return
            self.in_flight.add(svc)
        self.pool.submit(self.warm, obj)

    # ------------------------------------------------------------------ loops
    def keepalive(self):
        """Hold prewarmed tools for their TTL: a ping every window/2, the last at TTL end - window (FR-7)."""
        while True:
            time.sleep(1)
            now = time.time()
            with self.lock:
                due = [(svc, h, last) for svc, h in self.holds.items()
                       for ping, last in [keepalive_due(h, now)] if ping and svc not in self.in_flight]
                for svc, h, last in due:
                    h["last_ping"] = now
                    h["pings"] += 1
                    if last:
                        del self.holds[svc]
            for svc, h, last in due:
                self.pool.submit(self._keepalive_ping, svc, h, last)

    def _keepalive_ping(self, svc: str, h: dict, last: bool):
        try:
            code, ms = self.ping(svc, h["tool"])
            KEEPALIVES.labels(h["tool"]).inc()
            if last:
                self.set_status(h["name"], message=f"TTL over: released after {h['pings']} keep-alive ping(s); "
                                                   f"Knative scales the tool down after its {h['window']} s window")
            self.log(event="keepalive", tool=h["tool"], service=svc, http=code, ping_ms=ms, last=last,
                     ttl_ends_at=to_iso(h["until"]))
        except Exception as e:
            self.log(event="keepalive_failed", tool=h["tool"], service=svc, error=f"{type(e).__name__}: {e}")

    def sweep(self):
        """Mark Ready objects Evicted once Knative has scaled their tool back to zero."""
        while True:
            time.sleep(5)
            try:
                for obj in self.crd.list_namespaced_custom_object(GROUP, VERSION, NS, PLURAL)["items"]:
                    st = obj.get("status") or {}
                    if st.get("phase") == "Ready" and obj["spec"]["service"] not in self.holds \
                            and self.pods(obj["spec"]["service"]) == 0:
                        at = now_iso()
                        self.set_status(obj["metadata"]["name"], phase="Evicted", evictedAt=at,
                                        message="Knative scaled the tool to zero (idle window passed)")
                        ready = parse_iso(st.get("readyAt"))
                        EVICTIONS.labels(obj["spec"]["tool"]).inc()
                        if ready:
                            WARM_SECONDS.observe(parse_iso(at) - ready)
                        self.log(event="evicted", tool=obj["spec"]["tool"], service=obj["spec"]["service"],
                                 warm_for_s=round(parse_iso(at) - ready, 1) if ready else None,
                                 effective_ttl_s=st.get("effectiveTtlSeconds"),
                                 requested_ttl_s=st.get("requestedTtlSeconds"))
            except Exception as e:   # keep sweeping even if one round fails
                print(f"sweep error: {type(e).__name__}: {e}")

    def run(self):
        ctx = "in-cluster" if os.getenv("KUBERNETES_SERVICE_HOST") else config.list_kube_config_contexts()[1]["name"]
        from pulse.net import bind_host
        print(f"prewarm controller | TTL {'spec.ttlSeconds (keep-alive)' if TTL_KEEPALIVE else 'idle window only'} | "
              f"context {ctx} | namespace {NS} | ingress {URL_TPL} | log {LOG}")
        if METRICS_PORT:
            try:
                start_http_server(METRICS_PORT, addr=bind_host())     # this machine only on a laptop (pulse/net.py)
                print(f"metrics: http://{bind_host()}:{METRICS_PORT}/metrics")
            except OSError as e:     # e.g. a second controller on the same laptop: run without metrics
                print(f"metrics port {METRICS_PORT} not available ({e}); no /metrics")
        threading.Thread(target=self.sweep, daemon=True).start()
        threading.Thread(target=self.keepalive, daemon=True).start()
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
