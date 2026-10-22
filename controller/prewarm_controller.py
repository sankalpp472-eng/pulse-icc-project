"""
Prewarm controller, Option A: "warm-up request + Knative's idle window as the TTL".

Watches PrewarmRequest objects (deploy/prewarm/prewarmrequest-crd.yaml), written by the predictor
(predictor/crd_writer.py with CRD_MODE=k8s). For every NEW or REFRESHED request (spec.requestedAt
newer than the last one handled) it sends the tool's Knative service one MCP `ping` through the
ingress -- the same path the agent's tool calls take:

    tool at zero  -> Knative's activator starts a pod (a cold start, but before the agent needs it)
    tool warm     -> the ping just counts as activity, so the idle window starts again

It never evicts anything: Knative scales the tool back to zero after its idle window
(autoscaling.knative.dev/window) without traffic, so that window IS the TTL. A sweeper marks
objects Evicted once their tool is back at zero pods, for the record (NFR-8 / E7).

Run from the project root, with the agent's Python (needs `kubernetes`), next to the predictor:

    agent/.venv/Scripts/python controller/prewarm_controller.py

Settings (environment):
  KUBE_CONTEXT       cluster to watch      (default: current kubectl context)
  CRD_NAMESPACE      namespace             (default: default)
  MCP_URL_TEMPLATE   ingress URL for a tool (default http://127.0.0.1:8081/mcp; EKS: set by EKS_deployment/use_eks.sh)
  MCP_HOST_TEMPLATE  Host header           (default {svc}.default.example.com)
  MAX_REQUEST_AGE_S  ignore requests older than this, e.g. left over from an earlier run (default 30)
  CONTROLLER_LOG     one JSON line per action (default controller/results/controller.jsonl)
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

GROUP, VERSION, PLURAL = "pulse.iisc.ac.in", "v1alpha1", "prewarmrequests"
HERE = Path(__file__).resolve().parent
NS = os.getenv("CRD_NAMESPACE", "default")
URL_TPL = os.getenv("MCP_URL_TEMPLATE", "http://127.0.0.1:8081/mcp")
HOST_TPL = os.getenv("MCP_HOST_TEMPLATE", "{svc}.default.example.com")
MAX_AGE_S = float(os.getenv("MAX_REQUEST_AGE_S", "30"))
LOG = Path(os.getenv("CONTROLLER_LOG", HERE / "results" / "controller.jsonl"))
PING = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Controller:
    def __init__(self):
        config.load_kube_config(context=os.getenv("KUBE_CONTEXT") or None)
        self.crd = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.in_flight: set[str] = set()
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
    def warm(self, obj: dict):
        name, spec = obj["metadata"]["name"], obj["spec"]
        svc, requested = spec["service"], spec["requestedAt"]
        try:
            cold = self.pods(svc) == 0
            ttl = self.window_s(svc)
            sent = now_iso()
            self.set_status(name, phase="Warming", lastRequestedAt=requested, lastPingAt=sent,
                            wasCold=cold, effectiveTtlSeconds=ttl, message="warm-up ping sent")
            req = urllib.request.Request(URL_TPL.format(svc=svc, tool=spec["tool"]), data=PING, method="POST",
                                         headers={"Host": HOST_TPL.format(svc=svc, tool=spec["tool"]),
                                                  "Content-Type": "application/json",
                                                  "Accept": "application/json, text/event-stream"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code          # any answer FROM THE POD (even 4xx) proves it is up ...
            ms = int((time.perf_counter() - t0) * 1000)
            if code in (404, 502, 503, 504):   # ... but these come from the ingress/activator: not warm
                raise RuntimeError(f"ingress answered HTTP {code}")
            ready = now_iso()
            expires = datetime.fromtimestamp(parse_iso(ready) + (ttl or 0), timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z") if ttl else None
            self.set_status(name, phase="Ready", readyAt=ready, pingLatencyMs=ms, expiresAt=expires,
                            evictedAt=None, message=f"ping answered (HTTP {code})")
            self.log(event="warmed", tool=spec["tool"], service=svc, was_cold=cold, ping_ms=ms, http=code,
                     probability=spec.get("probability"), episode=spec.get("episodeId"), requested_at=requested,
                     effective_ttl_s=ttl)
        except Exception as e:
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
                        self.log(event="evicted", tool=obj["spec"]["tool"], service=obj["spec"]["service"],
                                 warm_for_s=round(parse_iso(at) - ready, 1) if ready else None,
                                 effective_ttl_s=st.get("effectiveTtlSeconds"))
            except Exception as e:   # keep sweeping even if one round fails
                print(f"sweep error: {type(e).__name__}: {e}")

    def run(self):
        print(f"prewarm controller (Option A) | context {config.list_kube_config_contexts()[1]['name']} | "
              f"namespace {NS} | ingress {URL_TPL} | log {LOG}")
        threading.Thread(target=self.sweep, daemon=True).start()
        while True:      # the watch times out every few minutes; just start it again
            try:
                for ev in watch.Watch().stream(self.crd.list_namespaced_custom_object, GROUP, VERSION, NS,
                                               PLURAL, timeout_seconds=300):
                    if ev["type"] in ("ADDED", "MODIFIED"):
                        self.handle(ev["object"])
            except ApiException as e:
                if e.status == 404:
                    raise SystemExit("PrewarmRequest CRD not installed: kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml")
                print(f"watch error {e.status}; retrying in 2 s")
                time.sleep(2)
            except Exception as e:
                print(f"watch error: {type(e).__name__}: {e}; retrying in 2 s")
                time.sleep(2)


if __name__ == "__main__":
    Controller().run()
