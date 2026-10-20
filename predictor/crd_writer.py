"""
Step 9: turn the predictor's "prewarm" list into PrewarmRequest objects for the controller.

    agent --hint--> predictor /predict --prewarm list--> CRDWriter --> PrewarmRequest (Kubernetes)
                                                                           |
                                                     controller (Nithish) watches, warms the pod,
                                                     evicts it when the TTL runs out

One object per tool, named after its Knative service. If the tool is predicted again, the
object is UPDATED (new requestedAt, so the TTL starts again). Writes happen on a background
thread, so /predict never waits for Kubernetes.

Mode (env CRD_MODE):
  off     (default) do nothing
  dryrun  no cluster needed: write each object as one JSON line to CRD_DRYRUN_FILE
          (default predictor/results/prewarm_requests.jsonl)
  k8s     create / update the objects in the cluster
          (KUBE_CONTEXT, default kind-pulse-kn; CRD_NAMESPACE, default default)
"""

import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

GROUP, VERSION, PLURAL, KIND = "pulse.iisc.ac.in", "v1alpha1", "prewarmrequests", "PrewarmRequest"
HERE = Path(__file__).resolve().parent


def service_name(tool: str) -> str:
    """search_direct_flight -> search-direct-flight (same rule as deploy/knative/gen_services.py)"""
    return tool.replace("_", "-")


def make_object(tool, probability, threshold, ttl_s, episode_id="", workflow_id="",
                desired_replicas=1, namespace="default"):
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": KIND,
        "metadata": {"name": service_name(tool), "namespace": namespace,
                     "labels": {"pulse/role": "prewarm", "pulse/tool": tool}},
        "spec": {"tool": tool, "service": service_name(tool),
                 "probability": round(float(probability), 4), "threshold": float(threshold),
                 "ttlSeconds": int(ttl_s), "desiredReplicas": int(desired_replicas),
                 "requestedAt": now, "episodeId": episode_id or "", "workflowId": workflow_id or ""},
    }


class CRDWriter:
    def __init__(self, mode=None):
        self.mode = (mode or os.getenv("CRD_MODE", "off")).lower()
        self.namespace = os.getenv("CRD_NAMESPACE", "default")
        self.dryrun_file = Path(os.getenv("CRD_DRYRUN_FILE", HERE / "results" / "prewarm_requests.jsonl"))
        self.stats = {"written": 0, "created": 0, "updated": 0, "failed": 0}
        self.last_error = None
        self._q = queue.Queue()
        self._api = None
        if self.mode == "off":
            return
        if self.mode == "k8s":
            self._api = self._connect()
        elif self.mode == "dryrun":
            self.dryrun_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"CRD_MODE must be off, dryrun or k8s, not {self.mode!r}")
        threading.Thread(target=self._worker, daemon=True).start()

    def _connect(self):
        from kubernetes import client, config
        if os.getenv("KUBERNETES_SERVICE_HOST"):          # running inside the cluster (step 10)
            config.load_incluster_config()
        else:                                              # running on the laptop
            config.load_kube_config(context=os.getenv("KUBE_CONTEXT", "kind-pulse-kn"))
        return client.CustomObjectsApi()

    # ---- called by service.py: returns immediately
    def submit(self, prewarm, theta, ttl_s, episode_id="", workflow_id=""):
        """prewarm = [(tool, probability), ...] -- the tools at or above theta."""
        if self.mode == "off":
            return
        for tool, p in prewarm:
            self._q.put(make_object(tool, p, theta, ttl_s, episode_id, workflow_id,
                                    namespace=self.namespace))

    def flush(self, timeout=5.0):
        """Wait until everything submitted so far is written (used by tests)."""
        done = threading.Event()
        self._q.put(done)
        return done.wait(timeout)

    # ---- background thread
    def _worker(self):
        while True:
            obj = self._q.get()
            if isinstance(obj, threading.Event):
                obj.set()
                continue
            try:
                if self.mode == "dryrun":
                    with open(self.dryrun_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(obj) + "\n")
                else:
                    self._apply(obj)
                self.stats["written"] += 1
            except Exception as e:      # never crash the predictor because of Kubernetes
                self.stats["failed"] += 1
                self.last_error = f"{type(e).__name__}: {e}"

    def _apply(self, obj):
        from kubernetes.client.rest import ApiException
        name = obj["metadata"]["name"]
        try:   # update the existing object: only .spec changes, .status stays the controller's
            self._api.patch_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name,
                                                     {"metadata": {"labels": obj["metadata"]["labels"]},
                                                      "spec": obj["spec"]})
            self.stats["updated"] += 1
        except ApiException as e:
            if e.status != 404:
                raise
            self._api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, obj)
            self.stats["created"] += 1
