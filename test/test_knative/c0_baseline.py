"""
C0 baseline on Knative: scale-to-zero, no prewarming.

Each trial:
  1. stay idle until Knative ITSELF scales the service to zero (no pods left)
  2. call it through Kourier                 -> cold call (activator holds it, pod starts)
  3. call it again a few times               -> warm calls

Unlike the OpenFaaS CE test, nothing is forced: dormancy is Knative's normal
idle behaviour, which is exactly what C0 should be.

Records per trial: cold/warm latency, HTTP status, whether a fresh pod answered
(first_request), how long the idle service took to reach zero, and the pod
timeline (scheduled / container started / ready, 1 s resolution).

Stdlib only:  python c0_baseline.py --trials 10
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import urlparse

NS = "default"


def kubectl(*args) -> str:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, check=True).stdout


def pods(svc: str) -> list[dict]:
    return json.loads(kubectl("get", "pods", "-n", NS, "-l", f"serving.knative.dev/service={svc}",
                              "-o", "json"))["items"]


def wait_for_zero(svc: str, timeout: float = 300) -> float:
    """Block until the service has no pods at all. Returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not pods(svc):
            return time.time() - t0
        time.sleep(1)
    raise RuntimeError(f"{svc}: still has pods after {timeout}s idle -- not scaling to zero?")


def invoke(ingress: str, host: str, timeout: float = 60) -> dict:
    req = urllib.request.Request(ingress, data=b"{}", method="POST", headers={"Host": host})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    except Exception as e:  # timeout, connection reset ...
        body, status = str(e).encode(), 0
    ms = (time.perf_counter() - t0) * 1000
    try:
        info = json.loads(body)
    except ValueError:
        info = {"raw": body.decode(errors="replace")[:200]}
    return {"ms": round(ms, 1), "status": status, **info}


def _ts(s: str | None) -> float | None:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if s else None


def pod_breakdown(svc: str, pod_name: str, t_request: float) -> dict:
    """Seconds after the request was sent at which each pod milestone happened."""
    for p in pods(svc):
        if p["metadata"]["name"] != pod_name:
            continue
        # Only conditions that are already True; Knative's queue-proxy often lets the
        # request through before Kubernetes marks the pod Ready.
        cond = {c["type"]: _ts(c.get("lastTransitionTime"))
                for c in p["status"].get("conditions", []) if c.get("status") == "True"}
        user = next((c for c in p["status"].get("containerStatuses", []) if c["name"] == "user-container"), {})
        started = _ts(user.get("state", {}).get("running", {}).get("startedAt"))
        rel = lambda t: None if t is None else round(t - t_request, 1)
        return {"pod_created_s": rel(_ts(p["metadata"]["creationTimestamp"])),
                "scheduled_s": rel(cond.get("PodScheduled")),
                "container_started_s": rel(started),
                "ready_s": rel(cond.get("Ready")),
                "node": p["spec"].get("nodeName")}
    return {}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingress", default="http://127.0.0.1:8081")
    ap.add_argument("--svc", default="cold-tool")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warm-calls", type=int, default=5)
    a = ap.parse_args()

    host = urlparse(kubectl("get", "ksvc", a.svc, "-n", NS, "-o", "jsonpath={.status.url}")).netloc
    first = invoke(a.ingress, host)
    if first["status"] != 200:
        sys.exit(f"Service not reachable via {a.ingress} (Host: {host}): {first}")
    print(f"C0 baseline: {a.svc} via {a.ingress} (Host: {host}), {a.trials} trials\n")

    rows = []
    for i in range(1, a.trials + 1):
        idle_s = wait_for_zero(a.svc)
        t_req = time.time()
        cold = invoke(a.ingress, host)
        fresh = cold["status"] == 200 and cold.get("first_request") is True
        bd = pod_breakdown(a.svc, cold.get("pod", ""), t_req) if fresh else {}
        warm = [invoke(a.ingress, host) for _ in range(a.warm_calls)]
        warm_ms = [w["ms"] for w in warm if w["status"] == 200]
        rows.append({"trial": i, "config": "C0", "cold_ms": cold["ms"], "cold_status": cold["status"],
                     "fresh_pod": fresh, "time_to_zero_s": round(idle_s, 1), **bd,
                     "warm_median_ms": round(statistics.median(warm_ms), 1) if warm_ms else None,
                     "warm_failures": a.warm_calls - len(warm_ms)})
        print(f"trial {i:>2}: cold {cold['ms']:7.1f} ms (HTTP {cold['status']}, fresh pod={fresh}) "
              f"| warm median {rows[-1]['warm_median_ms']} ms | waited {idle_s:4.1f}s for zero | {bd}")
        if not fresh:
            print(f"          response: {cold}")

    ok = [r for r in rows if r["fresh_pod"]]
    warm_all = [r["warm_median_ms"] for r in rows if r["warm_median_ms"]]
    summary = {"config": "C0", "trials": len(rows), "cold_calls_ok": len(ok)}
    if ok:
        c = [r["cold_ms"] for r in ok]
        summary.update(cold_p50_ms=pct(c, 50), cold_p95_ms=pct(c, 95), cold_min_ms=min(c), cold_max_ms=max(c),
                       warm_p50_ms=pct(warm_all, 50), warm_p95_ms=pct(warm_all, 95),
                       cold_penalty_p50_ms=round(pct(c, 50) - pct(warm_all, 50), 1),
                       ready_s_median=statistics.median(r["ready_s"] for r in ok if r.get("ready_s") is not None))

    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M") + "_C0_single-function"
    with open(f"results/{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    with open(f"results/{stamp}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved results/{stamp}.csv and results/{stamp}_summary.json")


if __name__ == "__main__":
    main()
