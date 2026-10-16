"""
Cold-start test on OpenFaaS CE (Kubernetes) -- same idea as the faasd experiment:

  1. make the function dormant: scale its Deployment to 0 and wait until no pod is left
  2. call it through the gateway  -> cold call (gateway must scale it 0 -> 1)
  3. call it again a few times     -> warm calls
  4. repeat for N trials

Answers two questions:
  * Does the CE gateway wake a function from zero at all?   (status 200 + first_request)
  * How long is a cold start vs a warm call, and where does the time go?
    (pod timestamps: scheduled / container started / ready -- 1 s resolution from Kubernetes)

Stdlib only:  python cold_start.py --trials 20
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

NS = "openfaas-fn"


def kubectl(*args) -> str:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, check=True).stdout


def pods(fn: str) -> list[dict]:
    return json.loads(kubectl("get", "pods", "-n", NS, "-l", f"faas_function={fn}", "-o", "json"))["items"]


def make_dormant(fn: str, timeout: float = 120) -> None:
    kubectl("scale", "deployment", fn, "-n", NS, "--replicas=0")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pods(fn):  # pod fully terminated, not just "Terminating"
            return
        time.sleep(0.5)
    raise RuntimeError(f"{fn}: pods still present after scaling to 0")


def invoke(gateway: str, fn: str, timeout: float = 60) -> dict:
    req = urllib.request.Request(f"{gateway}/function/{fn}", data=b"{}", method="POST")
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
    except ValueError:  # plain-text function: body is the pod hostname
        info = {"pod": body.decode(errors="replace").strip()[:200]}
    return {"ms": round(ms, 1), "status": status, **info}


def _ts(s: str | None) -> float | None:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if s else None


def pod_breakdown(fn: str, pod_name: str, t_request: float) -> dict:
    """Seconds after the request was sent at which each pod milestone happened."""
    for p in pods(fn):
        if p["metadata"]["name"] != pod_name:
            continue
        cond = {c["type"]: _ts(c.get("lastTransitionTime")) for c in p["status"].get("conditions", [])}
        cs = (p["status"].get("containerStatuses") or [{}])[0]
        started = _ts(cs.get("state", {}).get("running", {}).get("startedAt"))
        rel = lambda t: None if t is None else round(t - t_request, 1)
        return {"pod_created_s": rel(_ts(p["metadata"]["creationTimestamp"])),
                "scheduled_s": rel(cond.get("PodScheduled")),
                "container_started_s": rel(started),
                "ready_s": rel(cond.get("Ready"))}
    return {}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8080")
    ap.add_argument("--fn", default="cold-tool")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warm-calls", type=int, default=5)
    a = ap.parse_args()

    # warm-up: make sure the function exists and answers
    first = invoke(a.gateway, a.fn)
    if first["status"] != 200:
        sys.exit(f"Function not reachable: {first}")

    rows = []
    for i in range(1, a.trials + 1):
        make_dormant(a.fn)
        t_req = time.time()
        cold = invoke(a.gateway, a.fn)
        # No pod existed before this call, so a 200 means the gateway woke it from zero.
        woke = cold["status"] == 200 and cold.get("first_request", True) is True
        bd = pod_breakdown(a.fn, cold.get("pod", ""), t_req) if woke else {}
        warm = [invoke(a.gateway, a.fn) for _ in range(a.warm_calls)]
        warm_ms = [w["ms"] for w in warm if w["status"] == 200]
        rows.append({"trial": i, "cold_ms": cold["ms"], "cold_status": cold["status"],
                     "woke_from_zero": woke, **bd,
                     "warm_median_ms": round(statistics.median(warm_ms), 1) if warm_ms else None})
        print(f"trial {i:>2}: cold {cold['ms']:7.1f} ms (status {cold['status']}, woke={woke}) "
              f"| warm median {rows[-1]['warm_median_ms']} ms | {bd}")
        if not woke:
            print(f"          response: {cold}")

    ok = [r for r in rows if r["woke_from_zero"]]
    warm_all = [r["warm_median_ms"] for r in rows if r["warm_median_ms"]]
    summary = {"trials": len(rows), "woke_from_zero": len(ok)}
    if ok:
        c = [r["cold_ms"] for r in ok]
        summary.update(cold_p50_ms=pct(c, 50), cold_p95_ms=pct(c, 95), cold_max_ms=max(c),
                       warm_p50_ms=pct(warm_all, 50), warm_p95_ms=pct(warm_all, 95),
                       cold_minus_warm_p50_ms=round(pct(c, 50) - pct(warm_all, 50), 1),
                       ready_s_median=statistics.median(r["ready_s"] for r in ok if r.get("ready_s") is not None))

    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f"results/cold_start_{stamp}.csv", "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))  # union of keys, in order
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(f"results/cold_start_{stamp}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print("\nCE gateway scale-from-zero:",
          "WORKS" if len(ok) == len(rows) else f"FAILED in {len(rows) - len(ok)}/{len(rows)} trials")


if __name__ == "__main__":
    main()
