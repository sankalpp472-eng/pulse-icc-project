"""
Export the cluster's resource telemetry from Prometheus (infra/monitoring) to CSV files, one per metric, for the
time windows of an experiment's runs. Prometheus in the cluster has no persistent storage, so this must run
before the cluster is deleted (experiments/team/lib.sh does it after every experiment).

    kubectl -n monitoring port-forward svc/prometheus-operated 19090:9090 &
    python -m pulse.experiments.prom_export --url http://127.0.0.1:19090 --manifest results/team/<id>/manifest.tsv \
        --exp main_latency --out results/team/<id>/prometheus/main_latency

Each CSV is long format: run (the manifest row's arm/trial/domain), unix time, the series' labels (JSON) and the
value; plus queries.json (the PromQL of every file). Step 15 s.
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

NS = 'namespace="default"'
QUERIES = {
    # tool pods, predictor, controller, run pods: CPU cores and memory per pod and container
    "pod_cpu_cores": f'sum by (pod, container) (rate(container_cpu_usage_seconds_total{{{NS},container!="",container!="POD"}}[1m]))',
    "pod_memory_bytes": f'sum by (pod, container) (container_memory_working_set_bytes{{{NS},container!="",container!="POD"}})',
    # how many tool pods each tool had (Knative labels every pod with its service)
    "tool_pods": f'count by (label_serving_knative_dev_service) (kube_pod_labels{{{NS},label_serving_knative_dev_service!=""}} '
                 f'* on (pod) group_left() (kube_pod_status_phase{{{NS},phase="Running"}} == 1))',
    "pods_pending": f'sum(kube_pod_status_phase{{{NS},phase="Pending"}})',
    # nodes: CPU busy share and memory in use, per node
    "node_cpu_busy": '1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m]))',
    "node_memory_used_bytes": 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes',
    # CPU / memory requested on each node (what limits how many pods fit)
    "node_cpu_requested_cores": 'sum by (node) (kube_pod_container_resource_requests{resource="cpu"})',
    "node_memory_requested_bytes": 'sum by (node) (kube_pod_container_resource_requests{resource="memory"})',
    # the predictor's answer time and the controller's work
    "predictor_p95_s": 'histogram_quantile(0.95, sum by (le) (rate(predictor_request_seconds_bucket[1m])))',
    "predictor_hints_per_s": 'sum(rate(predictor_hints_total[1m]))',
    "controller_warmups_per_s": 'sum by (cold) (rate(pulse_controller_warmups_total[1m]))',
    "controller_skips_per_s": 'sum by (why) (rate(pulse_controller_skips_total[1m]))',
    "controller_keepalive_pings_per_s": 'sum(rate(pulse_controller_keepalive_pings_total[1m]))',
}


def query_range(url: str, q: str, start: float, end: float, step: int) -> list[dict]:
    qs = urllib.parse.urlencode({"query": q, "start": f"{start:.0f}", "end": f"{end:.0f}", "step": step})
    with urllib.request.urlopen(f"{url.rstrip('/')}/api/v1/query_range?{qs}", timeout=60) as r:
        body = json.load(r)
    if body.get("status") != "success":
        raise RuntimeError(body.get("error", body))
    return body["data"]["result"]


def windows(manifest: Path, exp: str, pad: float) -> list[tuple[str, float, float]]:
    """(run name, start, end) of the experiment's finished runs: columns exp domain arm trial status dir start end."""
    out = []
    for r in csv.reader(open(manifest, encoding="utf-8"), delimiter="\t"):
        if len(r) >= 8 and r[0] == exp and r[4] == "done" and r[6] and r[7]:
            out.append((f"{r[1]}/{r[2]}/t{r[3]}", float(r[6]) - pad, float(r[7]) + pad))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:19090")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--pad", type=float, default=60, help="seconds before / after each run")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = windows(Path(a.manifest), a.exp, a.pad)
    if not runs:
        print(f"no finished runs of {a.exp} in {a.manifest}")
        return 0
    failed = []
    for name, q in QUERIES.items():
        rows = []
        for run, start, end in runs:
            try:
                for series in query_range(a.url, q, start, end, a.step):
                    labels = json.dumps(series.get("metric", {}), sort_keys=True)
                    rows += [(run, int(float(t)), labels, v) for t, v in series.get("values", [])]
            except Exception as e:
                failed.append(f"{name} ({run}): {type(e).__name__}: {e}")
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "time", "labels", "value"])
            w.writerows(rows)
        print(f"  {name}: {len(rows)} samples")
    (out / "queries.json").write_text(json.dumps({"exported_at": time.time(), "step_s": a.step, "queries": QUERIES,
                                                  "runs": runs, "errors": failed}, indent=2))
    for f in failed[:10]:
        print(f"  error: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
