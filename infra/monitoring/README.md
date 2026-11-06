# Monitoring: Prometheus + Grafana

kube-prometheus-stack in the namespace `monitoring`, for kind and EKS. Prometheus scrapes Pulse's
components and the cluster, and Grafana shows them in three dashboards kept as code. The demo UI
embeds the dashboards in its **Monitoring** tab.

## Install

```bash
bash infra/monitoring/install.sh              # into the current kubectl context (kind or EKS); rerun to update
bash infra/monitoring/install.sh uninstall    # remove it all
```

You can also install it from the UI: switch the top bar to Knative, then click **Install** next to
"no monitoring". The log streams in the Experiments tab.

- **Needs:** `helm` and `kubectl`, with the context pointing at the cluster.
- **EKS:** run it after `eks_up.sh`. The stack goes on the client node (`values-eks.yaml`), away
  from the tools, so it never takes CPU from a cold start. node-exporter runs on every node.
- **kind:** a smaller Prometheus (`values-kind.yaml`). It also scrapes the processes the laptop runs
  (the UI's predictor, the controller and the replay runner) through `host.docker.internal`.
- **Chart version:** `CHART_VERSION=<x>` pins it; the default is the newest.

## What is scraped

| Source | Metrics | How |
|---|---|---|
| `pulse/predictor/service.py` (any of the four predictors) | `predictor_request_seconds`, `predictor_hints_total`, `predictor_prewarm_candidates_total` | PodMonitor `pulse-predictors`: the `pulse-predictor` pod (port `http`) and the predictor the UI pod starts (port `predictor`) |
| `pulse/controller/prewarm_controller.py` | `pulse_controller_warmups_total{tool,cold}`, `_pings_total`, `_ping_seconds{cold}`, `_failures_total`, `_evictions_total`, `_warm_seconds` | PodMonitor `pulse-controller` (port `metrics`, `CONTROLLER_METRICS_PORT`, default 9102) |
| `pulse/experiments/replay.py` | `pulse_tool_call_seconds{tool,temperature}`, `pulse_tool_calls_total` (cold = slower than 1 s) | PodMonitor `pulse-runners`: run pods (port `metrics`) and the UI pod (port `runner-metrics`); `--metrics-port` / `PULSE_METRICS_PORT=9105`, which the UI and `run_cloud.sh` set |
| kube-state-metrics | pods per Knative service (`kube_pod_status_ready` × `kube_pod_labels{label_serving_knative_dev_service}`) | chart; the pod label is allow-listed in `values.yaml` |
| node-exporter, cAdvisor | node CPU and memory, container CPU | chart |
| Knative queue-proxy, activator, autoscaler | Knative's own metrics | PodMonitors `knative-*`; no panels yet: check the metric names on the cluster first |

## Dashboards (`dashboards/`, uids used by the UI)

| uid | Panels |
|---|---|
| `pulse-overview` | ready tool pods by tool; warm pod-seconds over the time range (NFR-7); cold share; tool call latency p50/p95/p99; calls per minute cold vs warm; controller prewarms per minute; warm-up ping p95 (≈ cold start) |
| `pulse-predictor` | predictor latency p95 against the 50 ms line (NFR-4); hints/s; prewarm candidates per tool; controller failures and evictions |
| `pulse-cluster` | node CPU and memory (cold starts are CPU-bound); tool containers' CPU; Pulse pods' CPU |

Edit the panels in `dashboards/build.py`, run `python infra/monitoring/dashboards/build.py`, then
`install.sh` (it loads every `*.json` as a ConfigMap that Grafana's sidecar picks up).

## Grafana access

- **In the UI:** the Monitoring tab, the "Grafana ↗" link in the top bar, and a Grafana link on every
  run in Results (that run's time window). The UI proxies Grafana under `/grafana/`: on the laptop
  through a `kubectl port-forward` it keeps up itself, and in the cluster directly. So the single
  `ui_cloud.sh` tunnel is enough.
- **Direct:** `kubectl port-forward -n monitoring svc/pulse-mon-grafana 3000:80`. It's read-only
  without a login; the admin password is printed by `install.sh`.
- **No public address:** Grafana's Service is ClusterIP.

## Notes

- **Retention:** Prometheus keeps 2 days of data (1 day on kind), and it is lost when the cluster is
  torn down. A result's Grafana link works only while the cluster that ran it is up. The run folders
  keep the full record.
- **Not yet instrumented:** live runs (`w1_live.py`) and the UI's interactive episodes don't publish
  tool-call metrics yet. Their pods, prewarms and predictor are still visible.
