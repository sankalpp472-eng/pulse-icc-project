"""
Pulse's Grafana dashboards as code: edit the panels here, then

    python deploy/monitoring/dashboards/build.py        # writes the *.json next to this file
    bash deploy/monitoring/install.sh                    # loads them into Grafana (ConfigMaps)

Every query uses metrics whose names are fixed by this repository (predictor/service.py, pulse_pred/service.py,
controller/prewarm_controller.py, test/test_knative/pulse_metrics.py) or by kube-state-metrics / node-exporter /
cAdvisor. Knative's own metrics are scraped too, but have no panels yet (check their names on the cluster first).
The UI's Monitoring tab embeds these dashboards by uid.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = {"type": "prometheus", "uid": "prometheus"}

# ready pods of the Knative tools (default namespace), by tool: kube-state-metrics joined on the pod's ksvc label
READY_TOOL_PODS = ('kube_pod_status_ready{namespace="default",condition="true"} * on (namespace, pod) '
                   'group_left (label_serving_knative_dev_service) '
                   'kube_pod_labels{namespace="default",label_serving_knative_dev_service!=""}')


def panel(title, exprs, x, y, w=12, h=8, kind="timeseries", unit=None, desc="", thresholds=None, stack=False):
    defaults = {"unit": unit} if unit else {}
    if thresholds is not None:                     # a red line at the target (e.g. NFR-4: 50 ms)
        defaults["thresholds"] = {"mode": "absolute",
                                  "steps": [{"color": "green", "value": None}, {"color": "red", "value": thresholds}]}
        defaults["custom"] = {"thresholdsStyle": {"mode": "line"}}
    if stack:
        defaults.setdefault("custom", {})["stacking"] = {"mode": "normal"}
    return {"type": kind, "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"datasource": DS, "expr": e, "legendFormat": legend, "refId": chr(65 + i)}
                        for i, (e, legend) in enumerate(exprs)],
            "fieldConfig": {"defaults": defaults, "overrides": []}}


def dashboard(uid, title, panels):
    return {"uid": uid, "title": title, "tags": ["pulse"], "schemaVersion": 39, "editable": True,
            "time": {"from": "now-15m", "to": "now"}, "refresh": "10s", "timezone": "browser",
            "panels": [dict(p, id=i + 1) for i, p in enumerate(panels)]}


OVERVIEW = dashboard("pulse-overview", "Pulse overview", [
    panel("Ready tool pods, by tool", [(f"sum by (label_serving_knative_dev_service) ({READY_TOOL_PODS})",
                                        "{{label_serving_knative_dev_service}}")], 0, 0, stack=True,
          desc="Knative tool pods that are ready. Their area over time is the warm pod-seconds (NFR-7)."),
    panel("Warm tool pod-seconds (time range)",
          [(f"sum_over_time((sum({READY_TOOL_PODS}))[$__range:15s]) * 15", "pod-seconds")], 12, 0, w=6, kind="stat",
          unit="s", desc="Ready tool pods integrated over the dashboard's time range (15 s resolution)."),
    panel("Cold share of tool calls (time range)",
          [('sum(increase(pulse_tool_calls_total{temperature="cold"}[$__range])) / '
            'sum(increase(pulse_tool_calls_total[$__range]))', "cold")], 18, 0, w=6, kind="stat", unit="percentunit",
          desc="Runner calls slower than 1 s (live rule; the run's calls.csv also uses the pod rule)."),
    panel("Tool call latency (runner)", [
        ('histogram_quantile(0.5, sum by (le) (rate(pulse_tool_call_seconds_bucket[1m])))', "p50"),
        ('histogram_quantile(0.95, sum by (le) (rate(pulse_tool_call_seconds_bucket[1m])))', "p95"),
        ('histogram_quantile(0.99, sum by (le) (rate(pulse_tool_call_seconds_bucket[1m])))', "p99")],
        0, 8, unit="s", desc="As the agent sees it: from the replay runner (w1_replay.py --metrics-port)."),
    panel("Tool calls per minute, cold vs warm",
          [("sum by (temperature) (rate(pulse_tool_calls_total[1m])) * 60", "{{temperature}}")], 12, 8, stack=True),
    panel("Prewarms per minute (controller)",
          [('sum by (cold) (rate(pulse_controller_warmups_total[1m])) * 60', "tool was cold: {{cold}}")], 0, 16,
          stack=True, desc="Warm-ups the prewarm controller did. cold=true: its ping started the pod."),
    panel("Warm-up ping latency p95 (controller)",
          [('histogram_quantile(0.95, sum by (le, cold) (rate(pulse_controller_ping_seconds_bucket[5m])))',
            "cold: {{cold}}")], 12, 16, unit="s", desc="With cold=true this is about the tool's cold start."),
])

PREDICTOR = dashboard("pulse-predictor", "Pulse predictor", [
    panel("Predictor latency p95 (target 50 ms, NFR-4)",
          [('histogram_quantile(0.95, sum by (le, component) (rate(predictor_request_seconds_bucket[1m])))',
            "{{component}}")], 0, 0, unit="s", thresholds=0.05),
    panel("Hints per second", [("sum by (component) (rate(predictor_hints_total[1m]))", "{{component}}")], 12, 0,
          unit="reqps"),
    panel("Prewarm candidates per minute, by tool",
          [("sum by (tool) (rate(predictor_prewarm_candidates_total[1m])) * 60", "{{tool}}")], 0, 8, stack=True),
    panel("Controller: warm-up failures and evictions (5 min)", [
        ("sum by (tool) (increase(pulse_controller_failures_total[5m]))", "failed {{tool}}"),
        ("sum(increase(pulse_controller_evictions_total[5m]))", "evicted (all tools)")], 12, 8),
])

CLUSTER = dashboard("pulse-cluster", "Pulse cluster", [
    panel("Node CPU used (cores)",
          [('sum by (nodename) (rate(node_cpu_seconds_total{mode!="idle"}[1m]) * on (instance) group_left (nodename) '
            'node_uname_info)', "{{nodename}}")], 0, 0,
          desc="Cold starts are CPU-bound: a busy tool node makes them slower."),
    panel("Node memory used",
          [('sum by (nodename) ((node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) * on (instance) '
            'group_left (nodename) node_uname_info)', "{{nodename}}")], 12, 0, unit="bytes"),
    panel("Tool containers CPU (cores)",
          [('sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default",container="user-container"}[1m]))',
            "{{pod}}")], 0, 8, desc="The tools' user containers (Knative), from cAdvisor."),
    panel("Pulse pods CPU (cores)",
          [('sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default",pod=~"pulse-.*",container!=""}[1m]))',
            "{{pod}}")], 12, 8, desc="Predictors, controller, UI and run pods."),
])

if __name__ == "__main__":
    for name, d in (("overview", OVERVIEW), ("predictor", PREDICTOR), ("cluster", CLUSTER)):
        (HERE / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {name}.json ({len(d['panels'])} panels)")
