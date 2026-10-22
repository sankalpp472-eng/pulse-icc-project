"""
Shared pieces for the C0 runners (w1_live.py, w1_replay.py): pod watcher, shared-database
reset, and the run_config.json that records exactly how a run was set up.
"""

import json
import os
import subprocess
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

COLD_LATENCY_S = 1.0     # a call slower than this counts as cold (warm calls take ~40-100 ms)
POD_POLL_S = 1.0
RESULTS = Path(__file__).resolve().parent / "results"


def run_dir(config: str, mode: str, detail: str = "", label: str = "") -> Path:
    """A new result folder named so a run can be told apart at a glance, e.g.
    results/2026-10-22_1336_C3_W1-replay_task9-x3   (date_time_config_workload-mode_detail[_label])."""
    parts = [time.strftime("%Y-%m-%d_%H%M"), config, f"W1-{mode}", detail, label]
    name = re.sub(r"[^A-Za-z0-9._+-]+", "-", "_".join(x for x in parts if x))
    path = RESULTS / name
    if path.exists():                       # two runs started in the same minute
        path = RESULTS / f"{name}_{time.strftime('%S')}s"
    path.mkdir(parents=True)
    return path


def tasks_tag(task_ids: list) -> str:
    """'task9', 'tasks4-9-10', or 'tasks23' (a count) when there are many."""
    ids = sorted(set(task_ids))
    if len(ids) == 1:
        return f"task{ids[0]}"
    return "tasks" + "-".join(map(str, ids)) if len(ids) <= 5 else f"tasks{len(ids)}"


class PodWatcher(threading.Thread):
    """Polls the tool pods every second and remembers when each was first and last seen."""

    def __init__(self, interval: float = POD_POLL_S):
        super().__init__(daemon=True)
        self.interval, self.pods, self.lock, self.stop_flag = interval, {}, threading.Lock(), False

    def snapshot(self) -> list[dict]:
        out = subprocess.run(["kubectl", "get", "pods", "-l", "pulse/role=mcp-tool", "-o", "json"],
                             capture_output=True, text=True)
        return json.loads(out.stdout)["items"] if out.returncode == 0 else []

    def run(self):
        while not self.stop_flag:
            now = time.time()
            for p in self.snapshot():
                name = p["metadata"]["name"]
                with self.lock:
                    rec = self.pods.setdefault(name, {
                        "pod": name, "tool": p["metadata"]["labels"].get("pulse/tool"),
                        "created": datetime.strptime(p["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ")
                        .replace(tzinfo=timezone.utc).timestamp(),
                        "first_seen": now})
                    rec["last_seen"] = now
            time.sleep(self.interval)

    def new_pod_during(self, tool: str, t0: float, t1: float, slack: float = 1.5) -> str | None:
        """A pod for `tool` that appeared between t0 and t1 (slack for polling / 1 s timestamp rounding)."""
        with self.lock:
            for rec in self.pods.values():
                if rec["tool"] == tool and (t0 - slack <= rec["first_seen"] <= t1 + slack
                                            or t0 - slack <= rec["created"] <= t1 + 0.5):
                    return rec["pod"]
        return None

    def running(self) -> int:
        return len(self.snapshot())

    def pod_rows(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.pods.values()]


def _kjson(*args) -> dict:
    out = subprocess.run(["kubectl", *args, "-o", "json"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else {}


def _cmd(*args) -> str:
    out = subprocess.run(list(args), capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def run_config(task_ids: list, db: str, workload: str, models: dict, extra: dict | None = None) -> dict:
    """Everything needed to identify a run later: models, Knative and cluster settings, code version."""
    tools = []
    for s in _kjson("get", "ksvc", "-l", "pulse/role=mcp-tool").get("items", []):
        tpl = s["spec"]["template"]
        c = tpl["spec"]["containers"][0]
        ann = tpl.get("metadata", {}).get("annotations", {})
        tools.append({"service": s["metadata"]["name"], "revision": s.get("status", {}).get("latestReadyRevisionName"),
                      "image": c["image"],
                      "min_scale": ann.get("autoscaling.knative.dev/min-scale"),
                      "window": ann.get("autoscaling.knative.dev/window"),
                      "redis": any(e["name"] == "REDIS_URL" for e in c.get("env", [])),
                      "requests": c.get("resources", {}).get("requests"),
                      "limits": c.get("resources", {}).get("limits")})
    autoscaler = _kjson("get", "configmap", "config-autoscaler", "-n", "knative-serving").get("data", {})
    ns = _kjson("get", "namespace", "knative-serving").get("metadata", {}).get("labels", {})
    uniq = lambda k: sorted({str(t[k]) for t in tools})
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "workload": workload, "task_ids": task_ids, "models": models,
        "config": "C0" if uniq("min_scale") == ["0"] else
                  "C1" if uniq("min_scale") == ["1"] else f"min_scale={uniq('min_scale')}",
        "knative": {
            "version": ns.get("app.kubernetes.io/version"),
            "tool_min_scale": uniq("min_scale"), "tool_window": uniq("window"),
            "autoscaler_overrides": {k: v for k, v in autoscaler.items() if not k.startswith("_")} or
                                    "none (Knative defaults: stable-window 60s, scale-to-zero-grace-period 30s, "
                                    "scale-to-zero-pod-retention-period 0s)",
            "ingress": os.environ.get("MCP_URL_TEMPLATE"), "host_template": os.environ.get("MCP_HOST_TEMPLATE"),
        },
        "tools": tools, "tool_images": uniq("image"), "database": db,
        "cluster": {"context": _cmd("kubectl", "config", "current-context"),
                    "nodes": len(_kjson("get", "nodes").get("items", [])),
                    "kubernetes": _kjson("version").get("serverVersion", {}).get("gitVersion")},
        "measurement": {"cold_if_latency_over_s": COLD_LATENCY_S, "pod_poll_interval_s": POD_POLL_S},
        "code": {"git_commit": _cmd("git", "rev-parse", "--short", "HEAD"),
                 "uncommitted_changes": bool(_cmd("git", "status", "--porcelain"))},
        **(extra or {}),
    }


def shared_db() -> bool:
    return subprocess.run(["kubectl", "get", "deploy", "redis"], capture_output=True).returncode == 0


def reset_db() -> None:
    """Restore reservations + users from the pristine seed copies inside Redis."""
    for t in ("reservations", "users"):
        subprocess.run(["kubectl", "exec", "deploy/redis", "--", "redis-cli", "COPY", f"seed:{t}", t, "REPLACE"],
                       capture_output=True, check=True)


def wait_all_ready(n_tools: int = 14, timeout: float = 600) -> None:
    """C1: wait until every tool has at least one ready pod (they never scale to zero)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        items = _kjson("get", "pods", "-l", "pulse/role=mcp-tool").get("items", [])
        ready = {p["metadata"]["labels"].get("pulse/tool") for p in items
                 if any(c.get("type") == "Ready" and c.get("status") == "True"
                        for c in p.get("status", {}).get("conditions", []))}
        if len(ready) >= n_tools:
            print(" " * 70, end="\r")
            return
        print(f"  waiting for all tools to be warm ({len(ready)}/{n_tools} ready)...", end="\r")
        time.sleep(5)
    raise SystemExit(f"Not all {n_tools} tools became ready within {timeout:.0f}s")


def wait_all_zero(watcher: PodWatcher, timeout: float = 600) -> None:
    t0 = time.time()
    while (n := watcher.running()) and time.time() - t0 < timeout:
        print(f"  waiting for all tools to scale to zero ({n} pods left)...", end="\r")
        time.sleep(5)
    print(" " * 70, end="\r")
    if watcher.running():
        raise SystemExit("Tools did not scale to zero within 10 min; is something calling them?")
