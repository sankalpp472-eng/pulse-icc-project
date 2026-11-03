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
ROOT_RESULTS = Path(__file__).resolve().parents[2] / "results"     # new runs (from 25 Oct 2026)
LEGACY_RESULTS = Path(__file__).resolve().parent / "results"       # runs up to 25 Oct 2026, left in place


def results_root() -> Path:
    """results/local_runs on a kind cluster, results/cloud_runs otherwise (EKS); PULSE_RESULTS_DIR overrides."""
    if os.getenv("PULSE_RESULTS_DIR"):
        return Path(os.environ["PULSE_RESULTS_DIR"]).resolve()
    ctx = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True).stdout.strip()
    return ROOT_RESULTS / ("local_runs" if ctx.startswith("kind-") else "cloud_runs")


RESULTS = results_root()


def run_dir(config: str, mode: str, detail: str = "", label: str = "") -> Path:
    """A new result folder named so a run can be told apart at a glance, e.g.
    results/local_runs/2026-10-22_1336_C3_W1-replay_task9-x3   (date_time_config_workload-mode_detail[_label])."""
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
        self.forgot_at = 0.0

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
                    if now < self.forgot_at:
                        break
                    rec = self.pods.setdefault(name, {
                        "pod": name, "tool": p["metadata"]["labels"].get("pulse/tool"),
                        "created": datetime.strptime(p["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ")
                        .replace(tzinfo=timezone.utc).timestamp(),
                        "first_seen": now})
                    rec["last_seen"] = now
                    if p["metadata"].get("deletionTimestamp") and "deleting_seen" not in rec:
                        rec["deleting_seen"] = now
            time.sleep(self.interval)

    def new_pod_during(self, tool: str, t0: float, t1: float, slack: float = 1.5) -> str | None:
        """A pod for `tool` that appeared between t0 and t1 (slack for polling / 1 s timestamp rounding)."""
        with self.lock:
            for rec in self.pods.values():
                if rec["tool"] == tool and (t0 - slack <= rec["first_seen"] <= t1 + slack
                                            or t0 - slack <= rec["created"] <= t1 + 0.5):
                    return rec["pod"]
        return None

    def had_pod_before(self, tool: str, t: float) -> bool:
        """Did `tool` already have a pod (not shutting down) at time t? Then a pod that appears during a
        call may be extra capacity for parallel calls, and the call itself may have been served warm."""
        with self.lock:
            return any(rec["tool"] == tool and rec["first_seen"] < t - self.interval and rec["last_seen"] >= t
                       and rec.get("deleting_seen", float("inf")) > t for rec in self.pods.values())

    def forget(self) -> None:
        """Drop the pods seen so far, e.g. an old revision's pods still shutting down before the run starts."""
        with self.lock:
            self.pods.clear()
            self.forgot_at = time.time()      # a poll that started before this is dropped

    def running(self) -> int:
        return len(self.snapshot())

    def pod_rows(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.pods.values()]


def peak_pods(pods: list[dict]) -> dict:
    """Most pods each tool had at once (from PodWatcher.pod_rows): > 1 = the autoscaler scaled out."""
    out = {}
    for tool in {p["tool"] for p in pods}:
        spans = [(p["first_seen"], p.get("deleting_seen", p["last_seen"])) for p in pods if p["tool"] == tool]
        out[tool] = max(sum(s <= t < e or s == t for s, e in spans) for t, _ in spans)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _kjson(*args) -> dict:
    out = subprocess.run(["kubectl", *args, "-o", "json"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else {}


def _cmd(*args) -> str:
    try:
        out = subprocess.run(list(args), capture_output=True, text=True)
    except FileNotFoundError:          # e.g. no git in the runner image
        return ""
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
                      "container_concurrency": tpl["spec"].get("containerConcurrency", 0),
                      "max_scale": ann.get("autoscaling.knative.dev/max-scale", "0"),
                      "target_utilization": ann.get("autoscaling.knative.dev/target-utilization-percentage", "70"),
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
            # requests per pod (0 = no limit) and most pods per tool (0 = no limit): with concurrency 1,
            # N parallel calls to one tool make the autoscaler start N pods
            "tool_container_concurrency": uniq("container_concurrency"), "tool_max_scale": uniq("max_scale"),
            "tool_target_utilization": uniq("target_utilization"),
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
        # in the runner image there is no git: EKS_deployment/cloud/build_image.sh bakes the commit in
        "code": {"git_commit": _cmd("git", "rev-parse", "--short", "HEAD") or os.getenv("PULSE_GIT_COMMIT", ""),
                 "uncommitted_changes": bool(_cmd("git", "status", "--porcelain"))
                 if _cmd("git", "rev-parse", "HEAD") else os.getenv("PULSE_GIT_DIRTY") == "1",
                 "image": os.getenv("PULSE_IMAGE")},
        "runner": os.getenv("PULSE_RUNNER", "laptop"),     # laptop, or a pod in the cluster (cloud runs)
        **(extra or {}),
    }


def save_predictor_health(predictor_url: str | None, out_dir: Path) -> None:
    """C3: the predictor's /healthz at the end of the run (model, theta, CRD writes, the LLM speculator's
    call counts) -> predictor_health.json in the run folder."""
    if not predictor_url:
        return
    import urllib.request
    try:
        with urllib.request.urlopen(predictor_url.rstrip("/") + "/healthz", timeout=10) as r:
            (out_dir / "predictor_health.json").write_text(json.dumps(json.loads(r.read()), indent=2))
    except Exception as e:     # never fail a finished run over this
        print(f"(could not save the predictor's health: {type(e).__name__}: {e})")


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
