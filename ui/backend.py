"""
The UI's back office: which cluster it talks to, the predictor / controller processes it starts on the laptop,
batch runs (w1_replay.py / w1_live.py / run_cloud.sh) and the result folders.

Nothing here prints secrets: processes inherit agent/.env through the agent's own code, never through the UI.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "ui" / ".state"                 # logs of the processes the UI starts (git-ignored)
PY = sys.executable                             # the agent's Python (agent/.venv) runs the UI and everything it starts
PREDICTOR_PORT = int(os.getenv("UI_PREDICTOR_PORT", "8000"))
PREDICTOR_URL = f"http://127.0.0.1:{PREDICTOR_PORT}"
CONTROLLER_LOG = STATE / "controller.jsonl"
RESULT_DIRS = {"local": ROOT / "results" / "local_runs", "cloud": ROOT / "results" / "cloud_runs",
               "legacy": ROOT / "test" / "test_knative" / "results"}
TOOL_NAMES = ["book_reservation", "calculate", "cancel_reservation", "get_reservation_details", "get_user_details",
              "list_all_airports", "search_direct_flight", "search_onestop_flight", "send_certificate", "think",
              "transfer_to_human_agents", "update_reservation_baggages", "update_reservation_flights",
              "update_reservation_passengers"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _run(args: list[str], timeout: float = 10) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
        return p.returncode, p.stdout.strip() or p.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, f"{type(e).__name__}"


def bash() -> str | None:
    """Git Bash (the deploy / cloud scripts are bash). Never WSL's bash.exe in System32."""
    for c in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if Path(c).exists():
            return c
    b = shutil.which("bash")
    return b if b and "system32" not in b.lower() else None


def plot_python() -> str:
    """A Python with matplotlib for plot_compare.py (the agent's venv has none)."""
    cand = [os.getenv("PLOT_PYTHON"), r"C:\Python311\python.exe", PY]
    for c in cand:
        if c and Path(c).exists() and _run([c, "-c", "import matplotlib"], 20)[0] == 0:
            return c
    return PY


# ====================================================================== cluster
class Cluster:
    """Where the tools are: a real Knative cluster (current kubectl context: kind or EKS) or the simulated one."""

    def __init__(self):
        self.mode = "sim"
        self.context, self.reachable, self.detail = None, False, "not checked"
        self.tools_found, self.min_scale = 0, None
        self.url_tpl, self.host_tpl = None, None
        self.pods: list[dict] = []
        self.pods_at = 0.0
        self.events: list[dict] = []           # controller actions (tailed from its log)
        self._log_pos = 0
        threading.Thread(target=self._poll, daemon=True).start()

    def detect(self) -> dict:
        code, ctx = _run(["kubectl", "config", "current-context"], 5)
        self.context = ctx if code == 0 else None
        if not self.context:
            self.reachable, self.detail = False, "no kubectl context"
            return self.info()
        code, out = _run(["kubectl", "get", "ksvc", "-l", "pulse/role=mcp-tool", "-o", "json",
                          "--request-timeout=6s"], 10)
        if code != 0:
            why = "is Docker / the cluster running?" if any(s in out for s in ("refused", "Unable to connect", "Timeout",
                                                                              "no such host")) else out[-160:]
            self.reachable, self.detail = False, f"'{self.context}' is not reachable ({why})"
            return self.info()
        items = json.loads(out).get("items", [])
        self.tools_found = len(items)
        scales = {(i["spec"]["template"]["metadata"].get("annotations", {}) or {})
                  .get("autoscaling.knative.dev/min-scale", "0") for i in items}
        self.min_scale = int(scales.pop()) if len(scales) == 1 else None
        self.reachable = self.tools_found > 0
        self.detail = f"{self.tools_found} tools deployed" if self.reachable else "no Pulse tools deployed"
        if self.context.startswith("kind-"):
            self.url_tpl = "http://127.0.0.1:8081/mcp"
        else:                                   # EKS: Kourier's load balancer
            code, lb = _run(["kubectl", "get", "svc", "kourier", "-n", "kourier-system", "-o",
                             "jsonpath={.status.loadBalancer.ingress[0].hostname}"], 10)
            self.url_tpl = f"http://{lb}/mcp" if code == 0 and lb else None
        self.host_tpl = "{svc}.default.example.com"
        return self.info()

    def info(self) -> dict:
        return {"mode": self.mode, "context": self.context, "reachable": self.reachable, "detail": self.detail,
                "tools": self.tools_found, "min_scale": self.min_scale, "ingress": self.url_tpl,
                "in_cluster_services": self.in_cluster_services() if self.reachable else []}

    def in_cluster_services(self) -> list[str]:
        code, out = _run(["kubectl", "get", "deploy", "-l", "pulse/role=system", "-o",
                          "jsonpath={.items[*].metadata.name}", "--request-timeout=5s"], 8)
        return out.split() if code == 0 and out else []

    def tool_env(self) -> dict:
        """Environment for anything that calls the tools on the real cluster."""
        env = {"TOOL_BACKEND": "mcp", "KUBE_CONTEXT": self.context or ""}
        if self.url_tpl:
            env["MCP_URL_TEMPLATE"] = self.url_tpl
            env["MCP_HOST_TEMPLATE"] = self.host_tpl
        return env

    # -------------------------------------------------------------- live state (knative mode)
    def _poll(self):
        while True:
            if self.mode == "knative" and self.reachable:
                code, out = _run(["kubectl", "get", "pods", "-l", "pulse/role=mcp-tool", "-o", "json",
                                  "--request-timeout=4s"], 6)
                if code == 0:
                    counts = {t: {"tool": t, "ready": 0, "starting": 0, "busy": 0} for t in TOOL_NAMES}
                    for p in json.loads(out).get("items", []):
                        if p["metadata"].get("deletionTimestamp"):
                            continue
                        t = p["metadata"].get("labels", {}).get("pulse/tool")
                        if t not in counts:
                            continue
                        st = p.get("status", {}).get("containerStatuses") or []
                        ok = st and all(c.get("ready") for c in st)
                        counts[t]["ready" if ok else "starting"] += 1
                    self.pods, self.pods_at = list(counts.values()), time.time()
                self._tail_controller()
            time.sleep(1.0)

    def _tail_controller(self):
        if not CONTROLLER_LOG.exists():
            return
        with open(CONTROLLER_LOG, encoding="utf-8") as f:
            f.seek(self._log_pos)
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") in ("warmed", "evicted", "failed"):
                    r["source"] = "controller"
                    self.events.append(r)
            self._log_pos = f.tell()


# ====================================================================== services
class Proc:
    def __init__(self, name: str, cmd: list[str], env: dict, info: dict):
        self.name, self.cmd, self.info = name, cmd, info
        self.log = STATE / "logs" / f"{name}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.log, "w", encoding="utf-8")
        self.pidfile = STATE / f"{name}.pid"
        self._kill_leftover()
        self.p = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **env, "PYTHONUNBUFFERED": "1",
                                                      "PYTHONIOENCODING": "utf-8"},
                                  stdout=self.fh, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        self.pidfile.write_text(str(self.p.pid))
        self.started = time.time()

    def _kill_leftover(self):
        """A process of this name left running by a UI that was killed (it would still hold the port)."""
        try:
            pid = int(self.pidfile.read_text())
        except (OSError, ValueError):
            return
        if os.name == "nt":
            _run(["taskkill", "/PID", str(pid), "/T", "/F"], 10)
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        self.pidfile.unlink(missing_ok=True)

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.terminate()
            try:
                self.p.wait(8)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.fh.close()
        self.pidfile.unlink(missing_ok=True)

    def tail(self, n: int = 6) -> list[str]:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        except OSError:
            return []


class Services:
    """The predictor (or the LLM speculator) and the prewarm controller, run on the laptop."""

    MODELS = {"gbt": "LightGBM next-action (main)", "lr_action": "Logistic regression (next action)",
              "ngram_action": "n-gram baseline", "ensemble": "Trajectory-only ensemble (older)",
              "speculator": "LLM speculator (Groq gpt-oss-20b)"}

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.procs: dict[str, Proc] = {}
        self.health: dict | None = None
        threading.Thread(target=self._health_loop, daemon=True).start()

    def start_predictor(self, model: str, theta: float, crd: bool) -> dict:
        if model not in self.MODELS:
            raise ValueError(f"unknown predictor {model}")
        self.stop("predictor")
        env = {"THETA": str(theta), "TTL_S": "60", "CRD_MODE": "k8s" if crd else "off",
               "PREDICTOR_DIR": str(ROOT / "predictor"), "CRD_DRYRUN_FILE": str(STATE / "prewarm_requests.jsonl")}
        if crd:
            env["KUBE_CONTEXT"] = self.cluster.context or ""
        if model == "speculator":
            app_dir = "baselines/llm_speculator"
        else:
            app_dir, env["PREDICTOR_MODEL"] = "predictor", model
        cmd = [PY, "-m", "uvicorn", "service:app", "--app-dir", app_dir, "--host", "127.0.0.1",
               "--port", str(PREDICTOR_PORT), "--log-level", "warning"]
        self.procs["predictor"] = Proc("predictor", cmd, env, {"model": model, "theta": theta,
                                                                "crd": "k8s" if crd else "off"})
        self.health = None
        return self.status()

    def start_controller(self) -> dict:
        self.stop("controller")
        if CONTROLLER_LOG.exists():
            CONTROLLER_LOG.unlink()
        self.cluster._log_pos = 0
        env = {**self.cluster.tool_env(), "CONTROLLER_LOG": str(CONTROLLER_LOG), "MAX_REQUEST_AGE_S": "30"}
        self.procs["controller"] = Proc("controller", [PY, "controller/prewarm_controller.py"], env,
                                        {"context": self.cluster.context})
        return self.status()

    def stop(self, name: str | None = None) -> dict:
        for n in ([name] if name else list(self.procs)):
            p = self.procs.pop(n, None)
            if p:
                p.stop()
            if n == "predictor":
                self.health = None
        return self.status()

    def _health_loop(self):
        while True:
            p = self.procs.get("predictor")
            if p and p.alive():
                try:
                    with urllib.request.urlopen(PREDICTOR_URL + "/healthz", timeout=3) as r:
                        self.health, self.health_at = json.loads(r.read()), time.time()
                except Exception:
                    pass                     # keep the last answer; ready() looks at its age
            time.sleep(2)

    def predictor_ready(self) -> bool:
        """Running and answered /healthz in the last 15 s (one slow check under load is not 'down')."""
        p = self.procs.get("predictor")
        return bool(p and p.alive() and self.health and time.time() - getattr(self, "health_at", 0) < 15)

    def wait_predictor(self, timeout: float = 90) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.predictor_ready():
                return True
            if "predictor" not in self.procs or not self.procs["predictor"].alive():
                return False
            time.sleep(0.5)
        return False

    def status(self) -> dict:
        out = {}
        for n, p in self.procs.items():
            out[n] = {"running": p.alive(), "pid": p.p.pid, "up_s": round(time.time() - p.started), **p.info,
                      "log_tail": p.tail(4)}
        if "predictor" in out:
            out["predictor"]["ready"] = self.predictor_ready()
            out["predictor"]["health"] = self.health
        return out


# ====================================================================== batch runs
class BatchRun:
    """One experiment run: a few commands in order (e.g. set min-scale, then the replay), output streamed to the UI."""

    def __init__(self, spec: dict, steps: list[tuple[str, list[str], dict]], services: Services | None = None):
        self.spec, self.steps, self.services = spec, steps, services
        self.lines: list[str] = []
        self.state, self.result_dir, self.summary, self.exit_code = "running", None, None, None
        self.started, self.ended = time.time(), None
        self.p: subprocess.Popen | None = None
        self.stop_flag = False
        threading.Thread(target=self._go, daemon=True).start()

    def _say(self, s: str):
        self.lines.append(s)

    def _go(self):
        try:
            for title, cmd, env in self.steps:
                if self.stop_flag:
                    break
                if title == "__predictor__":            # wait for the predictor the run needs
                    self._say("• waiting for the predictor to load ...")
                    if not self.services.wait_predictor():
                        raise RuntimeError("the predictor did not start; see its log on the Services card")
                    self._say(f"  predictor ready ({(self.services.health or {}).get('model')})")
                    continue
                self._say(f"• {title}")
                self._say("  $ " + " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd)))
                self.p = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **env, "PYTHONUNBUFFERED": "1",
                                                              "PYTHONIOENCODING": "utf-8", "MSYS_NO_PATHCONV": "1"},
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                          encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
                for line in self.p.stdout:
                    line = line.rstrip("\n")
                    self._say(line)
                    for tag in ("saved to ", "-> ", "Result: "):
                        if line.strip().startswith(tag) or (tag == "Result: " and tag in line):
                            cand = line.split(tag, 1)[1].strip().split()[0] if line.split(tag, 1)[1].strip() else ""
                            path = (ROOT / cand) if cand and not Path(cand).is_absolute() else Path(cand)
                            if cand and path.exists() and path.is_dir():
                                self.result_dir = path
                self.exit_code = self.p.wait()
                if self.exit_code != 0:
                    raise RuntimeError(f"step failed (exit code {self.exit_code})")
            self.state = "stopped" if self.stop_flag else "done"
        except Exception as e:
            self.state = "stopped" if self.stop_flag else "failed"
            self._say(f"✗ {e}")
        finally:
            self.ended = time.time()
            if self.result_dir and (self.result_dir / "summary.json").exists():
                self.summary = summarize(self.result_dir)

    def stop(self):
        self.stop_flag = True
        if self.p and self.p.poll() is None:
            self.p.terminate()

    def view(self, after: int = 0) -> dict:
        return {"state": self.state, "spec": self.spec, "lines": self.lines[after:], "n": len(self.lines),
                "started": self.started, "ended": self.ended, "exit_code": self.exit_code,
                "result_dir": str(self.result_dir.relative_to(ROOT)) if self.result_dir else None,
                "summary": self.summary}


def build_run(spec: dict, cluster: Cluster, services: Services) -> list[tuple[str, list[str], dict]]:
    """The commands for one batch run, from the form in the Experiments tab."""
    mode, config, where = spec["mode"], spec["config"], spec.get("where", "laptop")
    pred = spec.get("predictor", "gbt")
    tasks = [str(int(t)) for t in spec.get("tasks") or []]
    steps: list[tuple[str, list[str], dict]] = []
    env = cluster.tool_env()

    args: list[str] = []
    if tasks:
        args += ["--tasks", *tasks]
    if mode == "replay":
        if spec.get("fanout", True):
            args.append("--fanout")
        for k, flag in (("episodes", "--episodes"), ("seed", "--seed"), ("rate", "--rate"),
                        ("user_turn_median", "--user-turn-median"), ("think_median", "--think-median")):
            if spec.get(k) not in (None, ""):
                args += [flag, str(spec[k])]
        if spec.get("hint_timing") == "legacy":
            args += ["--hint-timing", "legacy"]
    else:
        args += ["--backend", spec.get("llm_backend", "groq"), "--agent-model", spec.get("agent_model", "qwen/qwen3.8-27b"),
                 "--user-model", spec.get("user_model", "openai/gpt-oss-20b")]
        for k, flag in (("repeats", "--repeats"), ("user_delay_median", "--user-delay-median")):
            if spec.get(k) not in (None, ""):
                args += [flag, str(spec[k])]

    if where == "cluster":                     # EKS_deployment/cloud/run_cloud.sh: the runner is a pod
        b = bash()
        if not b:
            raise ValueError("Git Bash not found (needed for run_cloud.sh)")
        if pred not in ("gbt", "speculator") and config == "C3":
            raise ValueError("in-cluster runs support the gbt and speculator predictors (cloud_up.sh deploys those)")
        label = spec.get("label") or "ui"
        cenv = {**env, "MODE": mode, "CONFIG": config, "PREDICTOR": pred, "WORKLOAD": "none", "LABEL": label,
                "PYTHON": PY}
        steps.append((f"{mode} {config} inside the cluster (run_cloud.sh)",
                      [b, "EKS_deployment/cloud/run_cloud.sh", *args], cenv))
        return steps

    # ---- laptop runner
    want = 1 if config == "C1" else 0
    if cluster.min_scale != want:
        b = bash()
        if not b:
            raise ValueError("Git Bash not found (needed for deploy/knative/deploy.sh)")
        steps.append((f"set the tools to min-scale {want} ({'always warm' if want else 'scale to zero'})",
                      [b, "deploy/knative/deploy.sh", str(want)], {**env, "SKIP_BUILD": "1", "PYTHON": PY}))
    label = spec.get("label") or "ui"
    if config == "C3":
        if not (services.procs.get("predictor") and services.procs["predictor"].alive()
                and services.procs["predictor"].info.get("model") == pred
                and services.procs["predictor"].info.get("crd") == "k8s"):
            services.start_predictor(pred, float(spec.get("theta", 0.3)), crd=True)
        if not (services.procs.get("controller") and services.procs["controller"].alive()):
            services.start_controller()
        steps.append(("__predictor__", [], {}))
        args += ["--predictor-url", PREDICTOR_URL]
        if pred == "speculator" and mode == "replay":
            args += ["--hint-context", "--hint-timeout", "10"]
        label = f"{label}-{pred}"
    script = "test/test_knative/w1_replay.py" if mode == "replay" else "test/test_knative/w1_live.py"
    steps.append((f"{mode} {config}{' (' + pred + ')' if config == 'C3' else ''} from the laptop",
                  [PY, script, *args, "--label", label], env))
    return steps


# ====================================================================== results
def summarize(d: Path) -> dict | None:
    try:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    lat = (s.get("latency_ms") or {}).get("all") or {}
    pred = ((cfg.get("prewarm") or {}).get("predictor") or {}).get("model")
    return {"name": d.name, "path": str(d.relative_to(ROOT)).replace("\\", "/"),
            "where": d.parent.name, "config": s.get("config") or cfg.get("config"),
            "mode": "live" if "W1-live" in d.name else "replay", "predictor": pred,
            "calls": s.get("tool_calls"), "cold": s.get("cold_calls"), "cold_share": s.get("cold_share"),
            "p50": lat.get("p50"), "p95": lat.get("p95"), "p99": lat.get("p99"), "mean": lat.get("mean"),
            "pod_seconds": s.get("pod_seconds_seen"), "pods": s.get("pods_started"),
            "episodes": s.get("episodes"), "hints": s.get("hints"),
            "cold_added_s": s.get("cold_start_time_added_s"), "files": sorted(p.name for p in d.iterdir())}


def list_results() -> dict:
    runs, compares = [], []
    for where, base in RESULT_DIRS.items():
        if not base.exists():
            continue
        for d in sorted(base.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            if d.name.startswith("compare_"):
                pngs = sorted(p.name for p in d.glob("*.png"))
                if pngs:
                    compares.append({"name": d.name, "path": str(d.relative_to(ROOT)).replace("\\", "/"),
                                     "where": where, "pngs": pngs})
            elif (d / "summary.json").exists():
                s = summarize(d)
                if s:
                    s["where"] = where
                    runs.append(s)
    runs.sort(key=lambda r: r["name"], reverse=True)          # newest first (names start with the date)
    compares.sort(key=lambda c: c["name"], reverse=True)
    return {"runs": runs, "compares": compares}


def make_plots(paths: list[str]) -> dict:
    dirs = [(ROOT / p).resolve() for p in paths]
    for d in dirs:
        inside = any(base.resolve() in d.parents for base in RESULT_DIRS.values())
        if not inside or not (d / "summary.json").exists():
            raise ValueError(f"not a run folder: {d}")
    base = dirs[0].parent if dirs[0].parent.name != "results" else RESULT_DIRS["local"]
    out = base / f"compare_{time.strftime('%Y-%m-%d_%H%M%S')}_ui_{len(dirs)}runs"
    p = subprocess.run([plot_python(), "test/test_knative/plot_compare.py", *map(str, dirs), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=180, creationflags=NO_WINDOW,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-800:])
    return {"name": out.name, "path": str(out.relative_to(ROOT)).replace("\\", "/"),
            "pngs": sorted(x.name for x in out.glob("*.png"))}
