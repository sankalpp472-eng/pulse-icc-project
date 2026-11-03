"""
The UI's back office: which cluster it talks to, the predictor / controller processes it starts,
batch runs (w1_replay.py / w1_live.py / run_cloud.sh) and the result folders.

The UI runs either on the laptop (kubectl context: kind or EKS) or INSIDE the EKS cluster as the pulse-ui pod
(EKS_deployment/cloud/ui_cloud.sh): then it calls the tools at their in-cluster URLs, uses the pod's service
account, the pulse-controller Deployment does the prewarming, and runs write to UI_RESULTS_DIR in the pod.

Nothing here prints secrets: processes inherit agent/.env (laptop) or the pulse-llm-keys Secret (pod).
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
IN_CLUSTER = bool(os.getenv("KUBERNETES_SERVICE_HOST"))   # the pulse-ui pod on EKS
STATE = Path(os.getenv("UI_STATE_DIR", ROOT / "ui" / ".state"))   # logs of the processes the UI starts
PY = sys.executable                             # the agent's Python (agent/.venv) runs the UI and everything it starts
PREDICTOR_PORT = int(os.getenv("UI_PREDICTOR_PORT", "8000"))
PREDICTOR_URL = f"http://127.0.0.1:{PREDICTOR_PORT}"
CONTROLLER_LOG = STATE / "controller.jsonl"
CONTROLLER_DEPLOY = "pulse-controller"           # in the cluster: the controller is this Deployment (cloud_up.sh)
# Where new runs and plots go: the pod's own folder in the cluster (fetch with ui_cloud.sh fetch), else results/
WRITE_DIR = Path(os.environ["UI_RESULTS_DIR"]) if os.getenv("UI_RESULTS_DIR") else None
RESULT_DIRS = {"local": ROOT / "results" / "local_runs", "cloud": ROOT / "results" / "cloud_runs",
               "legacy": ROOT / "test" / "test_knative" / "results"}
if WRITE_DIR:
    RESULT_DIRS = {"this pod": WRITE_DIR, **RESULT_DIRS}
TOOL_NAMES = ["book_reservation", "calculate", "cancel_reservation", "get_reservation_details", "get_user_details",
              "list_all_airports", "search_direct_flight", "search_onestop_flight", "send_certificate", "think",
              "transfer_to_human_agents", "update_reservation_baggages", "update_reservation_flights",
              "update_reservation_passengers"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
STARTED_ISO = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())   # the controller logs UTC ISO times


def rel(p: Path) -> str:
    """A result path as the browser sees it: relative to the repo, or 'pod/...' inside UI_RESULTS_DIR."""
    p = Path(p).resolve()
    if WRITE_DIR and (p == WRITE_DIR.resolve() or WRITE_DIR.resolve() in p.parents):
        return "pod/" + p.relative_to(WRITE_DIR.resolve()).as_posix()
    return p.relative_to(ROOT).as_posix()


def unrel(s: str) -> Path:
    """The inverse of rel()."""
    return (WRITE_DIR / s[4:]).resolve() if WRITE_DIR and s.startswith("pod/") else (ROOT / s).resolve()


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
        self._seen: set[str] = set()           # controller log lines already turned into events (in the cluster)
        self.controller_ready = None
        threading.Thread(target=self._poll, daemon=True).start()

    def detect(self) -> dict:
        if IN_CLUSTER:                          # the pod's service account; kubectl has no context to name
            self.context = os.getenv("CLUSTER_NAME", "pulse-eks") + " (in-cluster)"
        else:
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
        if IN_CLUSTER:                          # straight to the Knative service, no ingress / Host header
            self.url_tpl, self.host_tpl = "http://{svc}.default.svc.cluster.local/mcp", ""
        elif self.context.startswith("kind-"):
            self.url_tpl, self.host_tpl = "http://127.0.0.1:8081/mcp", "{svc}.default.example.com"
        else:                                   # EKS from the laptop: Kourier's load balancer
            code, lb = _run(["kubectl", "get", "svc", "kourier", "-n", "kourier-system", "-o",
                             "jsonpath={.status.loadBalancer.ingress[0].hostname}"], 10)
            self.url_tpl = f"http://{lb}/mcp" if code == 0 and lb else None
            self.host_tpl = "{svc}.default.example.com"
        return self.info()

    def info(self) -> dict:
        return {"mode": self.mode, "context": self.context, "reachable": self.reachable, "detail": self.detail,
                "tools": self.tools_found, "min_scale": self.min_scale, "ingress": self.url_tpl,
                "in_cluster": IN_CLUSTER, "controller_deployment": self.controller_ready if IN_CLUSTER else None}

    def tool_env(self) -> dict:
        """Environment for anything that calls the tools on the real cluster."""
        env = {"TOOL_BACKEND": "mcp"}
        if not IN_CLUSTER:
            env["KUBE_CONTEXT"] = self.context or ""
        if self.url_tpl:
            env["MCP_URL_TEMPLATE"] = self.url_tpl
            env["MCP_HOST_TEMPLATE"] = self.host_tpl or ""
        return env

    def set_min_scale(self, n: int, say=print) -> None:
        """C1 = 1 (always warm), C0 / C3 = 0: patch every tool's Knative Service (a new Revision), wait for Ready.
        The other settings (concurrency, max-scale, window) stay as deploy/knative/deploy.sh set them."""
        patch = json.dumps({"spec": {"template": {"metadata": {"annotations": {
            "autoscaling.knative.dev/min-scale": str(n)}}}}})
        code, out = _run(["kubectl", "get", "ksvc", "-l", "pulse/role=mcp-tool", "-o", "name"], 15)
        names = out.split() if code == 0 else []
        if not names:
            raise RuntimeError(f"no tools found ({out[:200]})")
        for name in names:
            code, out = _run(["kubectl", "patch", name, "--type", "merge", "-p", patch], 20)
            if code != 0:
                raise RuntimeError(f"could not patch {name}: {out[:200]}")
        say(f"  {len(names)} tools patched to min-scale {n}; waiting for the new revisions to be Ready ...")
        code, out = _run(["kubectl", "wait", "--for=condition=Ready", "ksvc", "-l", "pulse/role=mcp-tool",
                          "--timeout=600s"], 620)
        if code != 0:
            raise RuntimeError(f"tools not Ready: {out[-300:]}")
        self.min_scale = n
        say("  tools Ready")

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
                if IN_CLUSTER:
                    self._tail_controller_deployment()
                else:
                    self._tail_controller()
            time.sleep(1.0)

    def _tail_controller_deployment(self):
        """In the cluster the controller is the pulse-controller Deployment: read its JSON log lines."""
        code, out = _run(["kubectl", "logs", f"deploy/{CONTROLLER_DEPLOY}", "--tail=200", "--request-timeout=4s"], 6)
        self.controller_ready = code == 0
        if code != 0:
            return
        for line in out.splitlines():
            if not line.startswith("{") or line in self._seen:
                continue
            self._seen.add(line)
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") in ("warmed", "evicted", "failed") and str(r.get("time", "")) >= STARTED_ISO:
                r["source"] = "controller"
                self.events.append(r)
        if len(self._seen) > 5000:
            self._seen = set(list(self._seen)[-2000:])

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
def norm_policy(p: dict | None) -> dict:
    """The prewarm policy as the predictor reads it (predictor/service.py, LATENCY_IDEAS_RUN.md):
    theta = every tool with P >= theta (the original); topk = the top_k most likely tools, each with P >= p_min;
    lookahead = on a likely reply step, also prewarm the tool the agent calls after the customer answers."""
    p = p or {}
    return {"policy": "topk" if p.get("policy") == "topk" else "theta",
            "top_k": max(1, min(5, int(p.get("top_k") or 2))), "p_min": float(p.get("p_min", 0.1)),
            "lookahead": bool(p.get("lookahead")), "theta_next": float(p.get("theta_next", 0.3))}


def policy_env(p: dict) -> dict:
    """Environment for predictor/service.py. A predictor without the latency patch ignores these and stays on theta."""
    return {"PREWARM_POLICY": p["policy"], "TOP_K": str(p["top_k"]), "P_MIN": str(p["p_min"]),
            "LOOKAHEAD": "1" if p["lookahead"] else "0", "THETA_NEXT": str(p["theta_next"])}


def policy_label(p: dict | None, theta: float | None = None) -> str:
    p = norm_policy(p)
    s = f"top-{p['top_k']} (P ≥ {p['p_min']:g})" if p["policy"] == "topk" else f"θ {theta if theta is not None else ''}".strip()
    return s + (f" + lookahead {p['theta_next']:g}" if p["lookahead"] else "")


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

    def start_predictor(self, model: str, theta: float, crd: bool, policy: dict | None = None) -> dict:
        if model not in self.MODELS:
            raise ValueError(f"unknown predictor {model}")
        self.stop("predictor")
        policy = norm_policy(policy)
        env = {"THETA": str(theta), "TTL_S": "60", "CRD_MODE": "k8s" if crd else "off",
               "PREDICTOR_DIR": str(ROOT / "predictor"), "CRD_DRYRUN_FILE": str(STATE / "prewarm_requests.jsonl"),
               **policy_env(policy)}
        if crd and not IN_CLUSTER:          # in the cluster crd_writer.py uses the pod's service account
            env["KUBE_CONTEXT"] = self.cluster.context or ""
        if model == "speculator":
            app_dir = "baselines/llm_speculator"
        else:
            app_dir, env["PREDICTOR_MODEL"] = "predictor", model
        cmd = [PY, "-m", "uvicorn", "service:app", "--app-dir", app_dir, "--host", "127.0.0.1",
               "--port", str(PREDICTOR_PORT), "--log-level", "warning"]
        self.procs["predictor"] = Proc("predictor", cmd, env, {"model": model, "theta": theta, "policy": policy,
                                                                "crd": "k8s" if crd else "off"})
        self.health = None
        return self.status()

    def controller_running(self) -> bool:
        if IN_CLUSTER:
            return bool(self.cluster.controller_ready)
        return bool(self.procs.get("controller") and self.procs["controller"].alive())

    def start_controller(self) -> dict:
        if IN_CLUSTER:                 # the pulse-controller Deployment already watches the PrewarmRequests
            return self.status()
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
            out["predictor"]["policy_label"] = policy_label(out["predictor"].get("policy"), out["predictor"].get("theta"))
            # does this predictor understand the policy settings? (/healthz reports "policy" after the latency patch)
            out["predictor"]["policy_supported"] = bool(self.health and "policy" in self.health) if self.health else None
        if IN_CLUSTER:
            out["controller"] = {"running": self.controller_running(), "deployment": CONTROLLER_DEPLOY}
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
                        raise RuntimeError("the predictor did not start; see its log on the Predictor card")
                    self._say(f"  predictor ready ({(self.services.health or {}).get('model')})")
                    continue
                if callable(cmd):                       # a step done in Python (e.g. patch the tools' min-scale)
                    self._say(f"• {title}")
                    cmd(self._say)
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
                    self._find_result_dir(line)
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

    def _find_result_dir(self, line: str):
        """The runners print '-> <folder>' / 'saved to <folder>'; run_cloud.sh prints 'Result: <folder> ...'."""
        s = line.strip()
        for tag in ("saved to ", "-> ", "Done (runner exit code"):
            if not s.startswith(tag):
                continue
            rest = s.split("Result:", 1)[1].split()[0] if tag.startswith("Done") and "Result:" in s else s[len(tag):]
            path = Path(rest.strip())
            path = path if path.is_absolute() else ROOT / path
            if path.is_dir():
                self.result_dir = path

    def stop(self):
        self.stop_flag = True
        if self.p and self.p.poll() is None:
            self.p.terminate()

    def view(self, after: int = 0) -> dict:
        return {"state": self.state, "spec": self.spec, "lines": self.lines[after:], "n": len(self.lines),
                "started": self.started, "ended": self.ended, "exit_code": self.exit_code,
                "result_dir": rel(self.result_dir) if self.result_dir else None,
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

    if where == "cluster" and not IN_CLUSTER:  # EKS_deployment/cloud/run_cloud.sh: the runner is a pod
        b = bash()
        if not b:
            raise ValueError("Git Bash not found (needed for run_cloud.sh)")
        if pred not in ("gbt", "speculator") and config == "C3":
            raise ValueError("in-cluster runs support the gbt and speculator predictors (cloud_up.sh deploys those)")
        pol = norm_policy(spec.get("policy"))
        if config == "C3" and (pol["policy"] != "theta" or pol["lookahead"]):
            raise ValueError("run_cloud.sh uses the cluster's pulse-predictor, so set its policy there: kubectl set env "
                             "deploy/pulse-predictor PREWARM_POLICY=topk TOP_K=2 P_MIN=0.1 LOOKAHEAD=0 (then pick θ here)")
        label = spec.get("label") or "ui"
        cenv = {**env, "MODE": mode, "CONFIG": config, "PREDICTOR": pred, "WORKLOAD": "none", "LABEL": label,
                "PYTHON": PY}
        steps.append((f"{mode} {config} inside the cluster (run_cloud.sh)",
                      [b, "EKS_deployment/cloud/run_cloud.sh", *args], cenv))
        return steps

    # ---- the runner runs where the UI runs: the laptop, or the pulse-ui pod in the cluster
    want = 1 if config == "C1" else 0
    if cluster.min_scale != want:
        steps.append((f"set the tools to min-scale {want} ({'always warm' if want else 'scale to zero'})",
                      lambda say: cluster.set_min_scale(want, say), {}))
    if WRITE_DIR:
        env = {**env, "PULSE_RESULTS_DIR": str(WRITE_DIR), "PULSE_RUNNER": "pulse-ui pod"}
    label = spec.get("label") or ("ui-eks" if IN_CLUSTER else "ui")
    if config == "C3":
        policy = norm_policy(spec.get("policy"))
        cur = services.procs.get("predictor")
        if not (cur and cur.alive() and cur.info.get("model") == pred and cur.info.get("crd") == "k8s"
                and abs(cur.info.get("theta", -1) - float(spec.get("theta", 0.3))) < 1e-9
                and cur.info.get("policy") == policy):
            services.start_predictor(pred, float(spec.get("theta", 0.3)), crd=True, policy=policy)
        if not services.controller_running():
            services.start_controller()
        steps.append(("__predictor__", [], {}))
        args += ["--predictor-url", PREDICTOR_URL]
        if pred == "speculator" and mode == "replay":
            args += ["--hint-context", "--hint-timeout", "10"]
        tag = ("-top%d" % policy["top_k"] if policy["policy"] == "topk" else "") + ("-look" if policy["lookahead"] else "")
        label = f"{label}-{pred}{tag}"
    script = "test/test_knative/w1_replay.py" if mode == "replay" else "test/test_knative/w1_live.py"
    steps.append((f"{mode} {config}{' (' + pred + ')' if config == 'C3' else ''} "
                  f"{'in the pulse-ui pod' if IN_CLUSTER else 'from the laptop'}",
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
    return {"name": d.name, "path": rel(d),
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
                    compares.append({"name": d.name, "path": rel(d), "where": where, "pngs": pngs})
            elif (d / "summary.json").exists():
                s = summarize(d)
                if s:
                    s["where"] = where
                    s["mtime"] = (d / "summary.json").stat().st_mtime
                    runs.append(s)
    runs.sort(key=lambda r: r["mtime"], reverse=True)         # newest first
    compares.sort(key=lambda c: c["name"], reverse=True)
    return {"runs": runs, "compares": compares}


def result_path(s: str) -> Path:
    """A browser-supplied result path, only if it is inside one of the result folders."""
    p = unrel(s)
    if not any(base.resolve() in p.parents for base in RESULT_DIRS.values()):
        raise ValueError(f"not a result path: {s}")
    return p


def make_plots(paths: list[str]) -> dict:
    dirs = [result_path(p) for p in paths]
    for d in dirs:
        if not (d / "summary.json").exists():
            raise ValueError(f"not a run folder: {d.name}")
    # next to the first run, unless that folder is read-only (the results baked into the image): the pod's folder
    base = WRITE_DIR if WRITE_DIR else dirs[0].parent
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"compare_{time.strftime('%Y-%m-%d_%H%M%S')}_ui_{len(dirs)}runs"
    p = subprocess.run([plot_python(), "test/test_knative/plot_compare.py", *map(str, dirs), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=180, creationflags=NO_WINDOW,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-800:])
    return {"name": out.name, "path": rel(out), "pngs": sorted(x.name for x in out.glob("*.png"))}


def zip_folder(s: str) -> tuple[bytes, str]:
    """A run / comparison folder as a zip, to take it home from the pod (or use ui_cloud.sh fetch)."""
    import io
    import zipfile
    d = result_path(s)
    if not d.is_dir():
        raise ValueError("not a folder")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.rglob("*"):
            if f.is_file():
                z.write(f, f"{d.name}/{f.relative_to(d).as_posix()}")
    return buf.getvalue(), d.name + ".zip"
