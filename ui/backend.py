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
import urllib.parse
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
               "load": ROOT / "results" / "load_runs", "legacy": ROOT / "test" / "test_knative" / "results"}
# load tests (loadtest/) write here: the pod's own folder in the cluster, else results/load_runs
LOAD_WRITE_DIR = WRITE_DIR.parent / "load_runs" if WRITE_DIR else None
if WRITE_DIR:
    RESULT_DIRS = {"this pod": WRITE_DIR, "this pod (load)": LOAD_WRITE_DIR, **RESULT_DIRS}
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
              "speculator": "LLM speculator (Groq gpt-oss-20b)",
              "pulse_v3": "pulse-v3 text + tool graph (pulse_pred/)"}

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.procs: dict[str, Proc] = {}
        self.health: dict | None = None
        threading.Thread(target=self._health_loop, daemon=True).start()

    def start_predictor(self, model: str, theta: float, crd: bool, policy: dict | None = None,
                        model_version: str | None = None) -> dict:
        """model_version (pulse_v3 only): serve that registry version instead of the latest (Models tab)."""
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
        elif model == "pulse_v3":
            app_dir = "pulse_pred"
            if model_version and MODEL_URI:
                env["PULSE_MODEL_URI"] = f"{MODEL_URI.rstrip('/')}/{model_version}"
        else:
            app_dir, env["PREDICTOR_MODEL"] = "predictor", model
        cmd = [PY, "-m", "uvicorn", "service:app", "--app-dir", app_dir, "--host", "127.0.0.1",
               "--port", str(PREDICTOR_PORT), "--log-level", "warning"]
        self.procs["predictor"] = Proc("predictor", cmd, env, {"model": model, "theta": theta, "policy": policy,
                                                                "crd": "k8s" if crd else "off",
                                                                "model_version": model_version})
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

    def start_bgload(self, agents: int, predictor_url: str | None) -> dict:
        """Live demo, 'background load': N more agents replaying recorded episodes (loadtest/locust_episodes.py)
        while the demo episode runs, sending hints too when the demo uses the predictor. Stopped with the episode."""
        self.stop("bgload")
        env = {**self.cluster.tool_env(), **({"PULSE_LOAD_DIR": str(LOAD_WRITE_DIR)} if LOAD_WRITE_DIR else {})}
        cmd = [PY, "-m", "locust", "-f", "loadtest/locust_episodes.py", "--headless", "--only-summary",
               "--stages", f"{agents}:7200:{agents}", "--pulse-config", "C3" if predictor_url else "C0",
               "--label", "ui-background", "--loglevel", "WARNING"]
        if predictor_url:
            cmd += ["--predictor-url", predictor_url]
        self.procs["bgload"] = Proc("bgload", cmd, env, {"agents": agents})
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
            out["predictor"]["serving_version"] = served_version(self.health)
            out["predictor"]["policy_label"] = policy_label(out["predictor"].get("policy"), out["predictor"].get("theta"))
            # does this predictor understand the policy settings? (/healthz reports "policy" after the latency patch)
            out["predictor"]["policy_supported"] = bool(self.health and "policy" in self.health) if self.health else None
        if IN_CLUSTER:
            out["controller"] = {"running": self.controller_running(), "deployment": CONTROLLER_DEPLOY}
        return out


# ====================================================================== models (pulse-v3's S3 registry)
MODEL_URI = os.getenv("PULSE_MODEL_URI", "")     # s3://<bucket>/pulse-v3 (pulse_pred/model_store.py)


def served_version(health: dict | None) -> str | None:
    """The registry version a pulse_v3 predictor serves: its /healthz model_file is .../models/<version>/predictor.pkl."""
    f = (health or {}).get("model_file") or ""
    parts = Path(f).parts
    return parts[-2] if len(parts) >= 3 and parts[-3] == "models" else None


def _store():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pulse_pred import model_store
    return model_store


def models_list() -> dict:
    """Every pulse-v3 version in the registry (newest first), or why they can't be listed."""
    if not MODEL_URI:
        return {"uri": None, "versions": [], "error": "PULSE_MODEL_URI is not set: start the UI with "
                "PULSE_MODEL_URI=s3://<bucket>/pulse-v3 (terraform -chdir=terraform output -raw model_uri)"}
    try:
        return {"uri": MODEL_URI, "versions": _store().versions(MODEL_URI), "error": None}
    except ImportError as e:
        return {"uri": MODEL_URI, "versions": [], "error": f"{e}: pip install -r pulse_pred/requirements.txt"}
    except Exception as e:                       # no AWS credentials (laptop) or no IAM role (UI pod)
        return {"uri": MODEL_URI, "versions": [], "error": f"{type(e).__name__}: {e}"[:300] +
                (" (the UI pod has no S3 role yet: use the laptop UI)" if IN_CLUSTER else "")}


def models_promote(version: str):
    _store().promote(MODEL_URI, version)


def build_retrain(spec: dict) -> list[tuple[str, list[str], dict]]:
    """Retraining (pulse_pred/train.py): on this machine, or as the training Job in the cluster."""
    if not MODEL_URI:
        raise ValueError("PULSE_MODEL_URI is not set (see the Models tab)")
    promote, notes = spec.get("promote", "auto"), str(spec.get("notes") or "")
    if spec.get("where") == "cluster":
        b = bash()
        if IN_CLUSTER or not b:
            raise ValueError("the training Job is started from the laptop (deploy/training/run_training.sh)")
        return [("retrain pulse-v3 as a Job in the cluster (deploy/training/run_training.sh)",
                 [b, "deploy/training/run_training.sh"],
                 {"PULSE_MODEL_URI": MODEL_URI, "PROMOTE": promote, "NOTES": notes})]
    return [("retrain pulse-v3 here (pulse_pred/train.py)",
             [PY, "pulse_pred/train.py", "--uri", MODEL_URI, "--promote", promote, "--notes", notes], {})]


# ====================================================================== monitoring (deploy/monitoring/)
MON_NS, GRAFANA_SVC = "monitoring", "pulse-mon-grafana"
GRAFANA_LOCAL_PORT = int(os.getenv("UI_GRAFANA_PORT", "13000"))
DASHBOARDS = {"pulse-overview": "Pulse overview", "pulse-predictor": "Predictor", "pulse-cluster": "Cluster"}
_READY_TOOL_PODS = ('sum(kube_pod_status_ready{namespace="default",condition="true"} * on (namespace, pod) group_left () '
                    'kube_pod_labels{namespace="default",label_serving_knative_dev_service!=""})')


class Monitoring:
    """Prometheus + Grafana (deploy/monitoring/install.sh), Knative mode only. The page sees Grafana under this
    server's /grafana/ (server.py proxies it), so one tunnel to the UI is enough in the cluster too. Grafana is
    reached on the laptop through a kubectl port-forward kept up here, in the cluster at its Service. Prometheus
    is queried through Grafana's data source (uid prometheus): no second tunnel."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.installed, self.reachable, self.detail = None, False, "Knative mode only"
        self.figures: dict = {}
        self.pf: subprocess.Popen | None = None
        threading.Thread(target=self._loop, daemon=True).start()

    def base(self) -> str:
        return (f"http://{GRAFANA_SVC}.{MON_NS}.svc.cluster.local" if IN_CLUSTER
                else f"http://127.0.0.1:{GRAFANA_LOCAL_PORT}")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base() + "/api/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _stop_pf(self):
        if self.pf and self.pf.poll() is None:
            self.pf.terminate()
        self.pf = None

    def check(self):
        if self.cluster.mode != "knative" or not self.cluster.reachable:
            self._stop_pf()
            self.installed, self.reachable, self.detail, self.figures = None, False, "Knative mode only", {}
            return
        if IN_CLUSTER:
            self.installed = self.reachable = self._healthy()
            self.detail = "ok" if self.reachable else "not installed (deploy/monitoring/install.sh, from the laptop)"
        else:
            code, _ = _run(["kubectl", "get", "svc", GRAFANA_SVC, "-n", MON_NS, "-o", "name"], 8)
            self.installed = code == 0
            if not self.installed:
                self._stop_pf()
                self.reachable, self.detail, self.figures = False, "not installed", {}
                return
            if not (self.pf and self.pf.poll() is None):
                self.pf = subprocess.Popen(["kubectl", "port-forward", "-n", MON_NS, f"svc/{GRAFANA_SVC}",
                                            f"{GRAFANA_LOCAL_PORT}:80"], stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
                time.sleep(2)
            self.reachable = self._healthy()
            self.detail = "ok" if self.reachable else "port-forward to Grafana not answering yet"
        if self.reachable:
            self.figures = self._figures()

    def _loop(self):
        while True:
            try:
                self.check()
            except Exception as e:                       # never let monitoring break the UI
                self.reachable, self.detail = False, f"{type(e).__name__}: {e}"[:200]
            time.sleep(10)

    def query(self, promql: str) -> float | None:
        """One instant Prometheus query (a single number), through Grafana's data source."""
        url = (self.base() + "/api/datasources/proxy/uid/prometheus/api/v1/query?"
               + urllib.parse.urlencode({"query": promql}))
        with urllib.request.urlopen(url, timeout=5) as r:
            res = json.loads(r.read()).get("data", {}).get("result", [])
        return round(float(res[0]["value"][1]), 3) if res else None

    def _figures(self) -> dict:
        """The cluster-measured numbers the cards show next to the UI's own."""
        out = {"tool_pods": self.query(_READY_TOOL_PODS),
               "tool_pod_s_15m": self.query(f"sum_over_time(({_READY_TOOL_PODS})[15m:15s]) * 15")}
        # the predictor this UI started: scraped as the UI pod (EKS) or from the laptop (kind); not on EKS-from-laptop
        comp = "pulse-ui" if IN_CLUSTER else "laptop" if (self.cluster.context or "").startswith("kind-") else None
        if comp:
            p95 = self.query(f'histogram_quantile(0.95, sum by (le) '
                             f'(rate(predictor_request_seconds_bucket{{component="{comp}"}}[2m])))')
            out["predictor_p95_ms"] = round(p95 * 1000, 1) if p95 is not None and p95 == p95 else None
        return out

    def info(self) -> dict:
        return {"installed": self.installed, "reachable": self.reachable, "detail": self.detail,
                "url": "/grafana/", "dashboards": DASHBOARDS, "figures": self.figures,
                "can_install": not IN_CLUSTER and bool(bash())}


def run_window(path: str) -> tuple[float, float]:
    """A run's start and end (epoch s) from its calls.csv, for a Grafana link covering exactly that run."""
    import csv
    d = result_path(path)
    with open(d / "calls.csv" if (d / "calls.csv").exists() else d / "requests.csv", encoding="utf-8") as f:   # load runs
        rows = [(float(r["t_start"]), float(r.get("latency_ms") or 0) / 1000) for r in csv.DictReader(f) if r.get("t_start")]
    if not rows:
        raise ValueError("no calls in this run")
    return min(t for t, _ in rows), max(t + l for t, l in rows)


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
                # PULSE_METRICS_PORT: the replay serves live metrics for Prometheus (deploy/monitoring)
                self.p = subprocess.Popen(cmd, cwd=ROOT, env={"PULSE_METRICS_PORT": "9105", **os.environ, **env,
                                                              "PYTHONUNBUFFERED": "1",
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
    if mode == "load":
        return build_load(spec, cluster, services, tasks)
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
        if pred in ("speculator", "pulse_v3") and mode == "replay":
            args += ["--hint-context", "--hint-timeout", "10"]
        tag = ("-top%d" % policy["top_k"] if policy["policy"] == "topk" else "") + ("-look" if policy["lookahead"] else "")
        label = f"{label}-{pred}{tag}"
    script = "test/test_knative/w1_replay.py" if mode == "replay" else "test/test_knative/w1_live.py"
    steps.append((f"{mode} {config}{' (' + pred + ')' if config == 'C3' else ''} "
                  f"{'in the pulse-ui pod' if IN_CLUSTER else 'from the laptop'}",
                  [PY, script, *args, "--label", label], env))
    return steps


def ensure_predictor(spec: dict, services: Services, crd: bool) -> str:
    """Start the predictor the spec asks for, unless exactly that one already runs; returns its model name."""
    pred, theta, policy = spec.get("predictor", "gbt"), float(spec.get("theta", 0.3)), norm_policy(spec.get("policy"))
    cur = services.procs.get("predictor")
    if not (cur and cur.alive() and cur.info.get("model") == pred and cur.info.get("crd") == ("k8s" if crd else "off")
            and abs(cur.info.get("theta", -1) - theta) < 1e-9 and cur.info.get("policy") == policy):
        services.start_predictor(pred, theta, crd=crd, policy=policy)
    if crd and not services.controller_running():
        services.start_controller()
    return pred


def build_load(spec: dict, cluster: Cluster, services: Services, tasks: list[str]) -> list[tuple[str, list[str], dict]]:
    """A load test (loadtest/): Locust (closed loop, users ramp through --stages) or asyncio (open loop: exact
    hints/s, or tool bursts). The predictor test needs no cluster; tools and episodes need Knative."""
    target, engine, config = spec.get("target", "predictor"), spec.get("engine", "locust"), spec.get("config", "C0")
    if target not in ("predictor", "tools", "episodes"):
        raise ValueError(f"unknown load target {target}")
    if engine == "asyncio" and target == "episodes":
        raise ValueError("whole episodes are a Locust test (asyncio does the predictor and tool bursts)")
    knative = cluster.mode == "knative"
    if target != "predictor" and not knative:
        raise ValueError("the tools and episodes load tests need the real cluster: switch to Knative (top bar)")
    steps: list[tuple] = []
    env = {**(cluster.tool_env() if knative else {}), "PULSE_RUNNER": "pulse-ui pod" if IN_CLUSTER else "laptop"}
    if LOAD_WRITE_DIR:
        env["PULSE_LOAD_DIR"] = str(LOAD_WRITE_DIR)
    label = spec.get("label") or ("ui-eks" if IN_CLUSTER else "ui")
    if target != "predictor":
        want = 1 if config == "C1" else 0
        if cluster.min_scale != want:
            steps.append((f"set the tools to min-scale {want} ({'always warm' if want else 'scale to zero'})",
                          lambda say: cluster.set_min_scale(want, say), {}))
    url = None
    if target == "predictor" or config == "C3":
        label += "-" + ensure_predictor(spec, services, crd=knative and config == "C3")
        steps.append(("__predictor__", [], {}))
        url = PREDICTOR_URL
    shared = ["--seed", str(spec.get("seed") or 252)] + (["--tasks", " ".join(tasks)] if tasks else [])
    if engine == "locust":
        cmd = [PY, "-m", "locust", "-f", f"loadtest/locust_{target}.py", "--pulse-config", config, "--label", label,
               *shared, "--loglevel", "WARNING"]
        if spec.get("stages"):
            cmd += ["--stages", str(spec["stages"]).replace(" ", "")]
        if target == "predictor":
            cmd += ["--host", url]
        elif target == "episodes" and url:
            cmd += ["--predictor-url", url]
        web = bool(spec.get("interactive")) and not IN_CLUSTER
        cmd += ["--autostart", "--autoquit", "5", "--web-host", "127.0.0.1", "--web-port", "8089"] if web \
            else ["--headless", "--only-summary"]
        title = f"Locust load test: {target} {config}" + (" (live web UI: http://127.0.0.1:8089)" if web else "")
    else:
        cmd = [PY, "loadtest/async_load.py", target, "--config", config, "--label", label, *shared]
        if target == "predictor":
            cmd += ["--url", url, "--rates", *str(spec.get("rates") or "1 5 10 20").split(),
                    "--duration", str(spec.get("duration") or 60)]
        else:
            cmd += ["--burst", *str(spec.get("burst") or "1 5 10").split(), "--pause", str(spec.get("pause") or 0)]
        title = f"asyncio load test: {target} {config}"
    steps.append((title, cmd, env))
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
    if s.get("kind") == "load":                 # a load test (loadtest/): requests instead of tool calls
        return {"name": d.name, "path": rel(d), "where": d.parent.name, "kind": "load", "config": s.get("config"),
                "mode": f"load: {s.get('target')}", "target": s.get("target"), "engine": s.get("engine"),
                "predictor": (s.get("params") or {}).get("predictor"), "calls": s.get("requests"),
                "failures": s.get("failures"), "rps": s.get("rps"), "peak_load": s.get("peak_load"),
                "cold": None, "cold_share": s.get("cold_share"), "p50": lat.get("p50"), "p95": lat.get("p95"),
                "p99": lat.get("p99"), "mean": lat.get("mean"), "pod_seconds": None,
                "files": sorted(p.name for p in d.iterdir())}
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
    load = [json.loads((d / "summary.json").read_text(encoding="utf-8")).get("kind") == "load" for d in dirs]
    if any(load) and not all(load):
        raise ValueError("pick either experiment runs or load runs, not both")
    if all(load):                               # load runs: latency / throughput / failures against load
        base = LOAD_WRITE_DIR if LOAD_WRITE_DIR else dirs[0].parent
        base.mkdir(parents=True, exist_ok=True)
    script = "loadtest/plot_load.py" if all(load) else "test/test_knative/plot_compare.py"
    out = base / f"compare_{time.strftime('%Y-%m-%d_%H%M%S')}_ui_{len(dirs)}runs"
    p = subprocess.run([plot_python(), script, *map(str, dirs), "--out", str(out)],
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
