"""
Pulse demo UI: run live and replay episodes, pick the predictor, watch the tools' pods scale and the
prewarms land, launch experiment runs and compare their results.

    python -m pulse.ui.server            # from the repo root, then open http://127.0.0.1:8765

Runs on the laptop and listens on 127.0.0.1 only (it can start runs and change the cluster, and has no login).
"""

import os
import time
from pathlib import Path

import urllib.error
import urllib.request

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from pulse import paths

HERE = Path(__file__).resolve().parent
ROOT = paths.ROOT
os.chdir(ROOT)

import pulse.ui.backend as backend  # noqa: E402
from pulse.ui import setup as setup_actions  # noqa: E402
from pulse.ui.session import Session  # noqa: E402
from pulse.ui.sim_cluster import SimCluster  # noqa: E402

app = FastAPI(title="Pulse demo UI")
# which UI code this process runs: the newest modification time of the server's own files when it started
_CODE_FILES = [HERE / f for f in ("server.py", "backend.py", "session.py", "sim_cluster.py")]
LOADED_CODE = max(int(f.stat().st_mtime) for f in _CODE_FILES)


@app.get("/api/version")
def version():
    """The page checks this: a server older than the code on disk (started before an update) ignores new settings,
    e.g. the Top-k policy, so the page asks for a restart."""
    on_disk = max(int(f.stat().st_mtime) for f in _CODE_FILES if f.exists())
    return {"loaded": LOADED_CODE, "on_disk": on_disk, "restart_needed": on_disk > LOADED_CODE}


@app.middleware("http")
async def no_stale_page(request, call_next):
    """The page's own files are small: make browsers check for a new version on every load, so an updated UI
    (new app.js next to a new index.html) never runs with a cached old script."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp
cluster = backend.Cluster()
services = backend.Services(cluster)
monitoring = backend.Monitoring(cluster)
deployment = backend.Deployment(cluster)
sim = SimCluster(backend.TOOL_NAMES)
S: dict = {"session": None, "run": None, "history": []}
RECORDINGS = backend.STATE / "recordings"      # live episodes recorded by the UI, replayable exactly (kind=recorded)


def busy() -> str | None:
    s, r = S["session"], S["run"]
    if s and s.is_alive():
        return "an interactive episode is running"
    if r and r.state == "running":
        return "a job is running (experiment, setup or training): wait for it or stop it"
    return None


# ---------------------------------------------------------------- state
@app.get("/api/state")
def state(session_after: int = 0, run_after: int = 0, events_after: int = 0):
    s, r = S["session"], S["run"]
    if s and not s.is_alive() and s.events and s.events[-1]["type"] == "end" and not getattr(s, "_recorded", False):
        s._recorded = True
        S["history"].insert(0, s.events[-1]["summary"])
    if "bgload" in services.procs and not (s and s.is_alive()):     # background load ends with its episode
        services.stop("bgload")
    pods = sim.snapshot() if cluster.mode == "sim" else cluster.pods
    events = sim.events if cluster.mode == "sim" else cluster.events
    return {"cluster": cluster.info(), "services": services.status(),
            "pods": pods, "pod_seconds": round(sim.live_pod_seconds(), 1) if cluster.mode == "sim" else None,
            "sim_min_scale": sim.min_scale,
            "events": events[events_after:], "events_n": len(events),
            "session": s.view(session_after) if s else None,
            "run": r.view(run_after) if r else None,
            "history": S["history"][:20], "monitoring": monitoring.info(), "deployment": deployment.info(),
            "now": time.time()}


@app.post("/api/cluster")
def set_cluster(body: dict):
    mode = body.get("mode", "sim")
    if busy():
        raise HTTPException(409, f"Can't switch now: {busy()}")
    if mode == "knative":
        info = cluster.detect()
        if not info["reachable"]:
            raise HTTPException(400, f"Knative cluster not usable: {info['detail']}")
    services.stop("controller")
    cluster.mode = mode
    return cluster.info()


@app.post("/api/cluster/detect")
def detect():
    return cluster.detect()


@app.post("/api/domain")
def set_domain(body: dict):
    """Which tau-bench tools experiment runs use: airline (14 tools) or retail (16, deployed by Setup). The predictor
    serves one domain, so it stops (runs start the right one)."""
    domain = body.get("domain")
    if domain not in backend.DOMAINS:
        raise HTTPException(400, f"domain must be one of {sorted(backend.DOMAINS)}")
    if busy():
        raise HTTPException(409, f"Can't switch now: {busy()}")
    if domain != cluster.domain:
        services.stop("predictor")
        cluster.domain = domain
        cluster.pods = []
    return cluster.detect() if cluster.mode == "knative" else cluster.info()


@app.post("/api/sim/min_scale")
def sim_min_scale(body: dict):
    sim.set_min_scale(int(body.get("min_scale", 0)))
    return {"min_scale": sim.min_scale}


# ---------------------------------------------------------------- services
class PredictorIn(BaseModel):
    model: str = "lightgbm"         # lightgbm | markov | speculator | pulse_v3
    theta: float = 0.3
    policy: dict | None = None      # {"policy": "theta"|"topk", "top_k", "p_min", "lookahead", "theta_next"}


@app.post("/api/predictor/start")
def predictor_start(p: PredictorIn):
    crd = cluster.mode == "knative"
    try:
        st = services.start_predictor(p.model, p.theta, crd, p.policy)
    except ValueError as e:                 # e.g. the speculator with the retail domain
        raise HTTPException(400, str(e))
    if crd and not services.controller_running():
        st = services.start_controller()
    return st


@app.get("/api/eks_predictor")
def eks_predictor_info():
    """Which predictor the cluster's deploy/pulse-predictor runs (the live demo's 'EKS predictor' option)."""
    return services.eks_predictor_info()


@app.post("/api/predictor/stop")
def predictor_stop():
    services.stop("predictor")
    return services.stop("controller")


# ---------------------------------------------------------------- interactive episodes
@app.post("/api/session/start")
def session_start(cfg: dict):
    if busy():
        raise HTTPException(409, busy())
    if cfg.get("kind") not in ("live", "replay", "recorded"):
        raise HTTPException(400, "kind must be live, replay or recorded")
    if cluster.mode == "knative" and cluster.domain != "airline":
        raise HTTPException(400, "The live demo's agent works on the airline tools: switch the domain to airline (top bar)")
    if cfg["kind"] == "recorded" and not (RECORDINGS / f"{cfg.get('recording')}.json").exists():
        raise HTTPException(400, "pick a recorded live run (record one first: Live agent, then Start episode)")
    pred = None
    if cfg.get("predictor") and cfg.get("predictor_where") == "cluster":     # deploy/pulse-predictor in the cluster
        try:
            eks = services.eks_predictor()
        except ValueError as e:
            raise HTTPException(400, str(e))
        pred = eks["url"]
        cfg["predictor"], cfg["theta"] = f"{eks['name']} (EKS)", eks["theta"]
        cfg["policy_label"] = backend.policy_label(eks["policy"], eks["theta"])
    elif cfg.get("predictor"):
        if not services.wait_predictor(timeout=1 if services.predictor_ready() else 90):
            raise HTTPException(400, "Start the predictor first (Predictor card), or choose 'none (C0)'")
        pred = backend.PREDICTOR_URL
        info = services.procs["predictor"].info
        cfg["predictor"], cfg["theta"] = info["model"], info["theta"]
        cfg["policy_label"] = backend.policy_label(info.get("policy"), info["theta"])
        if cluster.mode == "knative" and info.get("crd") == "k8s" and not services.controller_running():
            services.start_controller()        # stopped by an episode with the EKS predictor
    if cluster.mode == "sim" and cfg.get("fresh", True):
        sim.set_min_scale(sim.min_scale)
    agents = int(cfg.get("bg_agents") or 0)
    if agents and cluster.mode != "knative":
        raise HTTPException(400, "Background load needs the real cluster: switch to Knative (top bar), or set it to 0")
    s = Session(cfg, cluster, sim if cluster.mode == "sim" else None, pred, record_dir=RECORDINGS)
    S["session"] = s
    s.start()
    if agents:                     # N more agents on the same tools while this episode runs (loadtest/)
        services.start_bgload(agents, pred)
    return {"ok": True}


@app.post("/api/session/reply")
def session_reply(body: dict):
    s = S["session"]
    if not s or not s.is_alive():
        raise HTTPException(400, "no episode is running")
    s.inbox.put(str(body.get("text", "")).strip() or "###STOP###")
    return {"ok": True}


@app.post("/api/session/stop")
def session_stop():
    if S["session"]:
        S["session"].stop()
    return {"ok": True}


@app.post("/api/sim/reset")
def sim_reset():
    """Every simulated tool back to zero pods (a fresh cold start for the next demo)."""
    if busy():
        raise HTTPException(409, busy())
    global sim
    old = sim.min_scale
    sim = SimCluster(backend.TOOL_NAMES, min_scale=old)
    return {"ok": True}


@app.post("/api/demo/prepare")
def demo_prepare():
    """Before each run of a C0-vs-C3 demo: same starting point for both.
    Simulated cluster: every tool back to zero pods. Knative: restore the shared airline database from its seed copy
    (as the runners do); the tools go cold by themselves (idle window), the page waits until no tool pod is left."""
    if busy():
        raise HTTPException(409, busy())
    global sim
    if cluster.mode == "sim":
        sim = SimCluster(backend.TOOL_NAMES, min_scale=sim.min_scale)
        return {"pods": 0, "db": "fresh copy per episode"}
    code, out = backend._run(["kubectl", "get", "deploy", "redis", "-o", "name"], 10)
    db = "no shared database"
    if code == 0:
        for t in ("reservations", "users"):
            c, o = backend._run(["kubectl", "exec", "deploy/redis", "--", "redis-cli", "COPY", f"seed:{t}", t, "REPLACE"], 20)
            if c != 0:
                raise HTTPException(500, f"could not reset the database: {o[:200]}")
        db = "reset from seed"
    return {"pods": sum(p["ready"] + p["starting"] for p in cluster.pods), "db": db}


@app.get("/api/recordings")
def recordings():
    """The recorded live runs, newest first (without their steps)."""
    import json
    out = []
    for f in sorted(RECORDINGS.glob("live-*.json"), reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        steps = r.get("steps", [])
        out.append({k: r.get(k) for k in ("id", "created", "task_id", "agent_model", "user", "config", "policy",
                                          "cluster", "live_e2e_s", "live_tool_wait_s", "live_cold", "live_calls", "state")}
                   | {"tool_steps": sum(s.get("kind") == "tool" for s in steps), "steps": len(steps)})
    return out


_TASKS: list | None = None


@app.get("/api/tasks")
def tasks():
    """The 50 airline tasks, with every recorded tau-bench run of each: calls, tool steps, customer turns and its
    fan-out (the widest group of calls the replay fires at once, w1_replay.fanout_groups). Computed once."""
    global _TASKS
    if _TASKS is None:
        from pulse.agent.tau_airline import load_tasks
        import pulse.experiments.replay as w1_replay
        runs: dict[int, list] = {}
        for t in w1_replay.load_traces(None):            # same order as the replay's "recorded run #"
            if t["calls"]:
                groups = w1_replay.fanout_groups(t)
                runs.setdefault(t["task_id"], []).append({
                    "calls": len(t["calls"]), "tool_steps": len(groups),
                    "turns": sum(c["user_turns_before"] for c in t["calls"]),
                    "fanout": max(len(g) for g in groups), "model": t.get("source_model")})
        _TASKS = [{"id": t["id"], "instruction": t["instruction"], "gold": [a["name"] for a in t["actions"]],
                   "recorded": len(runs.get(t["id"], [])), "runs": runs.get(t["id"], []),
                   "fanout": any(r["fanout"] > 1 for r in runs.get(t["id"], [])),
                   "plain_runs": sum(r["fanout"] == 1 for r in runs.get(t["id"], [])),
                   "heldout": t["id"] in (6, 20, 22, 26, 30)} for t in load_tasks()]
    return _TASKS


# ---------------------------------------------------------------- experiment runs
@app.post("/api/run/start")
def run_start(spec: dict):
    if busy():
        raise HTTPException(409, busy())
    predictor_load = spec.get("mode") == "load" and spec.get("target") == "predictor"   # needs no cluster
    if cluster.mode != "knative" and not predictor_load:
        raise HTTPException(400, "Experiment runs need a real cluster: switch to Knative (top bar) first")
    try:
        steps = backend.build_run(spec, cluster, services)
    except ValueError as e:
        raise HTTPException(400, str(e))
    S["run"] = backend.BatchRun(spec, steps, services)
    return {"ok": True, "steps": [s[0] for s in steps]}


@app.post("/api/run/stop")
def run_stop():
    if S["run"]:
        S["run"].stop()
    return {"ok": True}


# ---------------------------------------------------------------- models (the S3 registry: pulse_v3 | action)
@app.get("/api/models")
def models(family: str = "pulse_v3"):
    try:
        return backend.models_list(family)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/models/use")
def models_use(body: dict):
    """Serve one registry version (keeps the card's θ / policy; CRDs if in Knative mode). action = lightgbm + markov:
    keeps the running one of the two, else lightgbm."""
    v = str(body.get("version") or "")
    if not v:
        raise HTTPException(400, "which version?")
    try:
        preds = backend.family(body.get("family"))["predictors"]
    except ValueError as e:
        raise HTTPException(400, str(e))
    cur = services.procs.get("predictor")
    info = cur.info if cur else {}
    model = info.get("model") if info.get("model") in preds else preds[0]
    try:
        return services.start_predictor(model, float(info.get("theta", 0.3)), cluster.mode == "knative",
                                        info.get("policy"), model_version=v)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/models/promote")
def models_promote(body: dict):
    try:
        backend.models_promote(str(body.get("version") or ""), body.get("family"))
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return backend.models_list(body.get("family"))


@app.post("/api/models/pin")
def models_pin(body: dict):
    """GitOps: write the version into pulse-config in this checkout (no commit; review it, then push / PR)."""
    try:
        return backend.models_pin_in_git(str(body.get("version") or ""), body.get("family"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/models/retrain")
def models_retrain(spec: dict):
    """The family's training here, or its training Job in the cluster; the log streams in the Experiments tab."""
    if busy():
        raise HTTPException(409, busy())
    try:
        steps = backend.build_retrain(spec)
    except ValueError as e:
        raise HTTPException(400, str(e))
    S["run"] = backend.BatchRun({"mode": "retrain", **spec}, steps, services)
    return {"ok": True}


# ---------------------------------------------------------------- setup (data, models, keys, clusters, cloud)
@app.get("/api/setup")
def setup_status():
    """What is installed, downloaded, trained and running (setup.status: tools on PATH, AWS login, clusters)."""
    return {**setup_actions.status(cluster),
            "actions": {k: {"title": t, "confirm": c} for k, (t, c) in setup_actions.ACTIONS.items()}}


@app.post("/api/setup/run")
def setup_run(body: dict):
    """One Setup action (setup.ACTIONS) in the job runner; its log streams like an experiment run's."""
    if busy():
        raise HTTPException(409, busy())
    name = str(body.get("action") or "")
    try:
        steps = setup_actions.build(name, body.get("params") or {}, cluster)
    except ValueError as e:
        raise HTTPException(400, str(e))
    S["run"] = backend.BatchRun({"mode": "setup", "action": name}, steps, services)
    return {"ok": True}


@app.post("/api/setup/env")
def setup_env(body: dict):
    """Write LLM keys / settings into .env (groq_keys: new Groq keys, any number). The answer says which are set,
    never their values."""
    try:
        return setup_actions.save_env(body.get("values") or {}, body.get("groq_keys") or [])
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- monitoring (Prometheus + Grafana)
@app.post("/api/monitoring/install")
def monitoring_install():
    """Run infra/monitoring/install.sh like an experiment run (its log streams in the Experiments tab)."""
    if busy():
        raise HTTPException(409, busy())
    if cluster.mode != "knative":
        raise HTTPException(400, "Monitoring is installed into the real cluster: switch to Knative (top bar) first")
    if not backend.bash():
        raise HTTPException(400, "bash not found (Git Bash on Windows): needed for infra/monitoring/install.sh")
    S["run"] = backend.BatchRun({"mode": "monitoring install"},
                                [("install Prometheus + Grafana (infra/monitoring/install.sh)",
                                  [backend.bash(), "infra/monitoring/install.sh"], {})], services)
    return {"ok": True}


@app.get("/api/monitoring/link")
def monitoring_link(path: str, dashboard: str = "pulse-overview"):
    """Grafana for exactly one result run: its start-to-end window (from calls.csv), a minute either side."""
    try:
        t0, t1 = backend.run_window(path)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    return {"dashboard": dashboard, "from": int((t0 - 60) * 1000), "to": int((t1 + 60) * 1000)}


_GRAFANA_HEADERS = ("content-type", "accept", "x-grafana-org-id", "x-dashboard-uid", "x-panel-id",
                    "x-datasource-uid", "x-query-group-id", "x-grafana-device-id", "x-plugin-id")


@app.api_route("/grafana/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def grafana(path: str, request: Request):
    """Grafana through the UI (it builds its links under /grafana/, values.yaml): the page needs no second tunnel."""
    if not monitoring.reachable:
        raise HTTPException(503, f"Grafana not available: {monitoring.detail}")
    url = f"{monitoring.base()}/{path}" + (f"?{request.url.query}" if request.url.query else "")
    body = await request.body()
    req = urllib.request.Request(url, data=body or None, method=request.method,
                                 headers={k: v for k, v in request.headers.items() if k.lower() in _GRAFANA_HEADERS})

    def fetch():
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    status, data, headers = await run_in_threadpool(fetch)
    keep = {k: v for k, v in headers.items() if k.lower() in ("content-type", "cache-control", "etag", "last-modified")}
    return Response(data, status_code=status, headers=keep)


# ---------------------------------------------------------------- results
@app.get("/api/results")
def results():
    return backend.list_results()


@app.post("/api/results/plot")
def results_plot(body: dict):
    paths = body.get("paths") or []
    if len(paths) < 1:
        raise HTTPException(400, "pick at least one run")
    try:
        return backend.make_plots(paths)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/results/zip")
def results_zip(path: str):
    try:
        data, name = backend.zip_folder(path)
    except ValueError:
        raise HTTPException(404)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/files/{path:path}")
def files(path: str):
    try:
        p = backend.result_path(path)
    except ValueError:
        raise HTTPException(404)
    if not p.is_file() or p.suffix not in (".png", ".json", ".csv", ".jsonl"):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/")
def index():
    """The page, with its script and stylesheet addresses stamped with their file time: a changed file gets a new
    address, so no browser can keep running an old cached copy next to a new page."""
    from fastapi.responses import HTMLResponse
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8")
    for f in ("app.js", "style.css"):
        html = html.replace(f"/static/{f}\"", f"/static/{f}?v={int((HERE / 'static' / f).stat().st_mtime)}\"")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.on_event("startup")
def _startup():
    # load the main predictor right away: importing LightGBM + the models takes up to a minute on a laptop
    if os.getenv("UI_AUTOSTART_PREDICTOR", "1") != "0":
        services.start_predictor(os.getenv("UI_PREDICTOR", "lightgbm"), 0.3, crd=cluster.mode == "knative")


@app.on_event("shutdown")
def _shutdown():
    if S["session"]:
        S["session"].stop()
    if S["run"]:
        S["run"].stop()
    services.stop()
    monitoring._stop_pf()
    deployment._stop_pf()


if __name__ == "__main__":
    port = int(os.getenv("UI_PORT", "8765"))
    print(f"Pulse demo UI: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
