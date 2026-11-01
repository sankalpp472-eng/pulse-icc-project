"""
Pulse demo UI: run live and replay episodes, pick the predictor, watch the tools' pods scale and the
prewarms land, launch experiment runs and compare their results.

    agent/.venv/Scripts/python ui/server.py            # from the repo root, then open http://127.0.0.1:8765

Listens on 127.0.0.1 only. See ui/README.md.
"""

import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

import backend  # noqa: E402
from session import Session  # noqa: E402
from sim_cluster import SimCluster  # noqa: E402

app = FastAPI(title="Pulse demo UI")
cluster = backend.Cluster()
services = backend.Services(cluster)
sim = SimCluster(backend.TOOL_NAMES)
S: dict = {"session": None, "run": None, "history": []}


def busy() -> str | None:
    s, r = S["session"], S["run"]
    if s and s.is_alive():
        return "an interactive episode is running"
    if r and r.state == "running":
        return "an experiment run is in progress"
    return None


# ---------------------------------------------------------------- state
@app.get("/api/state")
def state(session_after: int = 0, run_after: int = 0, events_after: int = 0):
    s, r = S["session"], S["run"]
    if s and not s.is_alive() and s.events and s.events[-1]["type"] == "end" and not getattr(s, "_recorded", False):
        s._recorded = True
        S["history"].insert(0, s.events[-1]["summary"])
    pods = sim.snapshot() if cluster.mode == "sim" else cluster.pods
    events = sim.events if cluster.mode == "sim" else cluster.events
    return {"cluster": cluster.info(), "services": services.status(),
            "pods": pods, "pod_seconds": round(sim.live_pod_seconds(), 1) if cluster.mode == "sim" else None,
            "sim_min_scale": sim.min_scale,
            "events": events[events_after:], "events_n": len(events),
            "session": s.view(session_after) if s else None,
            "run": r.view(run_after) if r else None,
            "history": S["history"][:20], "now": time.time()}


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


@app.post("/api/sim/min_scale")
def sim_min_scale(body: dict):
    sim.set_min_scale(int(body.get("min_scale", 0)))
    return {"min_scale": sim.min_scale}


# ---------------------------------------------------------------- services
class PredictorIn(BaseModel):
    model: str = "gbt"
    theta: float = 0.3


@app.post("/api/predictor/start")
def predictor_start(p: PredictorIn):
    crd = cluster.mode == "knative"
    st = services.start_predictor(p.model, p.theta, crd)
    if crd and not ("controller" in services.procs and services.procs["controller"].alive()):
        st = services.start_controller()
    return st


@app.post("/api/predictor/stop")
def predictor_stop():
    services.stop("predictor")
    return services.stop("controller")


# ---------------------------------------------------------------- interactive episodes
@app.post("/api/session/start")
def session_start(cfg: dict):
    if busy():
        raise HTTPException(409, busy())
    if cfg.get("kind") not in ("live", "replay"):
        raise HTTPException(400, "kind must be live or replay")
    pred = None
    if cfg.get("predictor"):
        if not services.wait_predictor(timeout=1 if services.predictor_ready() else 90):
            raise HTTPException(400, "Start the predictor first (Predictor card), or choose 'none (C0)'")
        pred = backend.PREDICTOR_URL
        cfg["predictor"] = services.procs["predictor"].info["model"]
        cfg["theta"] = services.procs["predictor"].info["theta"]
    if cluster.mode == "sim" and cfg.get("fresh", True):
        sim.set_min_scale(sim.min_scale)
    s = Session(cfg, cluster, sim if cluster.mode == "sim" else None, pred)
    S["session"] = s
    s.start()
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


@app.get("/api/tasks")
def tasks():
    sys.path.insert(0, str(ROOT / "agent"))
    from tau_airline import load_tasks
    import json
    traces = [json.loads(l) for l in open(ROOT / "test/test_knative/data/w1_tau_traces.jsonl", encoding="utf-8")]
    recorded: dict[int, int] = {}
    fan: set[int] = set()
    for t in traces:
        if t["calls"]:
            recorded[t["task_id"]] = recorded.get(t["task_id"], 0) + 1
            if any(a["tool"] == b["tool"] and b["user_turns_before"] == 0
                   for a, b in zip(t["calls"], t["calls"][1:])):
                fan.add(t["task_id"])
    return [{"id": t["id"], "instruction": t["instruction"], "gold": [a["name"] for a in t["actions"]],
             "recorded": recorded.get(t["id"], 0), "fanout": t["id"] in fan,
             "heldout": t["id"] in (6, 20, 22, 26, 30)} for t in load_tasks()]


# ---------------------------------------------------------------- experiment runs
@app.post("/api/run/start")
def run_start(spec: dict):
    if busy():
        raise HTTPException(409, busy())
    if cluster.mode != "knative":
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


@app.get("/files/{path:path}")
def files(path: str):
    p = (ROOT / path).resolve()
    allowed = [d.resolve() for d in backend.RESULT_DIRS.values()]
    if not any(a in p.parents for a in allowed) or not p.is_file() or p.suffix not in (".png", ".json", ".csv", ".jsonl"):
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.on_event("startup")
def _startup():
    # load the main predictor right away: importing LightGBM + the models takes up to a minute on a laptop
    if os.getenv("UI_AUTOSTART_PREDICTOR", "1") != "0":
        services.start_predictor("gbt", 0.3, crd=False)


@app.on_event("shutdown")
def _shutdown():
    if S["session"]:
        S["session"].stop()
    if S["run"]:
        S["run"].stop()
    services.stop()


if __name__ == "__main__":
    port = int(os.getenv("UI_PORT", "8765"))
    print(f"Pulse demo UI: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
