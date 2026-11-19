"""
Regression tests for issue #19: after switching "Tools run on" from the simulated cluster to Knative, a C3 episode
must prewarm the cluster's tools (the predictor writes PrewarmRequests and the controller runs), or be refused.
It used to run with the predictor still in "CRDs off" mode: every call cold, yet shown as "prewarmed".

No cluster needed: the processes the UI would start are recorded instead of started.
"""

import pytest
from fastapi import HTTPException

import pulse.ui.backend as backend


class FakeProc:
    def __init__(self, name, cmd, env, info):
        self.name, self.cmd, self.env, self.info, self.up = name, cmd, env, info, True

    def alive(self):
        return self.up

    def stop(self):
        self.up = False


@pytest.fixture
def ui(monkeypatch):
    """The UI's cluster + services, with fake processes."""
    import pulse.ui.server as server
    monkeypatch.setattr(backend, "Proc", FakeProc)
    monkeypatch.setattr(backend.Services, "status", lambda self: {})
    monkeypatch.setattr(backend.Services, "stop",
                        lambda self, name=None: [p.stop() for n, p in self.procs.items() if name in (None, n)] and {})
    monkeypatch.setattr(backend.Services, "start_controller",
                        lambda self: self.procs.__setitem__("controller", FakeProc("controller", [], {}, {})) or {})
    monkeypatch.setattr(backend.Services, "predictor_ready", lambda self: True)
    monkeypatch.setattr(backend.Services, "wait_predictor", lambda self, timeout=90: True)

    def detect(self):                       # a reachable kind cluster with the airline tools
        self.context, self.reachable, self.detail = "kind-pulse-kn", True, "14 airline tools deployed"
        return self.info()

    monkeypatch.setattr(backend.Cluster, "detect", detect)
    server.cluster.mode, server.cluster.domain = "sim", "airline"
    server.services.procs.clear()
    server.S["session"] = server.S["run"] = None
    yield server
    server.cluster.mode = "sim"
    server.services.procs.clear()


def test_switch_to_knative_turns_prewarming_on(ui):
    svc = ui.services
    svc.start_predictor("lightgbm", crd=False, policy={"policy": "topk", "top_k": 3})   # how the UI starts it (sim)
    assert svc.procs["predictor"].info["crd"] == "off"

    ui.set_cluster({"mode": "knative"})                           # the issue's step 3: no click on Start

    pred = svc.procs["predictor"]
    assert pred.alive() and pred.info["crd"] == "k8s" and pred.env["CRD_MODE"] == "k8s"
    assert (pred.info["model"], pred.info["policy"]["policy"], pred.info["policy"]["top_k"]) == \
        ("lightgbm", "topk", 3)                                   # same predictor, same settings
    assert svc.controller_running()
    assert svc.prewarm_problem() is None


def test_switch_back_to_sim_turns_cluster_writes_off(ui):
    svc = ui.services
    ui.set_cluster({"mode": "knative"})
    svc.start_predictor("markov", crd=True)
    ui.set_cluster({"mode": "sim"})
    assert svc.procs["predictor"].info["crd"] == "off"
    assert svc.procs["predictor"].info["model"] == "markov"


def test_c3_episode_refused_when_prewarming_is_inactive(ui, monkeypatch):
    svc = ui.services
    ui.cluster.mode = "knative"                                   # e.g. a predictor that can't follow the switch
    svc.start_predictor("lightgbm", crd=False)
    monkeypatch.setattr(backend.Services, "match_cluster", lambda self: False)
    with pytest.raises(HTTPException) as e:
        ui.session_start({"kind": "replay", "predictor": "lightgbm", "task_id": "30"})
    assert e.value.status_code == 400 and "Prewarming is not active" in e.value.detail
    assert ui.S["session"] is None


def test_c3_episode_after_switch_starts_with_prewarming(ui, monkeypatch):
    svc = ui.services
    svc.start_predictor("lightgbm", crd=False)
    ui.set_cluster({"mode": "knative"})
    started = {}
    monkeypatch.setattr(ui, "Session", lambda cfg, *a, **k: started.setdefault("s", type(
        "S", (), {"start": lambda self: None, "cfg": cfg})()))
    ui.session_start({"kind": "replay", "predictor": "lightgbm", "task_id": "30"})
    assert "s" in started
    assert svc.procs["predictor"].info["crd"] == "k8s" and svc.controller_running()
    ui.S["session"] = None
