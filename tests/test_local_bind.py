"""
Regression tests for issue #16: on a laptop / VM the services Pulse starts listen on this machine only (127.0.0.1),
like the UI itself, and the predictor's /healthz does not show the machine's absolute file paths.
"""

import os
from types import SimpleNamespace

import pytest

from pulse import net, paths


@pytest.fixture
def laptop(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("PULSE_BIND", raising=False)


# ---------------------------------------------------------------------- which address
def test_laptop_binds_to_localhost(laptop):
    assert net.bind_host() == "127.0.0.1"


def test_pod_binds_to_all_interfaces(laptop, monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.100.0.1")      # set by Kubernetes in every pod
    assert net.bind_host() == "0.0.0.0"


def test_pulse_bind_overrides(laptop, monkeypatch):
    monkeypatch.setenv("PULSE_BIND", "172.17.0.1")
    assert net.bind_host() == "172.17.0.1"


# ---------------------------------------------------------------------- the services use it
def test_ui_starts_the_predictor_on_localhost(laptop, monkeypatch):
    import pulse.ui.backend as backend
    started = {}
    monkeypatch.setattr(backend, "Proc", lambda name, cmd, env, info: started.setdefault(name, cmd))
    monkeypatch.setattr(backend.Services, "_health_loop", lambda self: None)
    monkeypatch.setattr(backend.Services, "status", lambda self: {})
    svc = backend.Services(SimpleNamespace(mode="sim", domain="airline", context=None))
    svc.start_predictor("markov", theta=0.3, crd=False)
    cmd = started["predictor"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_run_metrics_listen_on_localhost(laptop, monkeypatch):
    import pulse.experiments.metrics as metrics
    seen = {}
    monkeypatch.setattr(metrics, "start_http_server", lambda port, addr="0.0.0.0": seen.update(port=port, addr=addr))
    metrics.RunMetrics(9105, "test-run", "replay", ["get_user_details"],
                       {"config": "C0", "window": "60", "agent_model": "-", "user_model": "-", "database": "-"})
    assert seen == {"port": 9105, "addr": "127.0.0.1"}


# ---------------------------------------------------------------------- no absolute paths on /healthz
def test_shown_path_is_never_absolute(tmp_path):
    assert paths.shown(paths.MODELS / "action_model.pkl") == "models/action_model.pkl"
    assert paths.shown(paths.ROOT / "pulse" / "x.pkl") == "pulse/x.pkl"
    assert paths.shown(tmp_path / "elsewhere.pkl") == "elsewhere.pkl"      # outside the repo: the name only


def test_predictor_healthz_has_no_absolute_path(monkeypatch):
    if not (paths.MODELS / "action_model.pkl").exists():
        pytest.skip("models/action_model.pkl is not in this checkout")
    monkeypatch.setenv("PREDICTOR", "markov")
    monkeypatch.setenv("CRD_MODE", "off")
    monkeypatch.delenv("ACTION_MODEL_URI", raising=False)
    import importlib
    import pulse.predictor.service as service
    service = importlib.reload(service)
    from fastapi.testclient import TestClient
    body = TestClient(service.app).get("/healthz").json()
    text = str(body)
    assert body.get("model_file") == "models/action_model.pkl"
    assert str(paths.ROOT.resolve()) not in text and os.path.expanduser("~") not in text
