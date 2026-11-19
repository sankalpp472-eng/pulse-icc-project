"""
Regression tests for issue #36: request-supplied workflow_id values must not add Prometheus series. Every /predict
call used to create predictor_hints_total{workflow_id="<whatever the caller sent>"}, so /metrics grew without bound.
"""

import importlib
import re

import pytest

from pulse import paths


@pytest.fixture(scope="module")
def client():
    if not (paths.MODELS / "action_model.pkl").exists():
        pytest.skip("models/action_model.pkl is not in this checkout")
    mp = pytest.MonkeyPatch()
    mp.setenv("PREDICTOR", "markov")
    mp.setenv("CRD_MODE", "off")
    mp.delenv("ACTION_MODEL_URI", raising=False)
    mp.delenv("METRIC_WORKFLOWS", raising=False)
    import pulse.predictor.service as service
    service = importlib.reload(service)
    from fastapi.testclient import TestClient
    yield service, TestClient(service.app)
    mp.undo()


def hint_series(c):
    text = c.get("/metrics/").text
    return dict(re.findall(r'^predictor_hints_total\{workflow_id="([^"]*)"\} (\S+)$', text, re.M)), len(text)


def predict(c, **body):
    return c.post("/predict", json={"trajectory": ["get_user_details"], **body})


def test_distinct_workflow_ids_do_not_add_series(client):
    service, c = client
    assert predict(c, workflow_id="W1").status_code == 200
    _, size_before = hint_series(c)
    for i in range(500):                                   # the issue's attack, scaled down
        assert predict(c, workflow_id=f"atk{i}").status_code == 200
    series, size_after = hint_series(c)
    assert set(series) <= service.METRIC_WORKFLOWS | {"other"}
    assert float(series["other"]) == 500 and float(series["W1"]) == 1
    assert size_after - size_before < 200                  # only the counters' values changed


def test_known_workflows_keep_their_own_series(client):
    service, c = client
    for wf in ("W1", "W3"):
        assert predict(c, workflow_id=wf).json()["workflow_id"] == wf   # the response still echoes the id
    assert predict(c, domain="retail").status_code == 200              # no workflow id -> the domain is the label
    series, _ = hint_series(c)
    assert {"W1", "W3"} <= set(series)
    assert service.metric_workflow("retail") == "retail"
    assert service.metric_workflow(None) == "other"


@pytest.mark.parametrize("field,size", [("workflow_id", 65), ("episode_id", 129), ("domain", 33)])
def test_overlong_ids_are_refused(client, field, size):
    _, c = client
    assert predict(c, **{field: "x" * size}).status_code == 422
