"""
Regression tests for issues #23 and #24: /predict rejects (422, with the reason) a domain the predictor does not
serve, trajectory names that are not the domain's tools (RESPOND / END included), and oversized fields, instead of
answering with confident predictions for another domain or for input the model never saw.

No cluster or LLM needed: the LightGBM predictor (models/action_model.pkl, airline) runs in-process.
"""

import pytest
from fastapi.testclient import TestClient

from pulse.predictor import client as hint_client
from pulse.predictor.contract import MAX_NAME, MAX_TRAJECTORY, domain_tools


@pytest.fixture(scope="module")
def api():
    import pulse.predictor.service as service
    assert service.PRED.domains == ("airline",)
    return TestClient(service.app)


def predict(api, **kw):
    return api.post("/predict", json={"episode_id": "rt", **kw})


def test_valid_hint_still_answers(api):
    r = predict(api, domain="airline", trajectory=["get_user_details"])
    assert r.status_code == 200
    names = {a["tool"] for a in r.json()["ranked"]}
    assert names <= domain_tools("airline") | {"RESPOND", "END"}


@pytest.mark.parametrize("start", [{}, {"current_tool": "START"}, {"trajectory": []}])
def test_first_step_still_answers(api, start):
    assert predict(api, domain="airline", **start).status_code == 200


# ---------------------------------------------------------------------- #23: the domain
@pytest.mark.parametrize("domain", ["retail", "zzz"])
def test_unserved_domain_is_rejected(api, domain):
    r = predict(api, domain=domain, trajectory=["get_user_details"])
    assert r.status_code == 422
    assert f"domain '{domain}' is not served" in r.json()["detail"] and "airline" in r.json()["detail"]


# ---------------------------------------------------------------------- #24: the tool names
@pytest.mark.parametrize("traj, named", [(["no_such_tool"], "'no_such_tool'"),
                                         (["RESPOND", "END"], "'RESPOND', 'END'"),
                                         (["get_user_details", "get_order_details"], "'get_order_details'")])
def test_unknown_tools_are_rejected(api, traj, named):
    r = predict(api, domain="airline", trajectory=traj)
    assert r.status_code == 422 and named in r.json()["detail"]


def test_unknown_current_tool_is_rejected(api):
    assert predict(api, domain="airline", current_tool="bogus").status_code == 422


def test_oversized_name_is_rejected_without_echo(api):
    r = predict(api, domain="airline", trajectory=["x" * 100_000])
    assert r.status_code == 422
    assert len(r.content) < 2000                    # the 100,000 characters are not sent back
    assert r.json()["detail"][0]["loc"] == ["body", "trajectory", 0]


def test_too_long_trajectory_is_rejected(api):
    r = predict(api, domain="airline", trajectory=["get_user_details"] * (MAX_TRAJECTORY + 1))
    assert r.status_code == 422


def test_max_name_fits_every_tool():
    assert max(len(t) for d in ("airline", "retail") for t in domain_tools(d)) <= MAX_NAME


# ---------------------------------------------------------------------- the senders never trip it
def test_hint_drops_names_that_are_not_tools(api):
    hint = hint_client.make_hint({"trajectory"}, "rt", ["get_user_details", "made_up_tool", "think"],
                                 domain="airline")
    assert hint["trajectory"] == ["get_user_details", "think"] and hint["current_tool"] == "think"
    assert api.post("/predict", json=hint).status_code == 200
