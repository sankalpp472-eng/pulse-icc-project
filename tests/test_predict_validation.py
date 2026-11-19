"""
Regression tests for issues #23 and #24: /predict rejects (422, with the reason) a domain the predictor does not
serve; for trajectory names that are not the domain's tools (RESPOND / END included) it answers with every
probability 0, nothing prewarmed, and the names in unknown_tools, instead of confident predictions for input the
model never saw. No size limits on the request (pulse_v3 reads the conversation).

No cluster or LLM needed: the LightGBM predictor (models/action_model.pkl, airline) runs in-process.
"""

import pytest
from fastapi.testclient import TestClient

from pulse.predictor import client as hint_client
from pulse.predictor.contract import domain_tools


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
def assert_all_zero(r, named):
    assert r.status_code == 200
    d = r.json()
    assert d["ranked"] and all(a["p"] == 0 for a in d["ranked"])
    assert {a["tool"] for a in d["ranked"]} == domain_tools("airline") | {"RESPOND", "END"}
    assert d["prewarm"] == [] and d["unknown_tools"] == named
    assert "every probability is 0" in d["note"]


@pytest.mark.parametrize("traj, named", [(["no_such_tool"], ["no_such_tool"]),
                                         (["RESPOND", "END"], ["RESPOND", "END"]),
                                         (["get_user_details", "get_order_details"], ["get_order_details"])])
def test_unknown_tools_give_probability_zero(api, traj, named):
    assert_all_zero(predict(api, domain="airline", trajectory=traj), named)


def test_unknown_current_tool_gives_probability_zero(api):
    assert_all_zero(predict(api, domain="airline", current_tool="bogus"), ["bogus"])


def test_huge_unknown_name_is_not_echoed_back(api):
    r = predict(api, domain="airline", trajectory=["x" * 100_000])
    assert_all_zero(r, ["x" * 64 + "... (100000 characters)"])
    assert len(r.content) < 5000                    # the 100,000 characters are not sent back


# ---------------------------------------------------------------------- no size limits (pulse_v3 reads the conversation)
def test_long_input_is_accepted(api):
    long_msg = {"role": "user", "content": "please help " * 5000}
    r = predict(api, domain="airline", trajectory=["get_user_details"] * 200, history=[long_msg] * 300,
                episode_id="e" * 1000)
    assert r.status_code == 200 and r.json()["unknown_tools"] == []


# ---------------------------------------------------------------------- the senders never trip it
def test_hint_drops_names_that_are_not_tools(api):
    hint = hint_client.make_hint({"trajectory"}, "rt", ["get_user_details", "made_up_tool", "think"],
                                 domain="airline")
    assert hint["trajectory"] == ["get_user_details", "think"] and hint["current_tool"] == "think"
    assert api.post("/predict", json=hint).status_code == 200
