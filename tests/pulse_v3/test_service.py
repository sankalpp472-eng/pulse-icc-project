"""The pulse_v3 predictor behind predictor/service.py: hints -> a pulse-v3 session -> ranked tools."""

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if not (os.getenv("PULSE_MODEL_URI") or Path(os.getenv("PULSE_MODEL") or ROOT / "models" / "pulse_v3" / "predictor.pkl").exists()):
    pytest.skip("no pulse-v3 model: set PULSE_MODEL_URI (S3) or PULSE_MODEL", allow_module_level=True)

from pulse.predictor.predictors.pulse_v3 import appended, last_step  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    os.environ["PREDICTOR"] = "pulse_v3"
    sys.modules.pop("pulse.predictor.service", None)
    service = importlib.import_module("pulse.predictor.service")
    yield service, TestClient(service.app)
    sys.modules.pop("pulse.predictor.service", None)
    os.environ.pop("PREDICTOR", None)


def test_appended_follows_a_sliding_trajectory_window():
    assert appended([], ["a"]) == ["a"]
    assert appended(["a", "b"], ["a", "b", "c"]) == ["c"]
    assert appended(list("abcdefgh"), list("bcdefghi")) == ["i"]
    assert appended(["a"], ["a"]) == []


def test_last_step_reads_calls_results_and_the_customer():
    hist = [{"role": "user", "content": "cancel my trip"},
            {"role": "assistant", "content": "", "calls": [["get_user_details", '{"user_id": "u1"}']]},
            {"role": "tool", "name": "get_user_details", "content": '{"reservations": ["R1"]}'}]
    calls, results, user = last_step(hist)
    assert calls == [("get_user_details", {"user_id": "u1"})] and results == ['{"reservations": ["R1"]}']
    assert user == "cancel my trip"


def test_a_session_through_predict(client):
    service, c = client
    assert c.get("/info").json()["inputs"] == ["trajectory", "history"]
    ep = "test-episode-1"
    hist = [{"role": "user", "content": "Hi, I want to cancel reservation EHGLP3. My user id is emma_kim_9957."}]
    r = c.post("/predict", json={"workflow_id": "W1", "episode_id": ep, "trajectory": [], "history": hist}).json()
    assert r["model"] == "pulse_v3" and abs(sum(x["p"] for x in r["ranked"]) - 1) < 0.05
    assert all("." not in x["tool"] for x in r["ranked"])                      # unprefixed, as the cluster names them
    hist += [{"role": "assistant", "content": "", "calls": [["get_user_details", {"user_id": "emma_kim_9957"}]]},
             {"role": "tool", "name": "get_user_details", "content": '{"reservations": ["EHGLP3"]}'}]
    r = c.post("/predict", json={"workflow_id": "W1", "episode_id": ep, "trajectory": ["get_user_details"],
                                 "history": hist}).json()
    assert {p["tool"] for p in r["prewarm"]} <= {x["tool"] for x in r["ranked"]} - {"END"}
    assert service.PRED.pred.sessions[ep].steps[-1].tools == ["airline.get_user_details"]
    assert c.get("/healthz").json()["sessions"] >= 1
