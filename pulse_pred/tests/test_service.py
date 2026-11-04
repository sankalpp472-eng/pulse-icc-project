"""pulse_pred/service.py: hints in predictor/service.py's format -> a pulse-v3 session -> ranked tools."""

import importlib.util
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parents[1]
if not (os.getenv("PULSE_MODEL_URI") or Path(os.getenv("PULSE_MODEL") or HERE / "data" / "predictor.pkl").exists()):
    pytest.skip("no pulse-v3 model: set PULSE_MODEL_URI (S3) or PULSE_MODEL", allow_module_level=True)

# loaded under its own name: predictor/service.py is also a top-level "service"
_spec = importlib.util.spec_from_file_location("pulse_v3_service", HERE / "service.py")
service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(service)

C = TestClient(service.app)


def test_appended_follows_a_sliding_trajectory_window():
    assert service.appended([], ["a"]) == ["a"]
    assert service.appended(["a", "b"], ["a", "b", "c"]) == ["c"]
    assert service.appended(list("abcdefgh"), list("bcdefghi")) == ["i"]
    assert service.appended(["a"], ["a"]) == []


def test_last_step_reads_calls_results_and_the_customer():
    hist = [{"role": "user", "content": "cancel my trip"},
            {"role": "assistant", "content": "", "calls": [["get_user_details", '{"user_id": "u1"}']]},
            {"role": "tool", "name": "get_user_details", "content": '{"reservations": ["R1"]}'}]
    calls, results, user = service.last_step(hist)
    assert calls == [("get_user_details", {"user_id": "u1"})] and results == ['{"reservations": ["R1"]}']
    assert user == "cancel my trip"


def test_a_session_through_predict():
    ep = "test-episode-1"
    hist = [{"role": "user", "content": "Hi, I want to cancel reservation EHGLP3. My user id is emma_kim_9957."}]
    r = C.post("/predict", json={"workflow_id": "W1", "episode_id": ep, "trajectory": [], "history": hist}).json()
    assert r["model"] == "pulse-v3" and abs(sum(x["p"] for x in r["ranked"]) - 1) < 0.05
    assert all("." not in x["tool"] for x in r["ranked"])                      # unprefixed, as the cluster names them
    hist += [{"role": "assistant", "content": "", "calls": [["get_user_details", {"user_id": "emma_kim_9957"}]]},
             {"role": "tool", "name": "get_user_details", "content": '{"reservations": ["EHGLP3"]}'}]
    r = C.post("/predict", json={"workflow_id": "W1", "episode_id": ep, "trajectory": ["get_user_details"],
                                 "history": hist}).json()
    assert {p["tool"] for p in r["prewarm"]} <= {x["tool"] for x in r["ranked"]} - {"END"}
    assert service.PRED.sessions[ep].steps[-1].tools == ["airline.get_user_details"]
    assert C.get("/healthz").json()["sessions"] >= 1
