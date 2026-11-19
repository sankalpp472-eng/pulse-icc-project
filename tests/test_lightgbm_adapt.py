"""
lightgbm_adapt (pulse/predictor/predictors/lightgbm_adapt.py): lightgbm + an online adapter.
Uses the airline models/action_model.pkl (in git); no cluster, LLM or embedding server. Time is faked.
"""

import pytest

import pulse.predictor.predictors.lightgbm_adapt as la
from pulse.predictor import predictors
from pulse.predictor.action_model import NUMERIC, RESPOND
from pulse.predictor.contract import PredictRequest
from pulse.predictor.predictors.lightgbm import LightGBM

SIG = {k: 0.0 for k in NUMERIC}


def req(episode, traj):
    return PredictRequest(workflow_id="W1", episode_id=episode, domain="airline", trajectory=traj, signals=SIG)


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(la.time, "time", lambda: t["now"])
    return t


def test_it_is_in_the_registry():
    assert predictors.canonical("lightgbm_adapt") == "lightgbm_adapt"


def test_a_fresh_adapter_answers_as_lightgbm(clock):
    p0 = LightGBM().predict(req("e1", ["get_user_details"]))
    q = la.LightGBMAdapt().predict(req("e1", ["get_user_details"]))
    assert set(q) == set(p0)
    assert all(abs(q[a] - p0[a]) < 1e-4 for a in p0)


def test_what_a_step_did_is_read_from_the_next_hint(clock):
    p = la.LightGBMAdapt()
    p.predict(req("e1", []))                                      # step 0: the agent replies (no new tool)
    p.predict(req("e1", []))                                      # step 1: it calls get_user_details
    p.predict(req("e1", ["get_user_details"]))                    # step 2: its outcome is never seen
    truths = [s[2] for s in p.episodes["e1"]["steps"]]
    assert truths == [[RESPOND], ["get_user_details"], None]


def test_it_learns_after_an_idle_episode_and_moves_toward_what_the_agent_did(clock):
    p = la.LightGBMAdapt()
    p0 = LightGBM().predict(req("x", ["get_user_details"]))
    rare = min((a for a in p0 if a != RESPOND), key=p0.get)      # a tool lightgbm thinks unlikely here
    for i in range(20):                                           # every episode: get_user_details, then `rare`
        p.predict(req(f"e{i}", ["get_user_details"]))
        clock["now"] += 1
        p.predict(req(f"e{i}", ["get_user_details", rare]))
        clock["now"] += la.SESSION_TIMEOUT_S + 1                  # idle: the next hint ends it
    q = p.predict(req("new", ["get_user_details"]))
    assert p.learned_episodes == 20
    assert q[rare] > 2 * p0[rare]
    assert p.health()["adapter"]["learned_steps"] == 20           # one labelled step per episode


def test_without_signals_it_is_lightgbm_markov_fallback_and_does_not_track(clock):
    p = la.LightGBMAdapt()
    r = PredictRequest(workflow_id="W1", episode_id="e1", domain="airline", trajectory=["get_user_details"])
    assert p.predict(r) == LightGBM().predict(r)
    assert p.episodes == {}
