"""
Checks for the LightGBM predictor and its signals.   From the repo root:

    python -m pytest predictor/tests -q

The model checks need predictor/data/lgbm.pkl (build_lgbm_data.py + train_lgbm.py); they are
skipped if it is missing. The signal checks always run.
"""

import json
import pickle
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from signals import SIGNAL_KEYS, compute_signals, events_from_messages  # noqa: E402

RES = json.dumps({"reservation_id": "X1", "cabin": "basic_economy", "insurance": "no",
                  "flight_type": "round_trip", "flights": [{"flight_number": "HAT1"}]})
USER = json.dumps({"membership": "gold", "reservations": ["X1", "X2", "X3"]})


def test_signals_from_events():
    s = compute_signals([("user", "Hi, I want to CANCEL my trip and get a refund"),
                         ("tool", "get_user_details", USER),
                         ("tool", "get_reservation_details", RES),
                         ("user", "yes please")])
    assert set(s) == set(SIGNAL_KEYS)
    assert s["n_calls"] == 2 and s["user_turns_since_tool"] == 1
    assert s["res_basic"] == 1 and s["res_ins"] == 0 and s["res_round"] == 1 and s["res_seen"] == 1
    assert s["user_n_res"] == 3 and s["member"] == 2
    assert s["kw_cancel"] == 1 and s["kw_comp"] == 1 and s["kw_bag"] == 0


def test_signals_are_numbers_only():                      # privacy: no raw text leaves
    s = compute_signals([("user", "my secret phrase is pineapple"), ("tool", "think", "Error: boom")])
    assert all(isinstance(v, int) for v in s.values())
    assert s["last_err"] == 1


def test_agent_messages_match_events():                   # same signals on the agent side
    msgs = [{"role": "system", "content": "policy"},
            {"role": "user", "content": "please cancel"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_call_id": "a", "content": RES}]
    ev = events_from_messages(msgs, ["get_reservation_details"])
    assert ev == [("user", "please cancel"), ("tool", "get_reservation_details", RES)]


@pytest.fixture(scope="module")
def pred():
    path = HERE / "data" / "lgbm.pkl"
    if not path.exists():
        pytest.skip("no lgbm.pkl yet")
    with open(HERE / "data" / "model.pkl", "rb") as f:
        p = pickle.load(f)
    with open(path, "rb") as f:
        p.lgbm = pickle.load(f)
    return p


def test_lgbm_probabilities_sum_to_one(pred):             # FR-4
    sig = compute_signals([("user", "change my flight"), ("tool", "get_reservation_details", RES)])
    for wf in ("W1", "W2", "airline"):
        for traj in ([], ["get_user_details"], ["get_user_details", "get_reservation_details"]):
            p = pred.proba(wf, traj, "lgbm", sig)
            assert abs(sum(p.values()) - 1) <= 0.01 and min(p.values()) >= 0


def test_lgbm_without_signals_falls_back_to_ensemble(pred):
    traj = ["get_user_details"]
    assert pred.proba("W1", traj, "lgbm", None) == pred.proba("W1", traj, "ensemble")


def test_lgbm_unknown_tool_does_not_crash(pred):
    p = pred.proba("W1", ["a_tool_nobody_has_seen"], "lgbm", {})
    assert abs(sum(p.values()) - 1) <= 0.01
