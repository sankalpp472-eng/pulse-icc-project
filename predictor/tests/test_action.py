"""
Checks for the next-action predictor (14 tools + RESPOND, fan-out k) and its signals.
From the repo root:

    python -m pytest predictor/tests -q

Model checks need predictor/data/action_model.pkl (build_action_data.py + train_action.py) and
are skipped without it. The signal checks always run.
"""

import importlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from signals import SIGNAL_KEYS, compute_signals, events_from_messages  # noqa: E402

USER = json.dumps({"membership": "gold", "reservations": ["NO6JO3", "AIXC49", "HKEG34"]})


def conversation():
    return [("user", "Please cancel ALL my trips, and I want a refund"),
            ("call", "get_user_details", {"user_id": "mia_li_3668"}),
            ("tool", "get_user_details", USER),
            ("call", "get_reservation_details", {"reservation_id": "NO6JO3"}),
            ("tool", "get_reservation_details", "Error: reservation not found")]


# ----------------------------------------------------------------- signals (spec feature list)

def test_signal_names_match_spec():
    for k in ["step", "n_calls", "n_distinct", "last_repeat", "calls_since_user", "last_event_user",
              "last_err", "pending_ids", "out_size", "kw_cancel", "kw_change", "kw_bag", "kw_book",
              "kw_upgrade", "kw_refund", "kw_all"]:
        assert k in SIGNAL_KEYS


def test_signal_values():
    s = compute_signals(conversation())
    assert s["step"] == 3 and s["n_calls"] == 2 and s["n_distinct"] == 2 and s["last_repeat"] == 1
    assert s["calls_since_user"] == 2 and s["last_event_user"] == 0
    assert s["last_err"] == 1 and s["pending_ids"] == 2 and s["out_size"] > 0
    assert s["kw_cancel"] == s["kw_all"] == s["kw_refund"] == 1 and s["kw_bag"] == 0


def test_repeat_and_customer_turn():
    ev = conversation() + [("call", "get_reservation_details", {"reservation_id": "AIXC49"}),
                           ("tool", "get_reservation_details", "{}"), ("respond",), ("user", "yes")]
    s = compute_signals(ev)
    assert s["last_repeat"] == 2 and s["pending_ids"] == 1
    assert s["calls_since_user"] == 0 and s["last_event_user"] == 1 and s["step"] == 5


def test_signals_are_numbers_only():                       # PRD: no raw text in the trace store
    s = compute_signals([("user", "my secret phrase is pineapple ZX81AB")])
    assert all(isinstance(v, int) for v in s.values())


def test_agent_and_langgraph_messages_give_same_signals():
    msgs = [{"role": "user", "content": "cancel all please"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_user_details", "arguments": json.dumps({"user_id": "u1"})}}]},
            {"role": "tool", "content": USER},
            {"role": "assistant", "content": "Which one?"}]
    s = compute_signals(events_from_messages(msgs))
    assert s["pending_ids"] == 3 and s["step"] == 3 and s["n_calls"] == 1
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    lc = [HumanMessage("cancel all please"),
          AIMessage("", tool_calls=[{"name": "get_user_details", "args": {"user_id": "u1"}, "id": "a"}]),
          ToolMessage(USER, tool_call_id="a"), AIMessage("Which one?")]
    assert compute_signals(events_from_messages(lc)) == s


# ----------------------------------------------------------------- models

@pytest.fixture(scope="module")
def pred():
    path = HERE / "data" / "action_model.pkl"
    if not path.exists():
        pytest.skip("no action_model.pkl yet")
    with open(HERE / "data" / "model.pkl", "rb") as f:
        p = pickle.load(f)
    with open(path, "rb") as f:
        p.action = pickle.load(f)
    return p


@pytest.mark.parametrize("model", ["gbt", "lr_action", "ngram_action", "lgbm"])
def test_action_probabilities(pred, model):                # FR-4 + RESPOND is a class
    sig = compute_signals(conversation())
    for traj in ([], ["get_user_details"], ["get_user_details", "get_reservation_details"]):
        p = pred.proba("W1", traj, model, sig)
        assert abs(sum(p.values()) - 1) <= 0.01 and min(p.values()) >= 0
        assert "RESPOND" in p and len(p) == 15


def test_respond_is_never_prewarmed(pred):
    from models import prewarm_set
    ranked = pred.rank("W1", ["get_user_details"], "gbt", compute_signals(conversation()))
    assert "RESPOND" not in prewarm_set(ranked, 0.0)


def test_no_signals_falls_back_to_ensemble(pred):
    assert pred.proba("W1", ["get_user_details"], "gbt", None) == pred.proba("W1", ["get_user_details"], "ensemble")


def test_fanout_rule(pred):
    rule = pred.action.fanout
    assert "get_reservation_details" in rule.id_tools
    assert rule.k("get_reservation_details", {"pending_ids": 3}) == 3
    assert rule.k("get_reservation_details", {"pending_ids": 0}) == 1
    assert rule.k("get_reservation_details", {"pending_ids": 40}) == 5          # CRD maximum
    assert rule.k("search_direct_flight", {"pending_ids": 3}) == 1


def test_gbt_is_fast(pred):                                 # spec: well under 1 ms
    sig = compute_signals(conversation())
    pred.proba("W1", ["get_user_details"], "gbt", sig)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        pred.proba("W1", ["get_user_details", "get_reservation_details"], "gbt", sig)
        ts.append(time.perf_counter() - t0)
    assert sorted(ts)[100] < 0.001


def test_api_returns_k(pred, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PREDICTOR_MODEL", "gbt")
    monkeypatch.setenv("THETA", "0.2")
    sys.modules.pop("service", None)
    service = importlib.import_module("service")
    r = TestClient(service.app).post("/predict", json={
        "workflow_id": "W1", "trajectory": ["get_user_details"],
        "signals": {**compute_signals(conversation()[:3]), "pending_ids": 3}}).json()
    assert r["model"] == "gbt"
    assert all(x["tool"] != "RESPOND" and x["k"] >= 1 for x in r["prewarm"])
    grd = [x for x in r["prewarm"] if x["tool"] == "get_reservation_details"]
    assert not grd or grd[0]["k"] == 3
    sys.modules.pop("service", None)
    os.environ.pop("PREDICTOR_MODEL", None)
