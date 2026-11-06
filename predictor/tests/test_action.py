"""
Checks for the signals, the lightgbm / markov predictors (action_model.pkl) and the one prediction API.
From the repo root:

    python -m pytest predictor/tests -q

Model checks need predictor/data/action_model.pkl (build_action_data.py + train_action.py) and
are skipped without it. The signal and policy checks always run.
"""

import importlib
import json
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from contract import PredictRequest  # noqa: E402
from policy import prewarm_select, prewarm_set  # noqa: E402
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


# ----------------------------------------------------------------- policy

def test_policy_never_prewarms_respond_or_end():
    ranked = [("RESPOND", 0.5), ("a", 0.3), ("END", 0.15), ("b", 0.05)]
    assert prewarm_set(ranked, 0.0) == ["a", "b"]
    assert prewarm_select(ranked, "topk", top_k=1) == ["a"]
    assert prewarm_select(ranked, "topk", top_k=2, p_min=0.1) == ["a"]


# ----------------------------------------------------------------- lightgbm / markov

@pytest.fixture(scope="module")
def loaded():
    if not (HERE / "data" / "action_model.pkl").exists():
        pytest.skip("no action_model.pkl yet")
    import predictors
    return {n: predictors.load(n) for n in ("lightgbm", "markov")}


def req(traj, signals=None, **kw):
    return PredictRequest(workflow_id="W1", trajectory=traj, signals=signals, **kw)


@pytest.mark.parametrize("name", ["lightgbm", "markov"])
def test_action_probabilities(loaded, name):               # FR-4 + RESPOND is a class
    sig = compute_signals(conversation())
    for traj in ([], ["get_user_details"], ["get_user_details", "get_reservation_details"]):
        p = loaded[name].predict(req(traj, sig))
        assert abs(sum(p.values()) - 1) <= 0.01 and min(p.values()) >= 0
        assert "RESPOND" in p and len(p) == 15


def test_markov_ignores_signals(loaded):
    m = loaded["markov"]
    assert m.predict(req(["get_user_details"])) == m.predict(req(["get_user_details"], compute_signals(conversation())))


def test_lightgbm_without_signals_falls_back_to_markov(loaded):
    assert loaded["lightgbm"].predict(req(["get_user_details"])) == loaded["markov"].predict(req(["get_user_details"]))


def test_fanout_rule(loaded):
    g = loaded["lightgbm"]
    assert g.fanout("get_reservation_details", req([], {"pending_ids": 3})) == 3
    assert g.fanout("get_reservation_details", req([], {"pending_ids": 0})) == 1
    assert g.fanout("get_reservation_details", req([], {"pending_ids": 40})) == 5          # CRD maximum
    assert g.fanout("search_direct_flight", req([], {"pending_ids": 3})) == 1


def test_lightgbm_is_fast(loaded):                          # spec: well under 1 ms
    g, r = loaded["lightgbm"], req(["get_user_details", "get_reservation_details"], compute_signals(conversation()))
    g.predict(r)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        g.predict(r)
        ts.append(time.perf_counter() - t0)
    assert sorted(ts)[100] < 0.001


# ----------------------------------------------------------------- the API, for each local predictor

@pytest.mark.parametrize("name,inputs", [("lightgbm", ["trajectory", "signals"]), ("markov", ["trajectory"])])
def test_api(loaded, monkeypatch, name, inputs):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PREDICTOR", name)
    monkeypatch.setenv("THETA", "0.2")
    sys.modules.pop("service", None)
    service = importlib.import_module("service")
    c = TestClient(service.app)
    info = c.get("/info").json()
    assert info["name"] == name and info["inputs"] == inputs and "fanout" in info["supports"]
    r = c.post("/predict", json={"workflow_id": "W1", "trajectory": ["get_user_details"],
                                 "signals": {**compute_signals(conversation()[:3]), "pending_ids": 3}}).json()
    assert r["model"] == name and r["inputs_used"] == inputs
    assert all(x["tool"] not in ("RESPOND", "END") and x["k"] >= 1 for x in r["prewarm"])
    grd = [x for x in r["prewarm"] if x["tool"] == "get_reservation_details"]
    assert not grd or grd[0]["k"] == (3 if name == "lightgbm" else 1)
    sys.modules.pop("service", None)
