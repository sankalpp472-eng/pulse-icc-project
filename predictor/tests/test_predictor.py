"""
Checks for the PRD requirements the predictor owns.   Run from the repo root:

    python -m pytest predictor/tests -q

Needs predictor/data/model.pkl (run export_traces.py, build_graphs.py, train.py first).
"""

import pickle
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from models import END, MODEL_NAMES, prewarm_set  # noqa: E402


@pytest.fixture(scope="module")
def pred():
    with open(HERE / "data" / "model.pkl", "rb") as f:
        return pickle.load(f)


TRAJS = [[], ["get_user_details"], ["get_user_details", "get_reservation_details"],
         ["get_user_details", "search_direct_flight", "search_direct_flight", "calculate"]]


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("wf", ["W1", "W2", "airline", "unknown-workflow", None])
def test_probabilities_sum_to_one(pred, model, wf):          # FR-4
    for traj in TRAJS:
        p = pred.proba(wf, traj, model)
        assert abs(sum(p.values()) - 1.0) <= 0.01
        assert all(v >= 0 for v in p.values())


def test_ranked_highest_first(pred):
    ranked = pred.rank("W1", ["get_user_details"])
    ps = [p for _, p in ranked]
    assert ps == sorted(ps, reverse=True)


def test_threshold_filters(pred):                             # FR-5
    for theta in (0.1, 0.3, 0.5, 0.9):
        ranked = pred.rank("W1", ["get_user_details"])
        warm = prewarm_set(ranked, theta)
        assert all(dict(ranked)[t] >= theta for t in warm)
        assert all(t in warm for t, p in ranked if p >= theta and t != END)
        assert END not in warm


def test_unknown_tool_does_not_crash(pred):                   # FR-3 (no unknown-node errors)
    for model in MODEL_NAMES:
        p = pred.proba("W1", ["some_new_tool_nobody_has_seen"], model)
        assert abs(sum(p.values()) - 1.0) <= 0.01


def test_learns_the_obvious_step(pred):
    # In the airline traces, a conversation almost always starts by identifying the user
    top = pred.rank("W2", [])[0][0]
    assert top in {"get_user_details", "get_reservation_details"}
