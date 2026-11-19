"""
Adaptive top-k, which replaced the θ threshold: the predictor's candidates (pulse/predictor/policy.py), their
hit rates learned online (pulse/predictor/calibration.py), the CRD fields they travel in (crd_writer.py), and the
controller's per-candidate test (pulse/controller/adaptive.py). Pure logic.
"""

import pytest

from pulse.controller.adaptive import slot_price, worth_warming
from pulse.predictor.calibration import Calibrator, appended
from pulse.predictor.contract import RESPOND
from pulse.predictor.crd_writer import make_object
from pulse.predictor.policy import prewarm_select

RANKED = [(RESPOND, 0.5), ("a", 0.3), ("b", 0.1), ("c", 0.06), ("d", 0.04)]


def test_adaptive_offers_the_top_three_tools_over_the_floor_and_topk_exactly_k():
    assert prewarm_select(RANKED) == ["a", "b", "c"]                                  # RESPOND never; adaptive = 3
    assert prewarm_select(RANKED, "topk", top_k=2) == ["a", "b"]
    assert prewarm_select(RANKED, "adaptive", p_min=0.2) == ["a"]                     # the floor
    with pytest.raises(ValueError):
        prewarm_select(RANKED, "theta")                                                # θ is gone


def test_a_hit_rate_starts_at_p_and_moves_to_what_really_happens():
    c = Calibrator(prior_n=10)
    assert c.rate(0.5) == pytest.approx(0.5)
    for i in range(30):                    # 30 episodes: 'a' offered at p 0.5, the agent then calls 'b' instead
        c.observe(f"e{i}", [], [("a", 0.5)], now=0)
        c.observe(f"e{i}", ["b"], [], now=1)
    assert c.rate(0.5) == pytest.approx(10 * 0.5 / 40)        # 0 hits in 30, the prior's weight 10
    for i in range(30):                    # and now it is right every time
        c.observe(f"f{i}", ["b"], [("a", 0.5)], now=2)
        c.observe(f"f{i}", ["b", "a"], [], now=3)
    assert c.rate(0.5) == pytest.approx((30 + 5) / 70)


def test_a_reply_step_is_a_miss_for_every_candidate():
    c = Calibrator(prior_n=0.001)
    c.observe("e", ["x"], [("a", 0.35), ("b", 0.35)], now=0)
    c.observe("e", ["x"], [], now=1)                          # trajectory unchanged: the agent answered the customer
    assert c.rate(0.35) == pytest.approx(0.0, abs=1e-3)


def test_idle_episodes_are_dropped_without_an_outcome():
    c = Calibrator(timeout_s=300)
    c.observe("e", [], [("a", 0.9)], now=0)
    c.observe("other", [], [], now=1000)                      # e went idle: its last step is never scored
    c.observe("e", ["a"], [], now=1001)
    assert c.stats()["bins"][-1]["candidates"] == 0
    assert appended(["a", "b"], ["a", "b", "c"]) == ["c"] and appended(["a"], ["a"]) == []


def test_the_crd_carries_rank_and_hit_rate_only_for_adaptive_candidates():
    spec = make_object("get_user_details", 0.3, 0.05, 20, rank=2, hit_rate=0.123456)["spec"]
    assert (spec["rank"], spec["hitRate"], spec["threshold"]) == (2, 0.1235, 0.05)
    assert "hitRate" not in make_object("get_user_details", 0.3, 0.05, 20)["spec"]


def test_a_candidate_is_worth_a_slot_from_5_percent_with_room_and_33_percent_when_full():
    # only a wrong prewarm wastes a slot: hit x cold >= (1 - hit) x price
    assert slot_price(free=10) == pytest.approx(0.075) and slot_price(free=0) == pytest.approx(0.75)
    assert slot_price(free=10, pending=2) == pytest.approx(0.75)      # pods waiting = full
    assert worth_warming(0.05, 1.5, free=10)[0] and not worth_warming(0.04, 1.5, free=10)[0]
    assert worth_warming(0.35, 1.5, free=0)[0] and not worth_warming(0.3, 1.5, free=0)[0]
    assert worth_warming(0.2, 4.0, free=0)[0]                          # a slow-starting tool is worth more
