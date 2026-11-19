"""
The k rule's warm time and budget learned at runtime (pulse/controller/adaptive.py, controller K_ADAPT=1).
Pure logic: no cluster or Prometheus (an unreachable one is part of the test).
"""

import math

import pytest

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pulse.controller.adaptive import (REF_COLD_S, WAIT_SEED_S, Estimates, budget, parse_prom, query_warm,
                                       slot_price, worth_warming)


def test_the_budget_is_the_setting_with_room_and_a_normal_cold_start():
    assert budget(0.3, REF_COLD_S, free=10) == pytest.approx(0.3)


def test_the_budget_grows_when_the_nodes_fill_and_when_pods_start_slowly():
    assert budget(0.3, REF_COLD_S, free=0) == pytest.approx(0.6)            # full: x2
    assert budget(0.3, REF_COLD_S, free=3) == pytest.approx(0.45)           # half full
    assert budget(0.3, 2 * REF_COLD_S, free=10) == pytest.approx(0.6)       # twice as slow to start: x2
    assert budget(0.3, 100.0, free=0) == pytest.approx(0.3 * 3 * 2)        # the cold factor is clamped to 3
    assert budget(0.3, 0.1, free=10) == pytest.approx(0.15)                 # ... and to 0.5


def test_prometheus_answers_keep_real_warm_medians_only():
    ans = {"data": {"result": [
        {"metric": {"tool": "fast"}, "value": [0, "0.045"]},
        {"metric": {"tool": "nodata"}, "value": [0, "NaN"]},
        {"metric": {"tool": "cold_mixed"}, "value": [0, "2.5"]},              # a 'warm' median over 1 s: not warm
        {"metric": {}, "value": [0, "0.1"]}]}}
    assert parse_prom(ans) == {"fast": 0.045}
    assert parse_prom({}) == {}


def test_estimates_fall_back_to_the_settings_and_learn_cold_starts():
    e = Estimates(0.05, {"slow": 0.5})
    assert (e.warm("x"), e.warm("slow"), e.cold("x")) == (0.05, 0.5, REF_COLD_S)
    e.set_warm({"x": 0.2})
    assert e.warm("x") == 0.2 and e.warm("slow") == 0.5
    e.observe_cold("x", 3.0)
    e.observe_cold("x", 1.0)
    assert e.cold("x") == pytest.approx(0.7 * 3.0 + 0.3 * 1.0)


def test_an_unreachable_prometheus_gives_nothing():
    assert query_warm("http://127.0.0.1:9", timeout=0.5) == {}


def test_the_controller_uses_the_learned_numbers_only_with_k_adapt(monkeypatch):
    pytest.importorskip("kubernetes")
    import pulse.controller.prewarm_controller as pc
    est = Estimates(0.05, {})
    monkeypatch.setattr(pc, "ESTIMATES", est)
    monkeypatch.setattr(pc, "K_ADAPT", False)
    assert pc.pods_for("t", 4) == 1                                         # fixed: 0.05 s, 0.3 s
    monkeypatch.setattr(pc, "K_ADAPT", True)
    est.set_warm({"t": 0.5})                                               # Prometheus says it is slow
    assert pc.pods_for("t", 4, free=10) == 4                               # 4 x 0.5 / 0.3 -> all 4
    est.observe_cold("t", 3.0)                                             # and slow to start, on a full node:
    assert pc.pods_for("t", 4, free=0) == math.ceil(4 * 0.5 / (0.3 * 2 * 2))   # budget 1.2 s -> 2 pods


# ---- learned slot price (TOPK_PRICE=learned)
def test_the_learned_price_starts_as_the_fixed_one():
    e = Estimates(0.05)
    for free, pending in ((10, 0), (3, 0), (0, 0), (5, 2)):
        assert slot_price(free, pending, wait_s=e.wait()) == pytest.approx(slot_price(free, pending))
    assert e.wait() == pytest.approx(WAIT_SEED_S) and e.snapshot()["slot_wait_s"] is None


def test_the_price_follows_the_slot_wait_when_full_only():
    assert slot_price(10, 0, wait_s=30.0) == pytest.approx(0.075)           # room: nobody waits for a slot
    assert slot_price(0, 3, wait_s=30.0) == pytest.approx(0.075 + 30.0)     # pods Pending: the whole wait
    assert slot_price(3, 0, wait_s=10.0) == pytest.approx(0.075 + 0.5 * 10.0)
    # a 1.5 s tool on a full node with pods waiting 10 s: warmed from about 87 %, not 33 %
    assert not worth_warming(0.85, 1.5, 0, 1, wait_s=10.0)[0]
    assert worth_warming(0.88, 1.5, 0, 1, wait_s=10.0)[0]
    assert worth_warming(0.06, 1.5, 10, 0, wait_s=10.0)[0]                  # with room it is still 5 %


def test_slot_waits_count_once_and_skip_scheduling_noise():
    e = Estimates(0.05)
    assert not e.observe_wait("a", 0.4)                                      # under 1 s: scheduling, not a queue
    assert e.wait() == pytest.approx(WAIT_SEED_S)
    assert e.observe_wait("b", 20.0) and e.wait() == pytest.approx(20.0)
    assert not e.observe_wait("b", 20.0)                                     # the same pod seen again
    e.observe_wait("c", 500.0)                                               # capped at 60 s
    assert e.wait() == pytest.approx(0.7 * 20.0 + 0.3 * 60.0)


def test_the_controller_reads_a_pods_slot_wait(monkeypatch):
    pytest.importorskip("kubernetes")
    import pulse.controller.prewarm_controller as pc
    t0 = datetime(2026, 11, 19, tzinfo=timezone.utc)
    cond = SimpleNamespace(type="PodScheduled", status="True", last_transition_time=t0 + timedelta(seconds=12))
    pod = SimpleNamespace(metadata=SimpleNamespace(creation_timestamp=t0), status=SimpleNamespace(conditions=[cond]))
    assert pc.slot_wait_s(pod) == pytest.approx(12.0)
    cond.status = "False"
    assert pc.slot_wait_s(pod) is None
    est = Estimates(0.05)
    monkeypatch.setattr(pc, "ESTIMATES", est)
    monkeypatch.setattr(pc, "TOPK_PRICE", "fixed")
    assert pc.price_kw() == {}
    monkeypatch.setattr(pc, "TOPK_PRICE", "learned")
    assert pc.price_kw() == {"wait_s": pytest.approx(WAIT_SEED_S)}
