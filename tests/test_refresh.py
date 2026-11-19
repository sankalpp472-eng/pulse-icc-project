"""
How long prewarmed pods live (pulse/controller/refresh.py): the refresh policy, pressure shedding, per-tool idle
windows, and the simulator's per-tool windows (pulse/ui/sim_cluster.py). Pure logic and a fake clock.
"""

import pytest

import pulse.ui.sim_cluster as sim_cluster
from pulse.controller.refresh import RefreshConfig, decide, refresh_threshold, tool_windows


def act(p, have, k=1, free=0, pending=0, **cfg):
    return decide(p, have, k, RefreshConfig(**cfg), free, pending)[0]


def test_a_cold_tool_is_started_and_always_refreshes_a_warm_one_as_before():
    assert act(0.3, have=0) == "start"
    assert act(0.3, have=1, k=2) == "start"                 # fewer pods than k: new pods
    assert act(0.3, have=1) == "refresh"                    # the default: every prediction pings


def test_never_leaves_a_warm_tool_to_its_real_calls():
    assert act(0.99, have=1, policy="never") == "skip"
    assert act(0.3, have=0, policy="never") == "start"      # a cold tool is still started


def test_value_refreshes_freely_with_room_and_only_likely_tools_when_full():
    cfg = RefreshConfig(policy="value", free_high=6, p_max=0.8)
    assert refresh_threshold(6, cfg) == 0 and refresh_threshold(0, cfg) == pytest.approx(0.8)
    assert refresh_threshold(3, cfg) == pytest.approx(0.4)
    assert act(0.31, have=1, free=10, policy="value") == "refresh"
    assert act(0.5, have=1, free=0, policy="value") == "skip"
    assert act(0.85, have=1, free=0, policy="value") == "refresh"


def test_shedding_under_pressure_drops_unlikely_predictions_only_while_pods_wait():
    assert act(0.4, have=0, pending=3, shed_p=0.6) == "skip"           # no new pods for a weak guess
    assert act(0.4, have=1, pending=3, shed_p=0.6) == "skip"           # nor a refresh: let it age out
    assert act(0.7, have=0, pending=3, shed_p=0.6) == "start"
    assert act(0.4, have=0, pending=0, shed_p=0.6) == "start"          # no pressure: as before
    assert RefreshConfig(shed_p=0.6).needs_capacity() and not RefreshConfig().needs_capacity()


def test_tool_windows_take_a_quantile_of_the_reuse_gaps_clamped():
    w = tool_windows({"busy": [5, 6, 7, 8, 30], "mid": [10, 35, 40, 45, 50], "rare": [80, 90, 100, 200, 5],
                      "once": []}, q=0.8, lo=20, hi=60)
    assert w == {"busy": 20, "mid": 45, "rare": 20, "once": 20}


# ---------------------------------------------------------------------- the simulator's per-tool windows
@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(sim_cluster.time, "time", lambda: t["now"])
    return t


def test_the_simulator_scales_each_tool_down_after_its_own_window(clock):
    sim = sim_cluster.SimCluster(["short", "long"], window_s=60, windows={"short": 20})
    sim.cold = {"short": 1.0, "long": 1.0}
    sim.prewarm("short", 1)
    sim.prewarm("long", 1)
    clock["now"] += 30
    sim.tick()
    assert sim.pods["short"] == [] and len(sim.pods["long"]) == 1
    clock["now"] += 40
    sim.tick()
    assert sim.pods["long"] == []


def test_the_simulator_counts_pods_waiting_for_a_slot(clock):
    sim = sim_cluster.SimCluster(["a", "b"], window_s=60, capacity=1)
    sim.prewarm("a", 1)
    assert sim.pending_pods() == 0
    sim.prewarm("b", 1)
    assert sim.pending_pods() == 1
