"""
Regression tests for issue #15: the per-episode "Pod-s" in the UI must include the idle window the episode's pods
live through after it ends, until they scale down (on the simulated cluster and on Knative).

No cluster, Docker or LLM needed: the simulator runs on a fake clock and the Knative tracker on fake pod listings.
"""

import pytest

import pulse.ui.backend as backend
import pulse.ui.sim_cluster as sim_cluster
from pulse.ui.session import Session

WINDOW = 60.0


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(sim_cluster.time, "time", c)
    return c


# ---------------------------------------------------------------------- simulated cluster
def test_sim_episode_pod_seconds_include_scale_down_tail(clock):
    sim = sim_cluster.SimCluster(["get_user_details", "book_reservation"], window_s=WINDOW)
    ep = Session({"kind": "replay"}, cluster=None, sim=sim, predictor_url=None)
    sim.prewarm("get_user_details", 1)                 # a pod starts (cold start ~1.6 s)
    clock.t += 10
    sim.prewarm("book_reservation", 2)                 # two more pods
    clock.t += 20                                      # the episode ends 30 s after it started
    s = ep.summary()

    clock.t += 300                                     # nothing else happens: every pod scales down
    sim.snapshot()
    assert all(not pods for pods in sim.pods.values())
    caused = sim.live_pod_seconds() - ep.pod_s0        # what the pods really cost, counted after scale-down

    assert s["pod_seconds_after"] > 0
    assert s["pod_seconds"] == pytest.approx(s["pod_seconds_during"] + s["pod_seconds_after"], abs=0.11)
    assert s["pod_seconds"] == pytest.approx(caused, abs=0.2)   # the bug: only pod_seconds_during was reported
    # each pod lives >= the 60 s window after it was ready, so the tail is at least 3 windows minus the episode
    assert s["pod_seconds"] >= 3 * WINDOW


def test_sim_reap_charges_exactly_up_to_the_window(clock):
    sim = sim_cluster.SimCluster(["a"], window_s=WINDOW)
    sim.prewarm("a", 1)
    ready = sim.pods["a"][0]["ready_at"]
    created = sim.pods["a"][0]["created"]
    clock.t += 1000                                    # noticed much later than the scale-down
    sim.snapshot()
    assert sim.pod_seconds == pytest.approx(ready + WINDOW - created)


def test_sim_min_scale_pods_have_no_tail(clock):
    sim = sim_cluster.SimCluster(["a"], window_s=WINDOW, min_scale=1)    # C1: always warm
    assert sim.pending_pod_seconds() == 0
    sim.prewarm("a", 2)                                # one extra pod above min-scale
    clock.t += 5
    assert 0 < sim.pending_pod_seconds() <= WINDOW + 3


# ---------------------------------------------------------------------- Knative (kind / EKS)
def _pod(name, tool):
    return {"metadata": {"name": name, "labels": {"pulse/tool": tool}}}


@pytest.fixture
def cluster(monkeypatch):
    monkeypatch.setattr(backend.Cluster, "_poll", lambda self: None)   # no kubectl: feed pod listings by hand
    c = backend.Cluster()
    c.mode, c.min_scale = "knative", 0
    return c


def test_knative_episode_waits_for_scale_down(cluster):
    T = 1000.0
    summ = {}
    for t in range(95, 191):                           # a1 lives 95..190, a2 110..125; the episode is 100..130
        items = [_pod("a1", "get_user_details")] + ([_pod("a2", "book_reservation")] if 110 <= t <= 125 else [])
        if t == 130:
            cluster.track_episode(summ, T + 100, T + 130)
            assert summ["pod_seconds"] is None and summ["pod_seconds_pending"]
        cluster._pods_seen(items, T + t)
        if t > 130:
            assert summ["pod_seconds_pending"]         # still counting while a pod is alive
    cluster._pods_seen([], T + 191)                    # all scaled down
    assert summ["pod_seconds_pending"] is False
    assert summ["pod_seconds_during"] == 45            # a1 30 s + a2 15 s
    assert summ["pod_seconds_after"] == 60             # a1 130..190
    assert summ["pod_seconds"] == 105
    assert summ["pod_seconds_note"] == "scaled down"


def test_knative_next_episode_cuts_the_tail(cluster):
    T = 1000.0
    summ = {}
    for t in range(290, 331):
        if t == 300:
            cluster.track_episode(summ, T + 280, T + 300)
        if t == 330:
            cluster.episode_started(T + 330)           # the next episode's pod time is its own
        cluster._pods_seen([_pod("b1", "x")], T + t)
    assert summ["pod_seconds"] == 40                   # 290..330, not beyond
    assert summ["pod_seconds_note"] == "next episode started"


def test_knative_always_warm_settles_at_once(cluster):
    cluster.min_scale = 1                              # C1: the min-scale pod never goes away
    T = 1000.0
    for t in range(330, 341):
        cluster._pods_seen([_pod("c1", "x")], T + t)
    summ = {}
    cluster.track_episode(summ, T + 330, T + 340)
    cluster._pods_seen([_pod("c1", "x")], T + 341)
    assert summ["pod_seconds_pending"] is False
    assert summ["pod_seconds_during"] == 10


def test_knative_gives_up_after_the_timeout(cluster):
    T = 1000.0
    summ = {}
    cluster._pods_seen([_pod("d1", "x")], T)
    cluster.track_episode(summ, T, T + 10)
    cluster._pods_seen([_pod("d1", "x")], T + 10 + backend.POD_TAIL_TIMEOUT_S + 1)
    assert summ["pod_seconds_pending"] is False
    assert summ["pod_seconds_note"] == "timed out"
