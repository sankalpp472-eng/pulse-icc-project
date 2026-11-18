"""
The lease controller's policy (pulse/controller/leases.py) and the simulator's capacity limit (pulse/ui/sim_cluster.py).
No cluster, Docker or LLM needed: the lease book is pure logic, the simulator runs on a fake clock.
"""

import pytest

import pulse.ui.sim_cluster as sim_cluster
from pulse.controller.leases import LeaseBook, LeaseConfig

COLD = {"a": 2.0, "b": 2.0, "c": 2.0}


def book(**kw):
    return LeaseBook(LeaseConfig(**{"reserve_pods": 2, "k_max": 5, "ttl_min_s": 5, "ttl_max_s": 120, **kw}),
                     cold_s=COLD)


def test_admits_when_it_fits_and_ttl_is_clamped():
    b = book()
    d = b.request("a", "e1", p=0.8, k=2, ttl_s=500, now=0.0, free_pods=10)
    assert d.action == "admit" and d.k == 2 and d.extra == 2
    assert b.leases["a"].expires == 120                       # clamped to ttl_max
    assert b.active() == {"a": 2}


def test_refresh_needs_no_new_pods_even_when_full():
    b = book()
    b.request("a", "e1", p=0.8, k=1, ttl_s=20, now=0.0, free_pods=10)
    d = b.request("a", "e2", p=0.5, k=1, ttl_s=20, now=1.0, free_pods=0)
    assert d.action == "admit" and d.extra == 0
    assert abs(b.leases["a"].p - (1 - 0.2 * 0.5)) < 1e-9     # two episodes: P(at least one calls it)


def test_k_is_lowered_before_anything_is_rejected():
    b = book()
    d = b.request("a", "e1", p=0.9, k=4, ttl_s=20, now=0.0, free_pods=3)     # room = 3 - 2 reserve = 1
    assert d.action == "reduced" and d.k == 1


def test_the_reserve_is_never_leased():
    b = book(reserve_pods=3)
    d = b.request("a", "e1", p=0.9, k=1, ttl_s=20, now=0.0, free_pods=3)
    assert d.action == "reject"                               # 3 free, all reserved for on-demand calls


def test_evicts_a_lower_value_lease_to_make_room():
    b = book()
    b.request("a", "e1", p=0.2, k=1, ttl_s=20, now=0.0, free_pods=10)
    d = b.request("b", "e2", p=0.9, k=1, ttl_s=20, now=1.0, free_pods=2)       # room 0: needs one pod
    assert d.action == "admit" and d.evict == ["a"]
    assert set(b.active()) == {"b"}


def test_never_evicts_a_higher_value_or_busy_lease():
    b = book()
    b.request("a", "e1", p=0.9, k=1, ttl_s=20, now=0.0, free_pods=10)
    assert b.request("b", "e2", p=0.2, k=1, ttl_s=20, now=1.0, free_pods=2).action == "reject"
    b2 = book()
    b2.request("a", "e1", p=0.1, k=1, ttl_s=20, now=0.0, free_pods=10)
    assert b2.request("b", "e2", p=0.9, k=1, ttl_s=20, now=1.0, free_pods=2, busy={"a"}).action == "reject"


def test_expiry_and_supersede_when_room_is_tight():
    b = book()                                        # free 5 - reserve 2 = 3 < idle_free_pods 8: no idle warmth
    b.request("a", "e1", p=0.8, k=1, ttl_s=10, now=0.0, free_pods=5)
    b.request("c", "e1", p=0.8, k=1, ttl_s=60, now=0.0, free_pods=5, lookahead=True)
    d = b.request("b", "e1", p=0.8, k=1, ttl_s=10, now=5.0, free_pods=5)    # e1's next hint: not "a" any more
    assert d.ended == ["a"] and set(b.active()) == {"b", "c"}               # the lookahead claim stays
    assert b.expire(20.0, free_pods=5) == ["b"] and set(b.active()) == {"c"}


def test_idle_warmth_while_there_is_room_and_shed_when_tight():
    b = book()
    b.request("a", "e1", p=0.8, k=1, ttl_s=10, now=0.0, free_pods=20)
    assert b.expire(11.0, free_pods=20) == [] and "a" in b.active()         # ended with room: idles 60 s
    assert b.expire(80.0, free_pods=20) == ["a"]                            # idle warmth runs out once
    b.request("b", "e1", p=0.8, k=1, ttl_s=10, now=100.0, free_pods=20)
    d = b.request("c", "e1", p=0.8, k=1, ttl_s=10, now=105.0, free_pods=20)
    assert d.ended == [] and "b" in b.active()                              # superseded with room: idles
    assert b.expire(106.0, free_pods=3) == ["b"]                            # room gone: idle warmth is shed
    assert b.stats["shed"] == 1


def test_adopt_a_called_tool_only_with_room():
    b = book()
    assert b.adopt("a", 0.0, free_pods=20) and "a" in b.active()
    assert not b.adopt("b", 0.0, free_pods=5)
    assert b.leases["a"].p == pytest.approx(0.05)    # low value: the first to be evicted


def test_adaptive_k_follows_the_observed_fanout():
    b = book()
    for _ in range(10):
        b.observe_fanout("a", 2)
    assert b.k_cap("a") == 2
    assert b.request("a", "e1", p=0.9, k=5, ttl_s=20, now=0.0, free_pods=50).k == 2


# ---------------------------------------------------------------------- the simulator's capacity
class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(sim_cluster.time, "time", c)
    return c


def test_pods_beyond_capacity_wait_for_a_slot(clock):
    sim = sim_cluster.SimCluster(["a", "b"], window_s=6, capacity=1, grace_s=2)
    sim.cold = {"a": 1.0, "b": 1.0}
    w1, cold1, p1 = sim.acquire("a")
    sim.release(p1)                                  # released at once; ready after the cold start
    w2, cold2, p2 = sim.acquire("b")
    assert w2 == sim_cluster.INF and cold2 and sim.free_pods() < 0
    clock.t += w1 + 6 + 2 + 0.5                      # a's pod idles out (window), then its slot frees (grace)
    sim.tick()
    assert p2["ready_at"] != sim_cluster.INF and sim.pending_starts == 1


def test_keepalive_keeps_a_leased_tool_from_scaling_down(clock):
    sim = sim_cluster.SimCluster(["a"], window_s=6)
    sim.prewarm("a", 1)
    for _ in range(10):
        clock.t += 3
        sim.keepalive("a")
        sim.tick()
    assert len(sim.pods["a"]) == 1
    clock.t += 10                                    # no more keep-alive: gone after the window
    sim.tick()
    assert sim.pods["a"] == []
