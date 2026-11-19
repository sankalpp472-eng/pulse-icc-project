"""
The lease controller's policy (pulse/controller/leases.py) and the simulator's capacity limit (pulse/ui/sim_cluster.py).
No cluster, Docker or LLM needed: the lease book is pure logic, the simulator runs on a fake clock.
"""

import pytest

import pulse.ui.sim_cluster as sim_cluster
from pulse.controller.leases import LeaseBook, LeaseConfig

def book(**kw):
    return LeaseBook(LeaseConfig(**{"reserve_pods": 2, "k_max": 5, "ttl_min_s": 5, "ttl_max_s": 120,
                                    "window_s": 60, "extend_margin_s": 5, **kw}))


def test_admits_the_new_pods_when_they_fit_and_clamps_the_ttl():
    b = book()
    d = b.request("a", "e1", k=3, ttl_s=500, now=0.0, free_pods=10, have_pods=1)
    assert (d.action, d.k, d.new) == ("admit", 3, 2)          # 1 pod already there: 2 new
    assert b.active() == {"a": 120}                           # clamped to ttl_max


def test_a_warm_tool_only_gets_its_window_refreshed_even_when_full():
    b = book()
    d = b.request("a", "e1", k=1, ttl_s=20, now=0.0, free_pods=0, have_pods=1)
    assert (d.action, d.new) == ("admit", 0)


def test_k_is_lowered_to_the_room_left_next_to_the_reserve():
    b = book()
    d = b.request("a", "e1", k=4, ttl_s=20, now=0.0, free_pods=3, have_pods=0)     # room = 3 - 2 reserve = 1
    assert (d.action, d.k, d.new) == ("reduced", 1, 1)
    d = b.request("b", "e1", k=4, ttl_s=20, now=0.0, free_pods=2, have_pods=2)     # no room: keep the 2
    assert (d.action, d.k, d.new) == ("reduced", 2, 0)


def test_the_reserve_is_never_used_by_prewarms():
    b = book(reserve_pods=3)
    d = b.request("a", "e1", k=1, ttl_s=20, now=0.0, free_pods=3, have_pods=0)
    assert d.action == "reject" and "a" not in b.active()     # a cold tool's first real call starts it
    assert b.stats["rejected"] == 1 and b.stats["pods_started"] == 0


def test_a_lease_longer_than_the_window_gets_one_extending_ping():
    b = book()
    b.request("a", "e1", k=1, ttl_s=100, now=0.0, free_pods=10, have_pods=0)
    b.request("b", "e1", k=1, ttl_s=20, now=0.0, free_pods=10, have_pods=0)
    assert b.due_pings(50.0) == []                            # the window (60 s from the ping) still covers it
    assert b.due_pings(56.0) == ["a"]                         # 5 s before it would run out; "b" never needs one
    b.pinged("a", 56.0)
    assert b.due_pings(60.0) == []                            # now covered until 116 s > the lease's 100 s
    assert sorted(b.expire(101.0)) == ["a", "b"]


def test_k_rule_gives_fast_tools_one_pod_and_slow_tools_k():
    from pulse.controller.leases import rule_k
    assert [rule_k(k, 0.045, 0.3) for k in (1, 3, 6, 7, 20)] == [1, 1, 1, 2, 3]   # ~45 ms calls: queueing is cheap
    assert rule_k(4, 0.5, 0.3) == 4                                               # 0.5 s calls: every pod pays
    assert rule_k(4, 0.1, 0.3) == 2
    assert rule_k(3, 0.0, 0.3) == 3 and rule_k(0, 0.05, 0.3) == 1                 # no call time known: as predicted


def test_controller_applies_the_rule_per_tool(monkeypatch):
    pytest.importorskip("kubernetes")
    import pulse.controller.prewarm_controller as pc
    monkeypatch.setattr(pc, "TOOL_WARM", {"slow_tool": 0.6})
    assert pc.pods_for("get_user_details", 4) == 1 and pc.pods_for("slow_tool", 4) == 4
    monkeypatch.setattr(pc, "K_RULE", False)
    assert pc.pods_for("get_user_details", 4) == 4


def test_adaptive_k_follows_the_observed_fanout():
    b = book()
    for _ in range(10):
        b.observe_fanout("a", 2)
    assert b.k_cap("a") == 2
    assert b.request("a", "e1", k=5, ttl_s=20, now=0.0, free_pods=50, have_pods=0).k == 2


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


# ---------------------------------------------------------------------- the simulator's KPA model
def kpa_sim(**kw):
    sim = sim_cluster.SimCluster(["a"], window_s=60, autoscaler="kpa", **kw)
    sim.cold = {"a": 1.0}
    return sim


def test_kpa_a_busy_tool_queues_on_its_pod_instead_of_starting_another(clock):
    sim = kpa_sim()
    w, cold, p = sim.acquire("a")
    clock.t += w + 0.01                              # the pod is ready, and busy with the first call
    w2, cold2, p2 = sim.acquire("a")
    assert p2 is p and not cold2 and w2 < 0.1        # waits ~one warm call, as the activator would
    assert len(sim.pods["a"]) == 1


def test_kpa_short_burst_is_scaled_away_and_sustained_load_panics(clock):
    sim = kpa_sim()
    sim.prewarm("a", 3)                              # 3 pings of ~1 s: too short to move the 6 s panic average
    for _ in range(3):
        clock.t += 2
        sim.tick()
    assert len(sim.pods["a"]) == 1                   # extra pods gone within a few steps; one kept (traffic)
    clock.t += 70                                    # nothing for a whole stable window: to zero
    sim.tick()
    assert sim.pods["a"] == []

    sim2 = kpa_sim()
    sim2.pods["a"].append(sim2._new_pod("a", clock.t, ready_now=True))
    for _ in range(4):                               # 4 requests in flight for the last 6 s: a real burst
        sim2._request("a", clock.t - 6, None)
    sim2.tick()
    assert len(sim2.pods["a"]) >= 4 and sim2.panic_until["a"] > clock.t
    sim2.reqs["a"].clear()                           # the burst ends at once ...
    clock.t += 30
    sim2.tick()
    assert len(sim2.pods["a"]) >= 4                  # ... but panic never scales down within the stable window


def test_kpa_scale_down_halves_at_most_and_frees_slots_after_the_grace(clock):
    sim = kpa_sim(capacity=10, grace_s=30)
    for p in [sim._new_pod("a", clock.t, ready_now=True) for _ in range(8)]:
        sim.pods["a"].append(p)
    sim._request("a", clock.t - 10, clock.t - 9)      # a little traffic in the stable window: keep >= 1
    sim.tick()
    assert len(sim.pods["a"]) == 4                   # 8 -> at most half removed in one step
    assert sim.free_pods() == 10 - 8                 # the 4 removed still hold their slots (grace)
    clock.t += 31
    sim.tick()
    assert sim.free_pods() >= 10 - 4
