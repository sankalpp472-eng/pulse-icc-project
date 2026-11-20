"""
Keep-alive holds (#37) and the pod-lifetime levers (pulse/controller/refresh.py): a keep-alive ping counts as a
refresh, so REFRESH_POLICY=never|value and pressure shedding (SHED_P) end a hold early. With the defaults
(always, no shedding) every keep-alive is sent and the cluster is not even asked for its capacity.
Pure logic: a controller without a cluster.
"""

import threading
import time

import pytest

pc = pytest.importorskip("pulse.controller.prewarm_controller")
from pulse.controller.refresh import RefreshConfig  # noqa: E402


class FakeController(pc.Controller):
    """No cluster: a fixed pod count and capacity; records pings, status patches and log lines."""

    def __init__(self, pods=1, free=10, pending=0):
        self.in_flight, self.holds, self.lock = set(), {}, threading.Lock()
        self.pinged, self.status, self.logged = [], [], []
        self._pods, self._room, self.capacity_calls = pods, (free, pending), 0

    def pods(self, svc):
        return self._pods

    def capacity(self):
        self.capacity_calls += 1
        return self._room

    def ping(self, svc, tool):
        self.pinged.append(svc)
        return 200, 5

    def set_status(self, name, **status):
        self.status.append(status)

    def log(self, **rec):
        self.logged.append(rec)


def hold(c, p=0.5):
    h = {"name": "pwr-x", "tool": "get_user_details", "until": time.time() + 120, "window": 20,
         "last_ping": time.time(), "pings": 1, "probability": p}
    c.holds["get-user-details"] = h
    return h


def test_the_default_keeps_every_keepalive(monkeypatch):
    monkeypatch.setattr(pc, "REFRESH", RefreshConfig())          # always, no shedding
    c = FakeController()
    h = hold(c)
    c._keepalive_ping("get-user-details", h, last=False)
    assert c.pinged == ["get-user-details"] and c.holds and c.capacity_calls == 0
    assert c.logged[-1]["event"] == "keepalive"


def test_refresh_never_ends_the_hold_without_a_ping(monkeypatch):
    monkeypatch.setattr(pc, "REFRESH", RefreshConfig(policy="never"))
    c = FakeController(pods=1)
    h = hold(c)
    c._keepalive_ping("get-user-details", h, last=False)
    assert c.pinged == [] and not c.holds
    assert c.logged[-1]["event"] == "keepalive_skipped" and c.logged[-1]["why"] == "refresh"
    assert "released early" in c.status[-1]["message"]


def test_refresh_value_keeps_a_likely_tool_on_a_full_node_and_drops_an_unlikely_one(monkeypatch):
    monkeypatch.setattr(pc, "REFRESH", RefreshConfig(policy="value"))   # threshold 0.8 with no free slot
    c = FakeController(pods=1, free=0)
    keep = hold(c, p=0.9)
    c._keepalive_ping("get-user-details", keep, last=False)
    assert c.pinged == ["get-user-details"] and c.holds
    drop = hold(c, p=0.3)
    c._keepalive_ping("get-user-details", drop, last=False)
    assert c.pinged == ["get-user-details"] and not c.holds


def test_shedding_ends_a_low_value_hold_while_pods_wait_for_a_slot(monkeypatch):
    monkeypatch.setattr(pc, "REFRESH", RefreshConfig(shed_p=0.6))
    c = FakeController(pods=1, free=0, pending=2)
    h = hold(c, p=0.3)
    c._keepalive_ping("get-user-details", h, last=False)
    assert c.pinged == [] and not c.holds and c.logged[-1]["why"] == "pressure"
    c2 = FakeController(pods=1, free=0, pending=2)
    h2 = hold(c2, p=0.7)                                         # likely enough: still held
    c2._keepalive_ping("get-user-details", h2, last=False)
    assert c2.pinged == ["get-user-details"] and c2.holds


def test_an_early_release_does_not_drop_a_newer_hold(monkeypatch):
    monkeypatch.setattr(pc, "REFRESH", RefreshConfig(policy="never"))
    c = FakeController(pods=1)
    old = hold(c)
    newer = hold(c)                       # a new prediction replaced the hold meanwhile
    c._keepalive_ping("get-user-details", old, last=False)
    assert c.holds["get-user-details"] is newer
