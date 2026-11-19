"""
Regression tests for issue #37: the controller must honour the configured TTL (spec.ttlSeconds, FR-7) instead of
silently using the Knative idle window. A prewarmed tool stays warm for max(TTL, window): keep-alive pings hold it
while TTL > window, the last one at readyAt + TTL - window, so Knative scales it down at about readyAt + TTL.
"""

import threading

import pytest

pc = pytest.importorskip("pulse.controller.prewarm_controller")


def schedule(ttl, window, step=0.5):
    """Times of the keep-alive pings for a tool warmed at t = 0, and when it would scale down."""
    hold = {"until": ttl, "window": window, "last_ping": 0.0}
    pings, t = [], 0.0
    while t <= ttl + window:
        ping, last = pc.keepalive_due(hold, t)
        if ping:
            pings.append(t)
            hold["last_ping"] = t
            if last:
                break
        t += step
    last_activity = pings[-1] if pings else 0.0
    return pings, last_activity + window       # Knative: zero pods one idle window after the last request


@pytest.mark.parametrize("ttl", [30, 60, 120])
def test_scale_down_tracks_ttl_with_short_window(ttl):
    """The issue's setup: window 20 s, advertised TTL 30 / 60 / 120 s -> warm for about 30 / 60 / 120 s."""
    pings, down = schedule(ttl, window=20)
    assert abs(down - ttl) <= 0.5
    gaps = [b - a for a, b in zip([0.0] + pings, pings)]
    assert max(gaps) <= 20 / 2 + 0.5           # never a whole window without traffic before the TTL ends


def test_ttl_equal_to_window_needs_no_keepalive():
    """E1's setting (TTL 60, window 60): same behaviour as before, no extra pings."""
    assert pc.effective_ttl(60, 60) == 60
    hold = {"until": 60, "window": 60, "last_ping": 0.0}
    assert pc.keepalive_due(hold, 0.0) == (True, True)   # would be due at once -> warm() does not create a hold


def test_effective_ttl():
    assert pc.effective_ttl(120, 20) == 120
    assert pc.effective_ttl(30, 60) == 60      # the window is a floor
    assert pc.effective_ttl(None, 60) == 60
    assert pc.effective_ttl(120, 20, keepalive=False) == 20


class FakeController(pc.Controller):
    """Controller without a cluster: fixed pod count and window, records status patches and pings."""

    def __init__(self, window=20):
        self.in_flight, self.holds, self.lock = set(), {}, threading.Lock()
        self.status, self.pinged, self.logged, self._window = [], [], [], window

    def pods(self, svc):
        return 0

    def window_s(self, svc):
        return self._window

    def set_status(self, name, **status):
        self.status.append(status)

    def ping(self, svc, tool):
        self.pinged.append(svc)
        return 200, 5

    def log(self, **rec):
        self.logged.append(rec)


def pwr(ttl):
    return {"metadata": {"name": "get-user-details"},
            "spec": {"tool": "get_user_details", "service": "get-user-details", "ttlSeconds": ttl,
                     "desiredReplicas": 1, "requestedAt": pc.now_iso(), "probability": 0.8}}


def test_warm_holds_tool_for_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "LOG", tmp_path / "c.jsonl")
    c = FakeController(window=20)
    c.warm(pwr(120))
    ready = [s for s in c.status if s.get("readyAt")][0]
    assert c.status[0]["requestedTtlSeconds"] == 120 and c.status[0]["windowSeconds"] == 20
    assert c.status[0]["effectiveTtlSeconds"] == 120
    assert pc.parse_iso(ready["expiresAt"]) - pc.parse_iso(ready["readyAt"]) == pytest.approx(120, abs=0.01)
    hold = c.holds["get-user-details"]
    assert hold["until"] - hold["last_ping"] == pytest.approx(120, abs=0.01)
    assert c.logged[-1]["held"] is True


def test_warm_with_ttl_below_window_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "LOG", tmp_path / "c.jsonl")
    c = FakeController(window=60)
    c.warm(pwr(30))
    assert not c.holds                                   # nothing to hold: the window already keeps it 60 s
    assert c.status[0]["effectiveTtlSeconds"] == 60
    assert "below the 60 s idle window" in c.status[-1]["message"]


def test_keepalive_off_keeps_old_behaviour(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "LOG", tmp_path / "c.jsonl")
    monkeypatch.setattr(pc, "TTL_KEEPALIVE", False)
    c = FakeController(window=20)
    c.warm(pwr(120))
    assert not c.holds and c.status[0]["effectiveTtlSeconds"] == 20


def test_refresh_restarts_the_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "LOG", tmp_path / "c.jsonl")
    c = FakeController(window=20)
    c.warm(pwr(60))
    first = c.holds["get-user-details"]["until"]
    c.warm(pwr(60))
    assert c.holds["get-user-details"]["until"] >= first
