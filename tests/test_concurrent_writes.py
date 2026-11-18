"""
Regression tests for issue #31: two tool calls on different pods that read-modify-write the same record in the shared
Redis must not lose a write. The old code read records with HGET and wrote them back with a plain HSET, so the later
write silently overwrote the earlier one. Now a call's write commits only if nothing it read changed (WATCH + a
check of the values it read, then MULTI/EXEC), and a call that loses the race reruns on fresh data.

The race is made deterministic: the fake Redis lets a second call land at an exact moment inside the first one.
No cluster, Docker or LLM needed.
"""

import importlib
import json
import threading

import pytest
import redis

USER, CARD = "mei_hernandez_8984", "gift_card_5309492"     # a gift card with $98 and two reservations
RES_A, RES_B = "R9QDGB", "07S7FC"


class FakePipe:
    """redis-py's pipeline as mcp_server uses it: WATCH (then immediate commands), MULTI, queued HSET, EXEC."""

    def __init__(self, r):
        self.r, self.ops, self.watched, self.queued = r, [], None, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.watched = None

    def watch(self, *keys):
        self.watched = {k: self.r.version.get(k, 0) for k in keys}
        self.r.fire("watch")

    def hmget(self, name, keys):
        out = [self.r.hget(name, k) for k in keys]
        self.r.fire("checked")
        return out

    def multi(self):
        self.queued = True

    def delete(self, *keys):
        self.ops.append(("delete", keys, {}))

    def hset(self, *a, **kw):
        self.ops.append(("hset", a, kw))

    def execute(self):
        with self.r.lock:                    # EXEC is atomic: check the watched keys and apply, as one step
            if self.watched and any(self.r.version.get(k, 0) != v for k, v in self.watched.items()):
                self.ops = []
                raise redis.exceptions.WatchError("Watched variable changed.")
            for op, a, kw in self.ops:
                getattr(self.r, op)(*a, **kw)
        self.ops, self.watched = [], None
        return []


class FakeRedis:
    """An in-process Redis with per-key versions (for WATCH) and hooks to run another call at a chosen moment."""

    def __init__(self):
        self.h, self.kv, self.version, self.hooks, self.watches = {}, {}, {}, {}, 0
        self.lock = threading.RLock()
        self.after_read = None               # called as after_read(table, key) by every HGET (thread tests)

    def fire(self, when):
        if when == "watch":
            self.watches += 1
        hook = self.hooks.pop(when, None)
        if hook:
            hook()

    def _bump(self, name):
        self.version[name] = self.version.get(name, 0) + 1

    def hget(self, name, key):
        with self.lock:
            out = self.h.get(name, {}).get(key)
        if self.after_read:                  # the value is already read: a barrier here leaves it stale
            self.after_read(name, key)
        if name == "users":
            self.fire("read")
        return out

    def hexists(self, name, key):
        return key in self.h.get(name, {})

    def hgetall(self, name):
        return dict(self.h.get(name, {}))

    def hlen(self, name):
        return len(self.h.get(name, {}))

    def hset(self, name, key=None, value=None, mapping=None):
        with self.lock:
            d = self.h.setdefault(name, {})
            if key is not None:
                d[key] = value
            d.update(mapping or {})
            self._bump(name)

    def delete(self, *keys):
        for k in keys:
            self.h.pop(k, None)
            self.kv.pop(k, None)
            self._bump(k)

    def exists(self, *keys):
        return sum(1 for k in keys if self.h.get(k) or k in self.kv)

    def set(self, k, v, nx=False, ex=None):
        if nx and k in self.kv:
            return None
        self.kv[k] = v
        return True

    def pipeline(self, transaction=True):
        return FakePipe(self)


@pytest.fixture
def server(monkeypatch):
    """mcp_server imported as a tool pod on the shared Redis (the fake, seeded with the airline data)."""
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("REDIS_TIMEOUT_S", "0.5")
    monkeypatch.delenv("DOMAIN", raising=False)
    import pulse.agent.mcp_server as m
    m = importlib.reload(m)
    m._R = FakeRedis()
    m.ensure_seeded(m._R)
    yield m
    monkeypatch.delenv("REDIS_URL")
    importlib.reload(m)


def add_bag(server, reservation):
    """One more paid bag ($50) on the gift card: a read-modify-write of the user and the reservation."""
    res = json.loads(server._R.hget("reservations", reservation))
    return server.invoke_tool(server.TOOLS["update_reservation_baggages"], {
        "reservation_id": reservation, "total_baggages": res["total_baggages"] + 1,
        "nonfree_baggages": res["nonfree_baggages"] + 1, "payment_id": CARD})


def balance(server):
    return json.loads(server._R.hget("users", USER))["payment_methods"][CARD]["amount"]


@pytest.mark.parametrize("moment", ["read", "watch", "checked"])
def test_concurrent_calls_do_not_lose_a_write(server, moment):
    """$98 on the card, two $50 bags booked at once on two pods: one must be refused, not both charged.
    The other call lands right after this one read the user ('read', the old lost update), before this one checks
    what it read ('watch'), or after that check but before EXEC ('checked')."""
    assert balance(server) == 98
    other = {}
    json.loads(server._R.hget("users", USER))                       # (the test's own read fires no hook yet)
    server._R.hooks[moment] = lambda: other.setdefault("out", add_bag(server, RES_B))
    out_a = add_bag(server, RES_A)
    assert not other["out"].startswith("Error"), other["out"]       # the other call got the card first ...
    assert out_a == "Error: gift card balance is not enough"         # ... so this one, rerun on fresh data, is refused
    assert balance(server) == 48                                     # charged once, not twice and not lost
    assert json.loads(server._R.hget("reservations", RES_A))["nonfree_baggages"] == 0


def test_sequential_calls_are_unchanged(server):
    assert not add_bag(server, RES_A).startswith("Error")
    assert balance(server) == 48
    assert add_bag(server, RES_B) == "Error: gift card balance is not enough"


def test_read_only_calls_take_no_lock(server):
    out = server.invoke_tool(server.TOOLS["get_user_details"], {"user_id": USER})
    assert USER in out or "Mei" in out
    assert server._R.watches == 0


def test_a_record_created_meanwhile_is_a_conflict(server):
    """A call that adds a record must not overwrite one with the same key that another call added meanwhile."""
    r, t = server._R, server.RedisTable(server._R, "reservations")
    t["NEWRES"] = {"by": "this call"}
    r.hset("reservations", "NEWRES", json.dumps({"by": "another call"}))
    with pytest.raises(server.WriteConflict):
        server.commit(r, [t])
    assert json.loads(r.hget("reservations", "NEWRES")) == {"by": "another call"}


def test_gives_up_with_a_clear_error_when_it_never_wins(server, monkeypatch):
    monkeypatch.setattr(server, "WRITE_RETRIES", 2)

    def always_conflict(r, tables):
        raise server.WriteConflict("users changed during the call")

    monkeypatch.setattr(server, "commit", always_conflict)
    with pytest.raises(server.WriteConflict, match="gave up after 3 attempts"):
        add_bag(server, RES_A)
    assert balance(server) == 98


def test_issue_reproduction_threads_with_both_reads_before_either_write(server):
    """The issue's own scenario: two calls change DIFFERENT fields of the SAME reservation, on two threads, and a
    barrier holds each call right after its first read of it until both have read it. 100 rounds: neither change may be lost."""
    r = server._R
    pristine = r.hget("reservations", RES_A)
    gate, mine = threading.Barrier(2), threading.local()

    def hold_first_read(table, key):         # each thread, having read the record, waits until the other has too
        if table == "reservations" and key == RES_A and getattr(mine, "armed", False):
            mine.armed = False
            gate.wait(timeout=10)

    r.after_read = hold_first_read
    passengers = json.loads(pristine)["passengers"]
    lost, errors = 0, []
    for i in range(100):
        r.hset("reservations", RES_A, pristine)
        renamed = [{**passengers[0], "first_name": f"Round{i}"}, *passengers[1:]]
        calls = [("update_reservation_baggages", {"reservation_id": RES_A, "total_baggages": 5,
                                                  "nonfree_baggages": 0, "payment_id": "credit_card_2140654"}),
                 ("update_reservation_passengers", {"reservation_id": RES_A, "passengers": renamed})]

        def run(tool, args):
            mine.armed = True
            try:
                out = server.invoke_tool(server.TOOLS[tool], args)
                if out.startswith("Error"):
                    errors.append(out)
            except Exception as e:           # noqa: BLE001 (reported below)
                errors.append(repr(e))

        threads = [threading.Thread(target=run, args=c) for c in calls]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        final = json.loads(r.hget("reservations", RES_A))
        if final["total_baggages"] != 5 or final["passengers"][0]["first_name"] != f"Round{i}":
            lost += 1
    r.after_read = None
    assert errors == []
    assert lost == 0, f"rounds with a lost write: {lost}/100"
