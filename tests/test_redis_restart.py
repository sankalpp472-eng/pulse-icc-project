"""
Regression tests for issue #25: after a Redis restart the tools must not crash-loop on an empty (or unreachable)
database, must reseed it themselves, and while Redis is down a tool call must fail fast with an error that names
the cause (not hang for Knative's 300 s, not "ExceptionGroup: unhandled errors in a TaskGroup").

No cluster, Docker or LLM needed: Redis is an in-process fake.
"""

import asyncio
import importlib
import time
from types import SimpleNamespace

import pytest
import redis

USER = "mia_li_3668"
MIA = '"first_name": "Mia"'


class FakePipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def delete(self, *keys):
        self.ops.append(("delete", keys, {}))

    def hset(self, *a, **kw):
        self.ops.append(("hset", a, kw))

    def execute(self):
        for op, a, kw in self.ops:
            getattr(self.r, op)(*a, **kw)
        self.ops = []


class FakeRedis:
    """The hash / key commands mcp_server uses; down=True makes every command fail like a stopped Redis."""

    def __init__(self):
        self.h, self.kv, self.down = {}, {}, False

    def _up(self):
        if self.down:
            raise redis.exceptions.ConnectionError("Error 111 connecting to redis:6379. Connection refused.")

    def hget(self, name, key):
        self._up()
        return self.h.get(name, {}).get(key)

    def hexists(self, name, key):
        self._up()
        return key in self.h.get(name, {})

    def hgetall(self, name):
        self._up()
        return dict(self.h.get(name, {}))

    def hlen(self, name):
        self._up()
        return len(self.h.get(name, {}))

    def hset(self, name, key=None, value=None, mapping=None):
        self._up()
        d = self.h.setdefault(name, {})
        if key is not None:
            d[key] = value
        d.update(mapping or {})

    def delete(self, *keys):
        self._up()
        for k in keys:
            self.h.pop(k, None)
            self.kv.pop(k, None)

    def exists(self, *keys):
        self._up()
        return sum(1 for k in keys if self.h.get(k) or k in self.kv)

    def set(self, k, v, nx=False, ex=None):
        self._up()
        if nx and k in self.kv:
            return None
        self.kv[k] = v
        return True

    def pipeline(self, transaction=True):
        self._up()
        return FakePipe(self)

    def dbsize(self):
        return len([k for k in self.h if self.h[k]]) + len(self.kv)


@pytest.fixture
def server(monkeypatch):
    """mcp_server imported as a tool pod with Redis unreachable (nothing listens on port 1)."""
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("REDIS_TIMEOUT_S", "0.5")
    monkeypatch.delenv("DOMAIN", raising=False)
    import pulse.agent.mcp_server as m
    m = importlib.reload(m)            # must not raise SystemExit (the old code did -> CrashLoopBackOff)
    yield m
    monkeypatch.delenv("REDIS_URL")
    importlib.reload(m)


def test_pod_starts_and_calls_fail_fast_when_redis_is_down(server):
    t0 = time.monotonic()
    with pytest.raises(server.DatabaseUnavailable, match=r"Redis at redis://127.0.0.1:1/0.*(ConnectionError|TimeoutError)"):
        server.invoke_tool(server.TOOLS["get_user_details"], {"user_id": USER})
    assert time.monotonic() - t0 < 10


def test_empty_redis_is_reseeded_by_the_first_call(server):
    fake = server._R = FakeRedis()                      # Redis came back empty
    out = server.invoke_tool(server.TOOLS["get_user_details"], {"user_id": USER})
    assert MIA in out
    for t in server.TABLES:
        assert fake.hlen(t) > 0 and fake.hlen(f"seed:{t}") == fake.hlen(t)
    assert "seed:lock" not in fake.kv


def test_recovers_after_restart_mid_life(server):
    fake = server._R = FakeRedis()
    server.invoke_tool(server.TOOLS["get_user_details"], {"user_id": USER})
    fake.down = True                                    # Redis stopped
    with pytest.raises(server.DatabaseUnavailable, match="ConnectionError"):
        server.invoke_tool(server.TOOLS["get_user_details"], {"user_id": USER})
    fake.down, fake.h, fake.kv = False, {}, {}          # ... and back, empty
    assert MIA in server.invoke_tool(server.TOOLS["get_user_details"], {"user_id": USER})
    assert fake.dbsize() == 2 * len(server.TABLES)


def test_reset_on_an_empty_redis_reseeds(server):
    fake = server._R = FakeRedis()
    server.reset_redis(fake)
    assert all(fake.hlen(t) for t in server.SHARED)


def test_reset_restores_written_tables(server):
    fake = server._R = FakeRedis()
    server.ensure_seeded(fake)
    fake.hset("users", USER, '"changed"')
    server.reset_redis(fake)
    assert fake.hget("users", USER) == fake.hget("seed:users", USER)


def test_waits_for_another_pod_that_is_seeding(server):
    fake = FakeRedis()
    fake.kv["seed:lock"] = "other-pod"
    with pytest.raises(server.DatabaseUnavailable, match="being reseeded"):
        server.ensure_seeded(fake, wait_s=0.3)


# ---------------------------------------------------------------------- the client names the cause
def test_exception_group_is_unwrapped():
    from pulse.agent.tools import describe_error
    e = ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("Connection refused")])
    assert describe_error(e) == "ConnectionError: Connection refused"
    assert describe_error(ValueError("x")) == "ValueError: x"


def test_client_call_times_out(monkeypatch):
    from pulse.agent.tools import MCPToolBackend
    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "0.2")
    b = MCPToolBackend()

    async def hang(name, args):
        await asyncio.sleep(10)

    monkeypatch.setattr(b, "_call", hang)
    t0 = time.monotonic()
    out, _ = b.call("get_user_details", {"user_id": USER})
    assert time.monotonic() - t0 < 5
    assert out.startswith("Error: MCP call to get_user_details failed: no answer within")


# ---------------------------------------------------------------------- the UI's tool wait moves during a call
def test_live_tool_wait_counts_the_call_in_flight():
    from pulse.ui.session import Session
    waiting = SimpleNamespace(tool_wait=1.0, step_t0=time.time() - 2.0)
    assert Session.live_tool_wait(waiting) >= 3.0
    assert Session.live_tool_wait(SimpleNamespace(tool_wait=1.0, step_t0=None)) == 1.0
