"""
Regression tests for issue #27: a tool call that did not run (tool unreachable, timed out, its database down) is
counted as failed, not as cold; the episode summary has a failed count, and latency / cold share are over the calls
that ran, so a run where every call failed no longer looks like a run with cold starts.

No cluster or LLM needed: the Knative tool backend's MCP call is faked.
"""

from types import SimpleNamespace

import pytest

from pulse.agent.tools import call_failed
from pulse.ui.session import KnativeTools, Session, prewarm_tags

DOWN = ("Error: DatabaseUnavailable: the shared database (Redis at redis://redis:6379/0) is unavailable: "
        "ConnectionError: Error 111")


# ---------------------------------------------------------------------- what counts as failed
@pytest.mark.parametrize("out, failed", [
    ("Error: MCP call to get_user_details failed: no answer within 120 s", True),
    ("Error: MCP call to get_user_details failed: ExceptionGroup: unhandled errors in a TaskGroup", True),
    (DOWN, True),
    ("Error: user not found", False),          # the tool ran and answered: a normal result
    ('{"name": "Mia"}', False),
])
def test_call_failed(out, failed):
    assert call_failed(out) is failed


def knative_result(monkeypatch, out, latency):
    async def acall(name, args):
        return out, latency, 0.0
    tools = KnativeTools.__new__(KnativeTools)
    tools.mcp = SimpleNamespace(acall=acall)
    return tools.run("get_user_details", {})


def test_slow_failed_call_is_not_cold(monkeypatch):
    r = knative_result(monkeypatch, "Error: MCP call to get_user_details failed: no answer within 120 s", 120.0)
    assert r["failed"] and not r["cold"]


def test_slow_successful_call_is_cold(monkeypatch):
    r = knative_result(monkeypatch, '{"name": "Mia"}', 5.0)
    assert r["cold"] and not r["failed"]


# ---------------------------------------------------------------------- the summary
def summary(calls):
    s = Session.__new__(Session)
    s.calls, s.sim, s.hints = calls, None, {"sent": 0, "answered": 0, "rtt_ms": []}
    s.t0, s.t_start, s.t_end, s.tool_wait, s.llm_wait, s.state = 0.0, None, None, 0.0, 0.0, "done"
    s.cfg, s.cluster, s.domain = {"kind": "replay"}, SimpleNamespace(context="kind-pulse-kn"), "airline"
    return s.summary()


def call(latency_ms, cold=False, failed=False, prewarmed=False):
    return {"latency_ms": latency_ms, "cold": cold, "failed": failed, "prewarmed": prewarmed}


def test_all_failed_run_is_not_a_cold_run():
    s = summary([call(300_000, failed=True) for _ in range(9)])
    assert s["calls"] == 9 and s["failed"] == 9
    assert s["cold"] == 0 and s["cold_share"] is None and s["mean_ms"] is None and s["cold_wait_s"] == 0


def test_mixed_run_counts_latency_over_calls_that_ran():
    s = summary([call(100), call(3000, cold=True), call(300_000, failed=True)])
    assert (s["calls"], s["failed"], s["cold"]) == (3, 1, 1)
    assert s["cold_share"] == 0.5 and s["mean_ms"] == 1550.0


# ---------------------------------------------------------------------- prewarm tags
def test_failed_call_is_neither_prewarm_hit_nor_late():
    names = ["get_reservation_details"] * 3
    tags = prewarm_tags(names, cold=[False, False, False], asked={"get_reservation_details": 2},
                        failed=[True, False, False])
    assert [t["prewarmed"] for t in tags] == [False, True, True]
    assert not any(t["prewarm_late"] for t in tags)
