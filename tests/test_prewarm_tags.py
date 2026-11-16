"""
Regression tests for issue #21: "Prewarmed" counts calls served warm by a prewarmed pod (not calls to a tool the
predictor asked to prewarm), so Cold + Prewarmed never exceeds Calls, and "prewarm landed late" is said only of calls
a prewarm was actually meant for.
"""

from types import SimpleNamespace

from pulse.ui.session import Session, prewarm_tags


def tags(names, cold, asked):
    return [("hit" if t["prewarmed"] else "late" if t["prewarm_late"] else "-") for t in prewarm_tags(names, cold, asked)]


def test_issue_case_seven_parallel_calls_five_prewarmed():
    # LLM step 3 of task 30: "prewarm get_reservation_details x5", 7 parallel calls, 5 warm + 2 cold
    names = ["get_reservation_details"] * 7
    cold = [False, False, True, False, False, True, False]
    assert tags(names, cold, {"get_reservation_details": 5}) == ["hit", "hit", "-", "hit", "hit", "-", "hit"]


def test_prewarm_not_ready_in_time_is_late():
    assert tags(["get_user_details"], [True], {"get_user_details": 1}) == ["late"]


def test_more_prewarmed_than_calls_and_some_late():
    # 3 pods asked for, 2 calls: one warm (hit), one cold (a prewarm was meant for it: late)
    assert tags(["t", "t"], [True, False], {"t": 3}) == ["late", "hit"]


def test_tool_not_prewarmed_gets_no_tag():
    assert tags(["a", "b"], [False, True], {"c": 1}) == ["-", "-"]
    assert prewarm_tags(["a"], [False], {})[0]["prewarm_k"] == 0


class FakeTools:
    def __init__(self, cold):
        self.cold = list(cold)

    def run(self, name, args):
        c = self.cold.pop(0)
        return {"latency_s": 1.6 if c else 0.04, "cold": c, "wait_s": 1.5 if c else None, "out": "{}"}


def test_episode_summary_cold_plus_prewarmed_never_exceeds_calls():
    s = Session({"kind": "replay"}, cluster=SimpleNamespace(episode_started=lambda t: None, context="kind-test"), sim=None,
                predictor_url=None)
    s.prewarmed = {1: {"get_user_details": 1}, 3: {"get_reservation_details": 5}}
    # step 1: one call, the prewarm came too late (cold); step 3: the issue's 7 parallel calls
    s.run_calls(1, [("get_user_details", {})], FakeTools([True]), parallel=False)
    s.run_calls(3, [("get_reservation_details", {"reservation_id": str(i)}) for i in range(7)],
                FakeTools([False] * 5 + [True] * 2), parallel=False)
    summ = s.summary()
    assert (summ["calls"], summ["cold"], summ["prewarmed_hits"], summ["prewarm_late"]) == (8, 3, 5, 1)
    assert summ["cold"] + summ["prewarmed_hits"] <= summ["calls"]
    assert summ["prewarm_requested_calls"] == 8
    # the 2 extra cold lookups are not "late": no prewarm was asked for them
    late = [c for c in s.calls if c["prewarm_late"]]
    assert [c["tool"] for c in late] == ["get_user_details"]
