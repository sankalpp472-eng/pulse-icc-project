"""
Per-tool cold-start latency (inside the cluster): for every tool of the domain, a cold call (the tool at zero
pods) and right after it a warm call, repeated --reps times. Each round first waits until every tool is back at
zero; the tools are then called one at a time in a seeded random order, so a cold call never shares its
start-up with another tool's. The new pod of each cold call is matched to its start-up breakdown
(scheduling, image pull, init, readiness: common.startup_breakdown).

    DOMAIN=retail python -m pulse.experiments.coldstart --reps 5            # in a run pod: MODE=coldstart run_cloud.sh

Unlike pulse/agent/check_tools.py (laptop -> load balancer -> tool) the calls go pod -> tool, so the time is the
tool's, not the internet's. Arguments: valid-looking ones for airline (check_tools.ARGS), recorded read-only calls
for retail; an error reply is fine, the round trip is what is timed. The Redis data is reset before each round
and at the end (some airline write tools change it; send_certificate has only 3 slots per user).

Writes a run folder (common.run_dir): coldstart.csv (one row per round and tool), pods.csv, summary.json
(per tool: median cold / warm ms and start-up breakdown), run_config.json.
"""

import argparse
import csv
import json
import os
import random
import statistics
import time

os.environ.setdefault("TOOL_BACKEND", "mcp")
os.environ.setdefault("MCP_HOST_TEMPLATE", "")       # in the cluster the URL names the service: no Host header

from pulse.agent.check_tools import ARGS as AIRLINE_ARGS  # noqa: E402
from pulse.agent.tools import CALL_LOG, make_tool_backend  # noqa: E402
from pulse.experiments import common  # noqa: E402
from pulse.experiments.common import DOMAIN, RETAIL, PodWatcher, run_dir, startup_breakdown, wait_all_zero  # noqa: E402

FIELDS = ["round", "tool", "cold_ms", "warm_ms", "ok", "pod", "scheduling_s", "image_pull_s", "init_s",
          "readiness_s", "startup_s", "result", "warm_result"]


def tool_args(names: list[str]) -> dict[str, dict]:
    if not RETAIL:
        return {n: AIRLINE_ARGS.get(n, {}) for n in names}
    try:                                  # retail: the first recorded call of each read-only tool
        import pulse.loadtest.common as lc
        seen = {}
        for tool, args in lc.read_only_calls(lc.episodes(252, None)):
            seen.setdefault(tool, args)
    except Exception as e:                 # no recorded conversations in the image: empty arguments
        print(f"no recorded retail calls ({type(e).__name__}: {e}); calling with empty arguments")
        seen = {}
    return {n: seen.get(n, {}) for n in names}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=5, help="rounds: one cold + one warm call per tool each")
    ap.add_argument("--tools", nargs="+", help="only these tools (default: all of the domain's)")
    ap.add_argument("--seed", type=int, default=252)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    backend = make_tool_backend()
    names = a.tools or backend.names
    args = tool_args(names)
    out = run_dir("C0", "coldstart", f"x{a.reps}", a.label)
    print(f"cold start | {DOMAIN} | {len(names)} tools x {a.reps} rounds -> {out}", flush=True)
    (out / "run_config.json").write_text(json.dumps(
        common.run_config([], "redis", common.WORKLOAD, {}, {"experiment": "coldstart", "reps": a.reps,
                                                          "tools": names, "seed": a.seed}), indent=2, default=str))

    watcher = PodWatcher()
    watcher.start()
    rng = random.Random(a.seed)
    rows, t_start = [], time.time()
    for rnd in range(1, a.reps + 1):
        wait_all_zero(watcher)
        try:      # each round starts from the seed data (send_certificate fails after 3 certificates per user)
            common.reset_db()
        except Exception as e:
            print(f"could not reset the database: {type(e).__name__}: {e}")
        order = list(names)
        rng.shuffle(order)
        for name in order:
            t0 = time.time()
            out1, cold = backend.call(name, args[name])
            t1 = time.time()
            out2, warm = backend.call(name, args[name])
            # transport errors (no answer, connection refused, 5xx) fail a round; a tool's own error reply does not
            ok = all(c.get("transport_ok", True) for c in CALL_LOG[-2:]) if len(CALL_LOG) >= 2 else                 "MCP call" not in out1 and "MCP call" not in out2
            rows.append({"round": rnd, "tool": name, "cold_ms": round(cold * 1000, 1),
                         "warm_ms": round(warm * 1000, 1), "ok": ok,
                         "pod": watcher.new_pod_during(name, t0, t1), "result": out1[:80],
                         "warm_result": out2[:160]})
            print(f"  round {rnd} {name:30s} cold {cold * 1000:7.0f} ms  warm {warm * 1000:6.0f} ms"
                  + ("" if ok else f"  FAILED: {(out1 if 'MCP call' in out1 else out2)[:200]}"), flush=True)

    time.sleep(3)                                     # the last pods' Ready timestamps
    pods = watcher.pod_rows()
    watcher.stop_flag = True
    by_pod = {p["pod"]: startup_breakdown(p) for p in pods}
    for r in rows:
        r.update(by_pod.get(r["pod"]) or {})
    with open(out / "coldstart.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "pods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=common.POD_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(pods)

    per_tool = {}
    for name in names:
        rs = [r for r in rows if r["tool"] == name and r["ok"]]
        per_tool[name] = {"rounds": len(rs), "cold_ms": med([r["cold_ms"] for r in rs]),
                          "warm_ms": med([r["warm_ms"] for r in rs]),
                          **{k: med([r.get(k) for r in rs]) for k in ("scheduling_s", "image_pull_s", "init_s",
                                                                     "readiness_s", "startup_s")}}
    ok = [r for r in rows if r["ok"]]
    summary = {"domain": DOMAIN, "rounds": a.reps, "tools": len(names), "calls_ok": len(ok), "calls": len(rows),
               "cold_ms": {"median": med([r["cold_ms"] for r in ok]),
                           "p95": round(sorted(r["cold_ms"] for r in ok)[int(0.95 * (len(ok) - 1))], 1) if ok else None},
               "warm_ms": {"median": med([r["warm_ms"] for r in ok])},
               "startup": common.startup_summary(pods, t_start), "per_tool": per_tool,
               "duration_s": round(time.time() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        common.reset_db()                    # airline write tools may have changed reservations
    except Exception as e:
        print(f"could not reset the database: {type(e).__name__}: {e}")
    print(f"\n{len(ok)}/{len(rows)} calls OK | cold median {summary['cold_ms']['median']} ms | "
          f"warm median {summary['warm_ms']['median']} ms | {out}")
    # a tool with no good round = a broken run (e.g. the cluster out of memory): fail, so the script repeats it
    broken = [t for t, v in per_tool.items() if not v["rounds"]]
    if broken:
        print(f"FAILED: no successful call to {len(broken)} tool(s): {', '.join(broken)}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
