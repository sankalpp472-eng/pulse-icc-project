"""
Cold vs warm MCP call to each of the 14 airline tools on Knative.

For every tool: wait until it has no pods (scaled to zero), call it once (cold),
then again (warm). Calls go through the same MCP client the agent uses.

Run with the agent's Python (it needs the `mcp` package), from the project root:
  agent/.venv/Scripts/python deploy/knative/check_tools.py      # Windows
  agent/.venv/bin/python     deploy/knative/check_tools.py      # macOS / Linux

Results: deploy/knative/results/<date>_<time>_<config>_tools-cold-warm.csv
"""

import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "agent"))
os.environ.setdefault("TOOL_BACKEND", "mcp")
os.environ.setdefault("MCP_URL_TEMPLATE", "http://127.0.0.1:8081/mcp")
os.environ.setdefault("MCP_HOST_TEMPLATE", "{svc}.default.example.com")
from tools import make_tool_backend  # noqa: E402

R = "Z7GOZK"
ARGS = {  # valid-looking arguments; error replies are fine, the round trip is what we time
    "get_user_details": {"user_id": "mia_li_3668"},
    "get_reservation_details": {"reservation_id": R},
    "search_direct_flight": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
    "search_onestop_flight": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
    "list_all_airports": {},
    "calculate": {"expression": "2+2"},
    "think": {"thought": "check policy"},
    "send_certificate": {"user_id": "mia_li_3668", "amount": 50},
    "transfer_to_human_agents": {"summary": "test"},
    "cancel_reservation": {"reservation_id": R},
    "update_reservation_baggages": {"reservation_id": R, "total_baggages": 1, "nonfree_baggages": 0,
                                    "payment_id": "credit_card_1"},
    "update_reservation_flights": {"reservation_id": R, "cabin": "economy", "flights": [],
                                   "payment_id": "credit_card_1"},
    "update_reservation_passengers": {"reservation_id": R, "passengers": []},
    "book_reservation": {"user_id": "mia_li_3668"},
}


def n_pods(svc: str) -> int:
    out = subprocess.run(["kubectl", "get", "pods", "-l", f"serving.knative.dev/service={svc}", "--no-headers"],
                         capture_output=True, text=True).stdout
    return len([l for l in out.splitlines() if l.strip()])


def wait_for_zero(svc: str, timeout: float = 300) -> None:
    t0 = time.time()
    while n_pods(svc) and time.time() - t0 < timeout:
        time.sleep(2)


def main():
    b = make_tool_backend()
    rows = []
    print(f"{'tool':30s} {'cold':>9} {'warm':>8}  result")
    for name in b.names:
        svc = name.replace("_", "-")
        wait_for_zero(svc)
        if n_pods(svc):
            print(f"{name:30s} skipped: still has pods after 5 min")
            continue
        out1, cold = b.call(name, ARGS[name])
        out2, warm = b.call(name, ARGS[name])
        ok = "MCP call" not in out1 and "MCP call" not in out2   # transport errors, not tool errors
        rows.append({"tool": name, "cold_ms": round(cold * 1000, 1), "warm_ms": round(warm * 1000, 1),
                     "ok": ok, "result": out1[:80]})
        print(f"{name:30s} {cold * 1000:7.0f}ms {warm * 1000:6.0f}ms  {out1[:45]!r}")

    ok = [r for r in rows if r["ok"]]
    if ok:
        c, w = [r["cold_ms"] for r in ok], [r["warm_ms"] for r in ok]
        print(f"\n{len(ok)}/{len(rows)} tools OK | cold p50 {statistics.median(c):.0f} ms "
              f"(min {min(c):.0f}, max {max(c):.0f}) | warm p50 {statistics.median(w):.0f} ms")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{time.strftime('%Y-%m-%d_%H%M')}_C0_tools-cold-warm.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tool", "cold_ms", "warm_ms", "ok", "result"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
