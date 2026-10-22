"""
Score the predictor's live answers against what the agent actually did next.

The agent sends hint k right before LLM step k (agent/hints.py). If step k called a tool,
that tool is the "true next tool" for hint k; if the agent answered the customer instead,
nothing was called and any prewarm for that step was wasted.

  python test/test_knative/score_hints.py <hints.jsonl> <agent episode .jsonl> [--out report.json]

Per tool step:  top-1 hit, top-3 hit, prewarm hit (true tool was in the prewarm list, p >= theta)
Per step:       how many tools were prewarmed and how many of them were not used (wasted)
Lead time:      LLM think time after the hint = how long a prewarm had to finish before the call
"""

import argparse
import json
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hints")
    ap.add_argument("episodes")
    ap.add_argument("--out")
    a = ap.parse_args()

    hints = defaultdict(list)
    for line in open(a.hints, encoding="utf-8"):
        r = json.loads(line)
        hints[r["hint"]["episode_id"]].append(r)
    for v in hints.values():
        v.sort(key=lambda r: r["hint"]["ts"])           # hints are sent from threads: order by send time

    rows = []
    for line in open(a.episodes, encoding="utf-8"):
        ep = json.loads(line)
        hs = hints.get(ep["episode_id"], [])
        for i, st in enumerate(ep["steps"]):
            h = hs[i] if i < len(hs) else None
            true_tool = st["calls"][0]["tool"] if st["kind"] == "tool" and st.get("calls") else None
            ranked = [x["tool"] for x in (h or {}).get("top3", [])]
            prewarm = [x["tool"] for x in (h or {}).get("prewarm", [])]
            rows.append({"step": st["step"], "trajectory_len": len((h or {}).get("hint", {}).get("trajectory", [])),
                         "true_next": true_tool or "(answered customer)", "top3": ranked, "prewarm": prewarm,
                         "hint_ok": bool(h and h.get("ok")), "rtt_ms": (h or {}).get("rtt_ms"),
                         "top1_hit": bool(true_tool and ranked[:1] == [true_tool]),
                         "top3_hit": bool(true_tool and true_tool in ranked),
                         "prewarm_hit": bool(true_tool and true_tool in prewarm),
                         "wasted": len([t for t in prewarm if t != true_tool]),
                         "think_s": st["think_s"], "rate_wait_s": st.get("rate_wait_s", 0)})

    tool = [r for r in rows if r["true_next"] != "(answered customer)"]
    rate = lambda k: round(sum(r[k] for r in tool) / len(tool), 3) if tool else None
    summary = {
        "hints_sent": sum(len(v) for v in hints.values()), "hints_answered": sum(r["hint_ok"] for r in rows),
        "llm_steps": len(rows), "tool_steps": len(tool),
        "top1_accuracy": rate("top1_hit"), "top3_accuracy": rate("top3_hit"), "prewarm_hit_rate": rate("prewarm_hit"),
        "prewarmed_per_step": round(sum(len(r["prewarm"]) for r in rows) / max(1, len(rows)), 2),
        "wasted_per_step": round(sum(r["wasted"] for r in rows) / max(1, len(rows)), 2),
        "hint_rtt_ms_median": statistics.median([r["rtt_ms"] for r in rows if r["rtt_ms"]]) if rows else None,
        "lead_time_s_median (LLM think after hint)": statistics.median([r["think_s"] for r in tool]) if tool else None,
    }
    print(f"{'step':>4}  {'true next tool':28s} {'top-3 predicted':72s} prewarm")
    for r in rows:
        mark = "HIT " if r["prewarm_hit"] else ("top3" if r["top3_hit"] else "    ")
        print(f"{r['step']:>4}  {r['true_next']:28s} {', '.join(r['top3']):72s} {mark} {r['prewarm']}")
    print("\n" + json.dumps(summary, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "steps": rows}, f, indent=2)


if __name__ == "__main__":
    main()
