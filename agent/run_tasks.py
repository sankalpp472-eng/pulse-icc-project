"""
Run the agent on tau-bench airline tasks and save traces.

  python run_tasks.py --tasks 0 1 2           # simulated customer
  python run_tasks.py --tasks 1 --impl langgraph   # LangGraph version of the agent
  python run_tasks.py --tasks 1 --tools mcp        # tools called on MCP servers (start mcp_server.py first)
  python run_tasks.py --tasks 5 --user human  # you play the customer
  python run_tasks.py --list                  # show task ids + gold tool sequences

Traces go to runs/<timestamp>.jsonl, one episode per line.
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tau_airline import load_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[0])
    p.add_argument("--user", choices=["sim", "human"], default="sim")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--impl", choices=["plain", "langgraph"], default="plain",
                   help="plain loop (agent.py) or LangGraph (langgraph_agent.py)")
    p.add_argument("--tools", choices=["local", "mcp"], default=None,
                   help="where tools run: in-process (local) or on MCP servers (mcp). Default: TOOL_BACKEND or local")
    p.add_argument("--list", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args()

    tasks = load_tasks()
    if a.list:
        for t in tasks:
            print(f"{t['id']:>2}  {' -> '.join(x['name'] for x in t['actions']) or '(no actions)'}")
        return

    if a.tools:
        os.environ["TOOL_BACKEND"] = a.tools
    tool_backend = os.getenv("TOOL_BACKEND", "local")
    import llm  # imported late so --list works without an API key
    if a.impl == "langgraph":
        from langgraph_agent import run_episode
    else:
        from agent import run_episode

    os.makedirs("runs", exist_ok=True)
    out_path = f"runs/{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"impl={a.impl} tools={tool_backend} backend={llm.BACKEND} model={llm.DEFAULT_MODEL} -> {out_path}")

    results = []
    for tid in a.tasks:
        print(f"\n{'=' * 70}\nTask {tid}")
        ep = run_episode(tasks[tid], user=a.user, max_steps=a.max_steps, verbose=not a.quiet)
        ep.update(impl=a.impl, tools=tool_backend, backend=llm.BACKEND, model=llm.DEFAULT_MODEL)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ep) + "\n")
        results.append(ep)
        thinks = [s["think_s"] for s in ep["steps"]]
        print(f"--> reward={ep['reward']} (db_ok={ep['db_ok']}, outputs_ok={ep['outputs_ok']}) "
              f"end={ep['end_reason']} steps={len(ep['steps'])} "
              f"mean_think={sum(thinks) / max(len(thinks), 1):.2f}s")
        print(f"    agent tools: {' -> '.join(ep['trajectory'])}")
        print(f"    gold actions: {' -> '.join(ep['gold_trajectory'])}")

    if len(results) > 1:
        ok = sum(r["reward"] or 0 for r in results)
        print(f"\nSolved {ok:.0f}/{len(results)} tasks")


if __name__ == "__main__":
    main()
