"""
ReAct-style agent for the tau-bench airline domain.

Loop (one tool at a time):
    hint -> LLM reasons -> tool call -> observation -> LLM reasons -> ...
    and when the LLM replies in plain text, the (simulated) user answers.

Every LLM step is recorded with its think time, which is the window the
prewarmer will have once hints are wired to the predictor.
"""

import json
import time

from hints import send_hint
from llm import chat
from prompts import AGENT_SYSTEM_PROMPT
from tools import TERMINATE_TOOLS, LocalToolBackend, make_tool_backend
from user import STOP, HumanUser, SimulatedUser


def emit_hint(episode_id: str, trajectory: list[str]) -> None:
    """Called right before every LLM step. Sends {workflow id, current tool,
    recent trajectory} to the predictor in the background (see hints.py).
    Does nothing when PREDICTOR_URL is not set."""
    send_hint(episode_id, trajectory)

def _assistant_msg(m) -> dict:
    msg = {"role": "assistant", "content": m.content or ""}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in m.tool_calls
        ]
    return msg


def run_episode(task: dict, user: str = "sim", max_steps: int = 30,
                verbose: bool = True) -> dict:
    episode_id = f"task{task['id']}-{int(time.time())}"
    tools = make_tool_backend()   # local or MCP (TOOL_BACKEND)
    customer = SimulatedUser(task["instruction"]) if user == "sim" else HumanUser(task["instruction"])

    t_start = time.perf_counter()
    first = customer.start()
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": first}]
    trajectory: list[str] = []   # tool names, in call order
    steps: list[dict] = []       # one record per LLM step (the trace)
    agent_texts: list[str] = []  # what the agent said to the user
    end_reason = "max_steps"
    if verbose:
        print(f"\n[user]  {first}")

    for step in range(1, max_steps + 1):
        emit_hint(episode_id, trajectory)
        r = chat(messages, tools=tools.schemas)
        m = r.message
        messages.append(_assistant_msg(m))
        rec = {"step": step, "t": round(time.perf_counter() - t_start, 3),
               "think_s": round(r.latency_s, 3), "rate_wait_s": round(r.rate_wait_s, 1)}

        if m.tool_calls:
            # Native tool calling. Models occasionally emit >1 call; run them in order.
            calls = []
            for tc in m.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    obs, tool_s = tools.call(name, args)
                except json.JSONDecodeError:
                    args, obs, tool_s = {}, "Error: arguments were not valid JSON", 0.0
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
                trajectory.append(name)
                calls.append({"tool": name, "args": args, "tool_s": round(tool_s, 4)})
                if verbose:
                    print(f"  [{step:>2}] think {r.latency_s:5.2f}s -> {name}({json.dumps(args)})")
                    print(f"       obs: {obs[:200]}{'...' if len(obs) > 200 else ''}")
            steps.append({**rec, "kind": "tool", "calls": calls})
            if any(c["tool"] in TERMINATE_TOOLS for c in calls):
                end_reason = "transfer"
                break
        else:
            text = m.content or ""
            agent_texts.append(text)
            steps.append({**rec, "kind": "respond"})
            if verbose:
                print(f"  [{step:>2}] think {r.latency_s:5.2f}s -> respond")
                print(f"[agent] {text}")
            reply = customer.reply(text)
            if verbose:
                print(f"[user]  {reply}")
            # End only on a standalone STOP. Small sim models often write
            # "Yes, cancel it ###STOP###" -- the agent must still see the "yes".
            reply = reply.replace(STOP, "").strip() if reply.strip() != STOP else STOP
            if reply == STOP:
                end_reason = "user_stop"
                break
            messages.append({"role": "user", "content": reply})

    return {
        "episode_id": episode_id,
        "task_id": task["id"],
        "end_reason": end_reason,
        "duration_s": round(time.perf_counter() - t_start, 2),
        "trajectory": trajectory,
        "gold_trajectory": [a["name"] for a in task["actions"]],
        **score(task, tools, agent_texts),
        "steps": steps,
    }


def score(task: dict, tools, agent_texts: list[str]) -> dict:
    """tau-bench reward: final DB must equal the DB after replaying the gold
    actions, and every required output must appear in the agent's replies.
    Only possible with the local backend (with MCP the DB lives in the servers)."""
    if not isinstance(tools, LocalToolBackend):
        return {"db_ok": None, "outputs_ok": None, "reward": None}
    gold = LocalToolBackend()
    for a in task["actions"]:
        if a["name"] not in TERMINATE_TOOLS:
            gold.call(a["name"], a["kwargs"])
    db_ok = tools.data_hash() == gold.data_hash()
    said = " ".join(agent_texts).lower().replace(",", "")
    outputs_ok = all(o.lower() in said for o in task.get("outputs", []))
    return {"db_ok": db_ok, "outputs_ok": outputs_ok, "reward": float(db_ok and outputs_ok)}
