"""
ReAct-style agent for the tau-bench airline domain.

Loop:
    hint -> LLM reasons -> tool call(s) -> observation(s) -> LLM reasons -> ...
    and when the LLM replies in plain text, the (simulated) user answers.
Independent lookups requested in one reply run at the same time (parallel fan-out).

Every LLM step is recorded with its think time, which is the window the
prewarmer will have once hints are wired to the predictor.
"""

import asyncio
import json
import time

from hints import send_hint
from openai import APIStatusError

from llm import CONTEXT_BUDGET, chat, est_tokens, fit_context
from prompts import AGENT_SYSTEM_PROMPT
from tools import READ_ONLY_TOOLS, TERMINATE_TOOLS, LocalToolBackend, make_tool_backend
from user import STOP, HumanUser, SimulatedUser


def emit_hint(episode_id: str, trajectory: list[str], messages: list[dict] | None = None) -> None:
    """Called right before every LLM step. Sends {workflow id, current tool,
    recent trajectory} (+ signals computed from `messages`, for the LightGBM predictor)
    to the predictor in the background (see hints.py). Does nothing when PREDICTOR_URL is not set."""
    send_hint(episode_id, trajectory, messages)

def _chat_within_limit(messages: list[dict], schemas: list[dict], scale: float = 1.0):
    """chat() on a copy trimmed to the context budget, with two provider glitches handled:
      HTTP 413 (request still too large: our estimate was low) -> trim harder and retry
      HTTP 400 tool_use_failed (gpt-oss on Groq sometimes emits a malformed tool call) -> retry"""
    budget, glitches = CONTEXT_BUDGET, 0
    for _ in range(6):
        to_send, trimmed = fit_context(messages, schemas, budget=budget, scale=scale)
        try:
            r = chat(to_send, tools=schemas)
            if r.prompt_tokens:   # calibrate the next estimate against the provider's real count
                _chat_within_limit.scale = r.prompt_tokens / max(1, est_tokens(to_send, schemas))
            return r, trimmed
        except APIStatusError as e:
            if e.status_code == 400 and "tool_use_failed" in str(e) and glitches < 3:
                glitches += 1
                print(f"  [llm] malformed tool call from the model; retrying ({glitches}/3)")
                continue
            if e.status_code != 413 or not budget:
                raise
            budget = int(budget * 0.8)
            print(f"  [llm] request too large; retrying with a smaller context budget ({budget} tokens)")
    to_send, trimmed = fit_context(messages, schemas, budget=budget, keep_last=1, scale=scale)
    return chat(to_send, tools=schemas), trimmed


def _assistant_msg(m) -> dict:
    msg = {"role": "assistant", "content": m.content or ""}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in m.tool_calls
        ]
    return msg


def _run_calls(tools, parsed: list[tuple[str, dict]]) -> tuple[list[tuple[str, float]], bool]:
    """Run one reply's tool calls. Several read-only calls on the MCP backend go out at once
    (asyncio.gather); writes, or a mix, run in order (they change the shared database).
    Returns ([(observation, latency_s)], ran_in_parallel)."""
    if len(parsed) > 1 and hasattr(tools, "acall") and all(n in READ_ONLY_TOOLS for n, _ in parsed):
        async def fire():
            return await asyncio.gather(*(tools.acall(n, a) for n, a in parsed))
        return [(obs, lat) for obs, lat, _ in asyncio.run(fire())], True
    return [tools.call(n, a) for n, a in parsed], False


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

    scale = 1.0   # real/estimated prompt tokens, updated after every LLM call
    _chat_within_limit.scale = 1.0
    for step in range(1, max_steps + 1):
        emit_hint(episode_id, trajectory, messages)
        r, trimmed = _chat_within_limit(messages, tools.schemas, scale)
        scale = getattr(_chat_within_limit, "scale", scale)
        m = r.message
        messages.append(_assistant_msg(m))
        rec = {"step": step, "t": round(time.perf_counter() - t_start, 3),
               "think_s": round(r.latency_s, 3), "rate_wait_s": round(r.rate_wait_s, 1),
               "context_trimmed": trimmed, "prompt_tokens": r.prompt_tokens}

        if m.tool_calls:
            # Native tool calling: one reply may hold several calls (independent lookups run at once)
            parsed = {}                     # call index -> args (calls with invalid JSON are left out)
            for k, tc in enumerate(m.tool_calls):
                try:
                    parsed[k] = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    pass
            results, parallel = _run_calls(tools, [(m.tool_calls[k].function.name, a) for k, a in parsed.items()])
            results = dict(zip(parsed, results))
            calls = []
            for k, tc in enumerate(m.tool_calls):
                name = tc.function.name
                if k in parsed:
                    args, (obs, tool_s) = parsed[k], results[k]
                else:
                    args, obs, tool_s = {}, "Error: arguments were not valid JSON", 0.0
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
                trajectory.append(name)
                calls.append({"tool": name, "args": args, "tool_s": round(tool_s, 4)})
                if verbose:
                    tag = f" [parallel x{len(parsed)}]" if parallel else ""
                    print(f"  [{step:>2}] think {r.latency_s:5.2f}s -> {name}({json.dumps(args)}){tag}")
                    print(f"       obs: {obs[:200]}{'...' if len(obs) > 200 else ''}")
            steps.append({**rec, "kind": "tool", "calls": calls, "parallel": parallel})
            if any(c["tool"] in TERMINATE_TOOLS for c in calls):
                end_reason = "transfer"
                break
        else:
            text = m.content or ""
            agent_texts.append(text)
            if verbose:
                print(f"  [{step:>2}] think {r.latency_s:5.2f}s -> respond")
                print(f"[agent] {text}")
            reply = customer.reply(text)
            # How long the customer took to answer (LLM time or the simulated reading + typing delay,
            # whichever is longer; rate-limit waits excluded): recorded so a replay of this run can
            # reproduce the real gap between tool calls.
            steps.append({**rec, "kind": "respond", "user_s": round(getattr(customer, "last_latency_s", 0.0), 3),
                          "user_delay_s": round(getattr(customer, "last_delay_s", 0.0), 3)})
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
