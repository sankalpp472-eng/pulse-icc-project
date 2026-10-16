"""
LangGraph version of the ReAct agent (same behaviour as agent.py, built as a graph).

    START -> hint -> agent --tool call--> tools -> hint -> ...
                       |
                       +--plain text--> user -> hint -> ...      (END on ###STOP###,
                                                                  transfer, or max steps)

The graph only describes the loop. The LLM still chooses every tool, so the
next tool stays something the predictor has to guess.

Tools run through the same LocalToolBackend as agent.py (later: MCP/serverless).
LangSmith tracing turns on automatically when LANGSMITH_TRACING=true and
LANGSMITH_API_KEY are set; keep it off for latency experiments.
"""

import json
import operator
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import RateLimitError

import llm
from agent import emit_hint, score
from prompts import AGENT_SYSTEM_PROMPT
from tools import TERMINATE_TOOLS, LocalToolBackend
from user import STOP, HumanUser, SimulatedUser


class State(TypedDict):
    messages: Annotated[list, add_messages]          # the agent's conversation
    trajectory: Annotated[list[str], operator.add]   # tool names, in call order
    steps: Annotated[list[dict], operator.add]       # one record per LLM step (the trace)
    agent_texts: Annotated[list[str], operator.add]  # what the agent said to the user
    think: dict                                       # timing of the latest LLM call
    end_reason: str


def _make_model(tool_schemas: list[dict]):
    chat = ChatOpenAI(model=llm.DEFAULT_MODEL, base_url=llm.BASE_URL, api_key=llm.API_KEY,
                      temperature=0, max_retries=0, timeout=llm.TIMEOUT)
    # One tool per step, as in ReAct
    return chat.bind_tools(tool_schemas, parallel_tool_calls=False)


def _invoke_with_rate_limit(model, messages, max_retries: int = 8):
    """Like llm.chat(): rate-limit waits are excluded from the measured think time."""
    waited = 0.0
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            return model.invoke(messages), time.perf_counter() - t0, waited
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            retry_after = e.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
            print(f"  [llm] rate limited, waiting {wait:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            waited += wait


def build_graph(episode_id: str, tools: LocalToolBackend, customer, max_steps: int,
                verbose: bool, t_start: float):
    model = _make_model(tools.schemas)

    def hint(state: State):
        emit_hint(episode_id, state["trajectory"])
        return {}

    def agent(state: State):
        ai, latency, waited = _invoke_with_rate_limit(model, state["messages"])
        step = len(state["steps"]) + 1
        think = {"step": step, "t": round(time.perf_counter() - t_start, 3),
                 "think_s": round(latency, 3), "rate_wait_s": round(waited, 1)}
        return {"messages": [ai], "think": think}

    def tools_node(state: State):
        ai: AIMessage = state["messages"][-1]
        th = state["think"]
        out, calls, names = [], [], []
        for tc in ai.tool_calls:
            obs, tool_s = tools.call(tc["name"], tc["args"])
            out.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
            names.append(tc["name"])
            calls.append({"tool": tc["name"], "args": tc["args"], "tool_s": round(tool_s, 4)})
            if verbose:
                print(f"  [{th['step']:>2}] think {th['think_s']:5.2f}s -> {tc['name']}({json.dumps(tc['args'])})")
                print(f"       obs: {obs[:200]}{'...' if len(obs) > 200 else ''}")
        update = {"messages": out, "trajectory": names, "steps": [{**th, "kind": "tool", "calls": calls}]}
        if any(n in TERMINATE_TOOLS for n in names):
            update["end_reason"] = "transfer"
        return update

    def user_node(state: State):
        text = state["messages"][-1].content or ""
        th = state["think"]
        if verbose:
            print(f"  [{th['step']:>2}] think {th['think_s']:5.2f}s -> respond")
            print(f"[agent] {text}")
        reply = customer.reply(text)
        if verbose:
            print(f"[user]  {reply}")
        update = {"agent_texts": [text], "steps": [{**th, "kind": "respond"}]}
        # End only on a standalone STOP (same rule as agent.py)
        reply = reply.replace(STOP, "").strip() if reply.strip() != STOP else STOP
        if reply == STOP:
            update["end_reason"] = "user_stop"
        else:
            update["messages"] = [HumanMessage(reply)]
        return update

    def after_agent(state: State):
        if len(state["steps"]) + 1 > max_steps:
            return END
        return "tools" if state["messages"][-1].tool_calls else "user"

    def after_turn(state: State):
        return END if state.get("end_reason") else "hint"

    g = StateGraph(State)
    g.add_node("hint", hint)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_node("user", user_node)
    g.add_edge(START, "hint")
    g.add_edge("hint", "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "user", END])
    g.add_conditional_edges("tools", after_turn, ["hint", END])
    g.add_conditional_edges("user", after_turn, ["hint", END])
    return g.compile()


def run_episode(task: dict, user: str = "sim", max_steps: int = 30,
                verbose: bool = True) -> dict:
    """Same inputs and output format as agent.run_episode."""
    episode_id = f"task{task['id']}-{int(time.time())}"
    tools = LocalToolBackend()
    customer = SimulatedUser(task["instruction"]) if user == "sim" else HumanUser(task["instruction"])

    t_start = time.perf_counter()
    first = customer.start()
    if verbose:
        print(f"\n[user]  {first}")

    graph = build_graph(episode_id, tools, customer, max_steps, verbose, t_start)
    init = {"messages": [SystemMessage(AGENT_SYSTEM_PROMPT), HumanMessage(first)],
            "trajectory": [], "steps": [], "agent_texts": [], "think": {}, "end_reason": ""}
    final = graph.invoke(init, config={
        "recursion_limit": 4 * max_steps + 10,   # ~4 graph nodes per agent step
        "run_name": f"airline-task-{task['id']}",
        "tags": ["pulse", "langgraph", llm.BACKEND],
        "metadata": {"episode_id": episode_id, "task_id": task["id"], "model": llm.DEFAULT_MODEL},
    })

    return {
        "episode_id": episode_id,
        "task_id": task["id"],
        "end_reason": final["end_reason"] or "max_steps",
        "duration_s": round(time.perf_counter() - t_start, 2),
        "trajectory": final["trajectory"],
        "gold_trajectory": [a["name"] for a in task["actions"]],
        **score(task, tools, final["agent_texts"]),
        "steps": final["steps"],
    }
