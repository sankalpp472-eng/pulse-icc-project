"""
Signals: what the predictor knows about the conversation besides the last tools.

ONE definition, used in two places so training and live use can never disagree:
  - predictor/build_action_data.py  computes them from the recorded tau-bench conversations
  - agent/hints.py                  computes them live (from the agent's messages) and sends them

Only numbers and yes/no flags leave this module: no raw customer text, no tool output, no IDs
(the PRD: the trace store keeps structure and timing, never raw prompts).

Input = the conversation so far, oldest first, as events:
    ("user", "<customer message>")
    ("respond",)                                 the agent answered the customer (an LLM step, no tool)
    ("call", "<tool name>", {arguments})         the agent called a tool (an LLM step)
    ("tool", "<tool name>", "<tool result text>")

Features (all per the predictor spec):
  position / usage   step, n_calls, n_distinct, last_repeat, calls_since_user, last_event_user
  tool output        last_err, pending_ids, out_size
  customer intent    kw_* flags (crude regexes; an intent classifier could replace them)
The sequence features (last tool, last two tools) come from the hint's trajectory.
"""

import json
import re

INTENT_WORDS = {
    "kw_cancel":  r"\bcancel",
    "kw_change":  r"\b(change|modify|switch|move|reschedul)",
    "kw_bag":     r"\b(bag|baggage|luggage)",
    "kw_book":    r"\b(book|new (flight|reservation|trip)|reserve)",
    "kw_upgrade": r"\b(upgrade|business class|first class)",
    "kw_refund":  r"\b(refund|compensat|certificate|voucher|money back)",
    "kw_all":     r"\b(all|every|each|both)\b",
    "kw_pass":    r"\b(passenger|name)\b",
    "kw_ins":     r"\binsurance",
    "kw_delay":   r"\b(delay|delayed|cancelled by|missed)",
    "kw_human":   r"\b(human|supervisor|representative|manager)",
}
_INTENT_RE = {k: re.compile(v, re.I) for k, v in INTENT_WORDS.items()}

# Reservation-style IDs: 5-8 upper-case letters/digits with at least one digit (e.g. "UM3OG5")
_ID_RE = re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{5,8}\b")

POSITION_KEYS = ["step", "n_calls", "n_distinct", "last_repeat", "calls_since_user", "last_event_user"]
OUTPUT_KEYS = ["last_err", "pending_ids", "out_size"]
INTENT_KEYS = list(INTENT_WORDS)
SIGNAL_KEYS = POSITION_KEYS + OUTPUT_KEYS + INTENT_KEYS


def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _ids_in_output(data):
    """IDs listed in a tool result, e.g. get_user_details -> "reservations": ["UM3OG5", ...]."""
    found = set()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                found |= {x for x in v if _ID_RE.fullmatch(x)}
    return found


def _arg_values(args):
    vals = set()
    for v in (args or {}).values():
        if isinstance(v, str):
            vals.add(v)
    return vals


def compute_signals(events) -> dict:
    s = {k: 0 for k in SIGNAL_KEYS}
    s["last_event_user"] = 1                       # a conversation starts with the customer
    seen, fetched, distinct = set(), set(), set()
    last_tool, repeat = None, 0
    for ev in events:
        kind = ev[0]
        if kind == "user":
            s["calls_since_user"] = 0
            s["last_event_user"] = 1
            text = ev[1] or ""
            for k, rx in _INTENT_RE.items():
                if rx.search(text):
                    s[k] = 1
            seen |= set(_ID_RE.findall(text))      # IDs the customer typed
        elif kind == "respond":
            s["step"] += 1
        elif kind == "call":
            _, name, args = ev
            s["step"] += 1
            s["n_calls"] += 1
            s["calls_since_user"] += 1
            distinct.add(name)
            repeat = repeat + 1 if name == last_tool else 1
            last_tool = name
            fetched |= _arg_values(args)
        elif kind == "tool":
            result = ev[2] or ""
            s["last_event_user"] = 0
            s["last_err"] = int(result.lstrip().lower().startswith("error"))
            s["out_size"] = len(result)
            seen |= _ids_in_output(_json(result))
    s["step"] += 1                                  # the step this hint is for
    s["n_distinct"] = len(distinct)
    s["last_repeat"] = repeat
    s["pending_ids"] = len(seen - fetched)
    return s


def pending_pool(events) -> set:
    """The pending IDs themselves (training only: which tools consume them). Never sent anywhere."""
    seen, fetched = set(), set()
    for ev in events:
        if ev[0] == "user":
            seen |= set(_ID_RE.findall(ev[1] or ""))
        elif ev[0] == "call":
            fetched |= _arg_values(ev[2])
        elif ev[0] == "tool":
            seen |= _ids_in_output(_json(ev[2] or ""))
    return seen - fetched


def uses_pending(args, pool) -> bool:
    return bool(_arg_values(args) & pool)


def _from_langchain(m) -> dict:
    kind = getattr(m, "type", "")
    if kind == "human":
        return {"role": "user", "content": m.content}
    if kind == "ai":
        return {"role": "assistant", "content": m.content,
                "tool_calls": [{"function": {"name": tc["name"], "arguments": tc.get("args") or {}}}
                               for tc in (getattr(m, "tool_calls", None) or [])]}
    if kind == "tool":
        return {"role": "tool", "content": m.content, "name": getattr(m, "name", None)}
    return {"role": kind}


def events_from_messages(messages, trajectory=None):
    """Agent side: OpenAI-style messages (agent/agent.py) -> events.
    Tool results are matched to calls in order; `trajectory` is a fallback for tool names."""
    events, calls = [], []
    for m in messages:
        if not isinstance(m, dict):              # LangChain message objects (agent/langgraph_agent.py)
            m = _from_langchain(m)
        role = m.get("role")
        if role == "user":
            events.append(("user", m.get("content") or ""))
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if not tcs:
                events.append(("respond",))
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    args = _json(args) or {}
                events.append(("call", fn.get("name", ""), args or {}))
                calls.append(fn.get("name", ""))
        elif role == "tool":
            k = sum(1 for e in events if e[0] == "tool")
            name = m.get("name") or (calls[k] if k < len(calls) else
                                     (trajectory[k] if trajectory and k < len(trajectory) else ""))
            events.append(("tool", name, m.get("content") or ""))
    return events
