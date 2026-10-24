"""
Signals: what the LightGBM predictor knows about the conversation besides the tool names.

ONE definition, used in two places so training and live use can never disagree:
  - predictor/build_lgbm_data.py  computes them from the recorded tau-bench conversations
  - agent/agent.py (via agent/hints.py) computes them live and sends them in the hint

Only numbers and yes/no flags leave this function: no raw customer text, no tool output
(the PRD: traces keep structure and timing, never raw prompts).

Input = the conversation so far as a list of events, oldest first:
    ("user", "<customer message>")
    ("tool", "<tool name>", "<tool result text>")
"""

import json
import re

# Customer intent: does ANY customer message so far mention ...
INTENT_WORDS = {
    "kw_cancel": r"\bcancel",
    "kw_change": r"\b(change|modify|switch|move|reschedul)",
    "kw_book":   r"\b(book|new (flight|reservation)|reserve)",
    "kw_bag":    r"\b(bag|baggage|luggage)",
    "kw_comp":   r"\b(refund|compensat|certificate|delay)",
    "kw_pass":   r"\b(passenger|name)\b",
    "kw_cabin":  r"\b(upgrade|cabin|business|economy)",
    "kw_ins":    r"\binsurance",
    "kw_human":  r"\b(human|supervisor|representative|manager)",
}
_INTENT_RE = {k: re.compile(v, re.I) for k, v in INTENT_WORDS.items()}

RESULT_KEYS = ["n_calls", "user_turns_since_tool", "last_err", "last_empty", "last_n_items",
               "res_seen", "res_basic", "res_ins", "res_round", "user_n_res", "member"]
INTENT_KEYS = list(INTENT_WORDS)
SIGNAL_KEYS = RESULT_KEYS + INTENT_KEYS
MEMBER = {"regular": 0, "silver": 1, "gold": 2}


def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def compute_signals(events, with_intent=True) -> dict:
    s = {k: 0 for k in SIGNAL_KEYS}
    for ev in events:
        if ev[0] == "user":
            s["user_turns_since_tool"] += 1
            if with_intent:
                for k, rx in _INTENT_RE.items():
                    if rx.search(ev[1] or ""):
                        s[k] = 1
            continue
        _, name, result = ev
        result = result or ""
        s["n_calls"] += 1
        s["user_turns_since_tool"] = 0
        s["last_err"] = int(result.lstrip().lower().startswith("error"))
        data = _json(result)
        s["last_empty"] = int(result.strip() in ("", "[]", "{}", "null") or data == [] or data == {})
        if isinstance(data, list):
            s["last_n_items"] = len(data)
        elif isinstance(data, dict):
            s["last_n_items"] = 1
        else:
            s["last_n_items"] = result.count("flight_number")
        if name == "get_reservation_details" and isinstance(data, dict):
            s["res_seen"] += 1
            s["res_basic"] = int(data.get("cabin") == "basic_economy")
            s["res_ins"] = int(data.get("insurance") == "yes")
            s["res_round"] = int(data.get("flight_type") == "round_trip")
        if name == "get_user_details" and isinstance(data, dict):
            s["user_n_res"] = len(data.get("reservations") or [])
            s["member"] = MEMBER.get(str(data.get("membership", "")).lower(), 0)
    if not with_intent:
        for k in INTENT_KEYS:
            s.pop(k)
    return s


def events_from_messages(messages, trajectory):
    """Agent side: OpenAI-style messages + the tool names in call order -> events.
    The k-th tool message is the result of the k-th tool in the trajectory."""
    events, k = [], 0
    for m in messages:
        role = m.get("role")
        if role == "user":
            events.append(("user", m.get("content") or ""))
        elif role == "tool":
            name = m.get("name") or (trajectory[k] if k < len(trajectory) else "")
            events.append(("tool", name, m.get("content") or ""))
            k += 1
    return events
