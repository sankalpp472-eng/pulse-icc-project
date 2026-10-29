"""
The speculator's prompt: a fast LLM guesses the agent's next action from the conversation so far.

Follows the speculator role of Speculative Actions (Ye et al., 2026), adapted to next-TOOL
prediction: it ranks the domain's tools + RESPOND instead of guessing the customer's next reply.

Trimmed to fit Groq's free tier (no policy text, no tool parameter schemas):
  * tool names with a one-line description
  * every tool called so far, in order (what the n-gram also sees)
  * the last LAST_N messages, each cut to MAX_CHARS characters
The answer is JSON with 4 actions and probabilities, so it can be scored like the other models
(top-1 / top-3) and run through the same theta sweep. 4, not 3: "tool top-3" drops RESPOND and
ranks tools only, so a guess list with RESPOND in it still has 3 tools to score.
"""

import json
import re
import sys

from common import AGENT_DIR, RESPOND

LAST_N = 10
MAX_CHARS = 400

# tau-bench's own description of search_onestop_flight is a copy of the direct one
FIXED_DESCRIPTIONS = {"search_onestop_flight": "Search one-stop flights between two cities on a specific date."}

SYSTEM = """You predict the next action of an airline customer-service agent.
The agent works one step at a time: at each step it either calls exactly one tool, or writes a
message to the customer (action RESPOND). Given the conversation so far, predict the agent's NEXT
action.

Actions:
{actions}

Answer with JSON only, exactly this shape, 4 different actions, most likely first, p values
summing to at most 1:
{{"predictions": [{{"action": "<name>", "p": 0.55}}, {{"action": "<name>", "p": 0.25}}, {{"action": "<name>", "p": 0.1}}, {{"action": "<name>", "p": 0.05}}]}}"""


def domain_actions(domain="airline"):
    """[(name, one-line description)] for the domain's tools, then RESPOND."""
    if domain != "airline":
        raise NotImplementedError("only the airline tools are wired up (agent/tau_airline)")
    sys.path.insert(0, str(AGENT_DIR))
    from tau_airline import ALL_TOOLS
    out = []
    for t in ALL_TOOLS:
        f = t.get_info()["function"]
        desc = FIXED_DESCRIPTIONS.get(f["name"]) or re.split(r"(?<=\.)\s", f["description"].strip())[0]
        out.append((f["name"], desc))
    out.append((RESPOND, "Write a message to the customer instead of calling a tool."))
    return out


def _cut(text, n=MAX_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " [...]"


def render_message(m):
    if m["role"] == "user":
        return f"CUSTOMER: {_cut(m['content'])}"
    if m["role"] == "tool":
        return f"TOOL RESULT ({m.get('name', '')}): {_cut(m['content'])}"
    lines = [f"AGENT: {_cut(m['content'])}"] if m.get("content") else []
    lines += [f"AGENT CALLS {name}({_cut(json.dumps(args), 200)})" for name, args in m.get("calls", [])]
    return "\n".join(lines) or "AGENT: (empty)"


def build_messages(history, trajectory, actions, last_n=LAST_N):
    """history = the conversation BEFORE the step (compact messages from steps.py)."""
    shown = history[-last_n:]
    skipped = len(history) - len(shown)
    convo = "\n".join(render_message(m) for m in shown) or "(no messages yet)"
    user = (f"Tools called so far, in order: {', '.join(trajectory) or 'none'}\n\n"
            f"Conversation{f' (last {len(shown)} of {len(history)} messages)' if skipped else ''}:\n"
            f"{convo}\n\nWhat is the agent's next action?")
    system = SYSTEM.format(actions="\n".join(f"- {n}: {d}" for n, d in actions))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse(text, names):
    """LLM answer -> {action: p} over known actions (ranked by p), or None if unusable.

    Unknown names are dropped; names match case-insensitively; missing/invalid p falls back to
    rank weights so the order the model gave is kept."""
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        preds = json.loads(m.group(0))["predictions"] if m else None
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(preds, list):
        return None
    lookup = {n.lower(): n for n in names}
    out = {}
    for rank, item in enumerate(preds):
        if not isinstance(item, dict):
            continue
        name = lookup.get(str(item.get("action", "")).strip().lower())
        if not name or name in out:
            continue
        try:
            pv = float(item.get("p"))
        except (TypeError, ValueError):
            pv = -1
        out[name] = pv if 0 <= pv <= 1 else 1 / (rank + 2)
    return out or None
