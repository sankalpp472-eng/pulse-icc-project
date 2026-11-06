"""
The speculator's prompt: a fast LLM guesses the agent's next action from the conversation so far.

Follows the speculator role of Speculative Actions (Ye et al., 2026), adapted to next-TOOL
prediction: it ranks the domain's tools + RESPOND instead of guessing the customer's next reply.

Trimmed to fit Groq's free tier (no policy text, no tool parameter schemas):
  * tool names with a one-line description
  * every tool called so far, in order (what the n-gram also sees)
  * the last LAST_N messages, each cut to MAX_CHARS characters
The answer is JSON with 4 actions and probabilities, so it can be scored like the other models.
The schema shows no example numbers: with 0.55/0.25/0.1/0.05 in it, gpt-oss-20b copied them in 147 of
300 answers (run 2026-10-25_1900), so theta could not tell sure guesses from unsure ones
(top-1 / top-3) and run through the same theta sweep. 4, not 3: "tool top-3" drops RESPOND and
ranks tools only, so a guess list with RESPOND in it still has 3 tools to score.
"""

import json
import re

from pulse.speculator.common import RESPOND

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

{answer}"""

# k candidates in one answer (the paper's chess speculator: "top-k move predictions, ordered by confidence").
# No example numbers in the schema: with 0.55/0.25/0.1/0.05 shown, gpt-oss-20b copied them in 147 of 300
# answers (run 2026-10-25_1900), so theta could not tell sure guesses from unsure ones.
ARGS_NOTE = ('"args" are the tool\'s arguments as the agent would pass them '
             '({} for RESPOND or if unknown).')
ITEM = '{"action": "<name>", "p": <probability>, "args": {...}}'


def answer_list(k):
    return (f"Answer with JSON only, exactly this shape, {k} different actions, most likely first:\n"
            + '{"predictions": [' + ", ".join([ITEM] * k) + "]}\n" + ARGS_NOTE + "\n"
            "Each p is your own estimate (0 to 1) that this action is the next one; together at most 1.\n"
            "Be honest about uncertainty: a clear next step deserves a high p, an unclear one a low p.")


# one guess per answer (sampling strategy: k answers at temperature > 0, confidence = vote share)
ANSWER_ONE = 'Answer with JSON only, exactly this shape:\n{"action": "<name>", "args": {...}}\n' + ARGS_NOTE


def domain_actions(domain="airline"):
    """[(name, one-line description)] for the domain's tools, then RESPOND."""
    if domain != "airline":
        raise NotImplementedError("only the airline tools are wired up (pulse/agent/tau_airline)")
    from pulse.agent.tau_airline import ALL_TOOLS
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


def build_messages(history, trajectory, actions, last_n=LAST_N, k=4, single=False):
    """history = the conversation BEFORE the step (compact messages from steps.py).
    k = candidates asked for in one answer; single = one guess (for sampling)."""
    shown = history[-last_n:]
    skipped = len(history) - len(shown)
    convo = "\n".join(render_message(m) for m in shown) or "(no messages yet)"
    user = (f"Tools called so far, in order: {', '.join(trajectory) or 'none'}\n\n"
            f"Conversation{f' (last {len(shown)} of {len(history)} messages)' if skipped else ''}:\n"
            f"{convo}\n\nWhat is the agent's next action?")
    answer = ANSWER_ONE if single else answer_list(k)
    system = SYSTEM.format(actions="\n".join(f"- {n}: {d}" for n, d in actions), answer=answer)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def parse_full(text, names):
    """LLM answer -> ({action: p}, {action: args}) ranked by p, or (None, None) if unusable.
    Accepts the list shape ({"predictions": [...]}) and the single shape ({"action": ...}).
    Unknown names are dropped; names match case-insensitively; a missing/invalid p falls back to
    rank weights so the order the model gave is kept."""
    data = _json(text)
    if not isinstance(data, dict):
        return None, None
    preds = data.get("predictions")
    if preds is None and "action" in data:
        preds = [{**data, "p": data.get("p", 1.0)}]
    if not isinstance(preds, list):
        return None, None
    lookup = {n.lower(): n for n in names}
    probs, args = {}, {}
    for rank, item in enumerate(preds):
        if not isinstance(item, dict):
            continue
        name = lookup.get(str(item.get("action", "")).strip().lower())
        if not name or name in probs:
            continue
        try:
            pv = float(item.get("p"))
        except (TypeError, ValueError):
            pv = -1
        probs[name] = pv if 0 <= pv <= 1 else 1 / (rank + 2)
        args[name] = item.get("args") if isinstance(item.get("args"), dict) else {}
    return (probs, args) if probs else (None, None)


def parse(text, names):
    """{action: p} only (service.py)."""
    return parse_full(text, names)[0]
