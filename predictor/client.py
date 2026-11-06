"""
What a hint sender (agent/hints.py, ui/session.py, test/test_knative/w1_replay.py) needs to talk to any predictor:

  inputs(url)        the inputs the predictor at `url` reads, from its GET /info (cached per url).
                     If /info can't be reached: trajectory + signals (the lightgbm default).
  make_hint(...)     a PredictRequest dict carrying only those inputs: the conversation (history) leaves the
                     agent only for predictors that read it, signals only for those that use them.
  compact(messages)  the conversation as compact messages {role, content, calls?, name?}, the last HISTORY_N
                     (the format baselines/llm_speculator/steps.py and pulse_v3 read).

Standard library only, so the agent needs nothing extra.
"""

import json
import time
import urllib.request

HISTORY_N = 10
DEFAULT_INPUTS = ("trajectory", "signals")
_INFO = {}


def info(url: str, timeout: float = 2.0) -> dict:
    """The predictor's GET /info, cached; {} if unreachable (retried on the next call)."""
    url = (url or "").rstrip("/")
    if url and url not in _INFO:
        try:
            with urllib.request.urlopen(url + "/info", timeout=timeout) as r:
                _INFO[url] = json.loads(r.read())
        except Exception:
            return {}
    return _INFO.get(url, {})


def inputs(url: str) -> set:
    return set(info(url).get("inputs") or DEFAULT_INPUTS)


def compact(messages: list, n: int = HISTORY_N) -> list[dict]:
    """OpenAI-style message dicts -> compact messages, the last n."""
    out = []
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ("user", "assistant", "tool"):
            continue
        cm = {"role": role, "content": str(m.get("content") or "")[:600]}
        if role == "tool":
            cm["name"] = m.get("name", "")
        if role == "assistant" and m.get("tool_calls"):
            cm["calls"] = [[tc["function"]["name"], tc["function"].get("arguments")] for tc in m["tool_calls"]]
        out.append(cm)
    return out[-n:]


def make_hint(wanted: set, episode_id: str, trajectory: list, workflow_id: str | None = None,
              domain: str = "airline", signals: dict | None = None, history: list | None = None) -> dict:
    """A PredictRequest dict with the inputs in `wanted` that are available."""
    traj = list(trajectory)[-8:]                     # the recent trajectory is enough
    hint = {"workflow_id": workflow_id, "episode_id": episode_id, "domain": domain, "trajectory": traj,
            "current_tool": traj[-1] if traj else "START", "ts": time.time()}
    if "signals" in wanted and signals is not None:
        hint["signals"] = signals
    if "history" in wanted and history is not None:
        hint["history"] = history
    return hint
