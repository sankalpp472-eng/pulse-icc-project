"""
Step 8: send prediction hints from the agent to the predictor (FR-2, FR-8).

The agent calls send_hint() right before every LLM step. The hint goes out on a
background thread with a short timeout, so the agent NEVER waits for the predictor:
if the predictor is slow, down, or wrong, the tool call just happens normally
(reactive fallback, FR-8).

Settings (environment variables, e.g. in .env):
  PREDICTOR_URL   e.g. http://127.0.0.1:8000   (unset -> hints are switched off)
  WORKFLOW_ID     e.g. W1 or W2                (unset -> predictor uses the pooled airline model)
  HINT_TIMEOUT_S  default 0.2
  HINT_LOG        default runs/hints.jsonl     (one line per hint + the predictor's answer)
  HINT_SIGNALS    default 1: 0 = never send signals or history (tool names only)

What a hint carries depends on the predictor (its GET /info "inputs", predictor/client.py):
  trajectory  always
  signals     numbers and yes/no flags about tool results and the customer's requests (predictor/signals.py),
              no raw text: lightgbm
  history     the last 10 conversation messages, compacted: speculator, pulse_v3 (they read text)
"""

import json
import os
import threading
import time
import urllib.request

from dotenv import load_dotenv

from pulse import paths

load_dotenv(paths.ENV_FILE)   # .env (keys, PREDICTOR_URL, ...): this module is imported before llm.py

PREDICTOR_URL = os.getenv("PREDICTOR_URL", "").rstrip("/")
WORKFLOW_ID = os.getenv("WORKFLOW_ID") or None
DOMAIN = "retail" if os.getenv("DOMAIN") == "retail" else "airline"
TIMEOUT_S = float(os.getenv("HINT_TIMEOUT_S", "0.2"))
HINT_LOG = os.getenv("HINT_LOG", "runs/hints.jsonl")
_log_lock = threading.Lock()

# The SAME signal code the LightGBM model was trained with (pulse/predictor/signals.py)
try:
    import pulse.predictor.client as client
    from pulse.predictor.signals import compute_signals, events_from_messages
except ImportError:          # predictor folder missing -> names-only hints
    client = compute_signals = events_from_messages = None
USE_CONTEXT = os.getenv("HINT_SIGNALS", "1") != "0" and client is not None


def _post(hint: dict) -> None:
    record = {"hint": hint}
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(PREDICTOR_URL + "/predict", data=json.dumps(hint).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            ans = json.loads(r.read())
            record.update(ok=True, prewarm=ans["prewarm"], top3=ans["ranked"][:3],
            theta=ans["theta"], ttl_s=ans["ttl_s"])
    except Exception as e:  # predictor down / slow -> ignore, the agent carries on
        record.update(ok=False, error=f"{type(e).__name__}: {e}")
    record["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    try:
        os.makedirs(os.path.dirname(HINT_LOG) or ".", exist_ok=True)
        with _log_lock, open(HINT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def send_hint(episode_id: str, trajectory: list[str], messages: list[dict] | None = None) -> None:
    """messages = the agent's conversation so far (OpenAI-style dicts), for the signals / history."""
    if not PREDICTOR_URL:
        return
    threading.Thread(target=_send, args=(episode_id, list(trajectory), list(messages) if messages is not None else None),
                     daemon=True).start()


def _send(episode_id, trajectory, messages):
    """Background: build the hint with the inputs this predictor reads, then post it."""
    signals = history = None
    wanted = {"trajectory"}
    if USE_CONTEXT and messages is not None:
        wanted = client.inputs(PREDICTOR_URL)          # GET /info once, cached
        try:
            if "signals" in wanted:
                signals = compute_signals(events_from_messages(messages, trajectory))
            if "history" in wanted:
                history = client.compact(messages)
        except Exception:    # never let a signal bug affect the agent
            pass
    if client is None:
        hint = {"workflow_id": WORKFLOW_ID, "episode_id": episode_id, "trajectory": trajectory[-8:],
                "current_tool": trajectory[-1] if trajectory else "START", "ts": time.time()}
    else:
        hint = client.make_hint(wanted, episode_id, trajectory, WORKFLOW_ID, domain=DOMAIN, signals=signals,
                                history=history)
    _post(hint)
