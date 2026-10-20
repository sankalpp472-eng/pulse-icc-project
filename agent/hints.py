"""
Step 8: send prediction hints from the agent to the predictor (FR-2, FR-8).

The agent calls send_hint() right before every LLM step. The hint goes out on a
background thread with a short timeout, so the agent NEVER waits for the predictor:
if the predictor is slow, down, or wrong, the tool call just happens normally
(reactive fallback, FR-8).

Settings (environment variables, e.g. in agent/.env):
  PREDICTOR_URL   e.g. http://127.0.0.1:8000   (unset -> hints are switched off)
  WORKFLOW_ID     e.g. W1 or W2                (unset -> predictor uses the pooled airline model)
  HINT_TIMEOUT_S  default 0.2
  HINT_LOG        default runs/hints.jsonl     (one line per hint + the predictor's answer)
"""

import json
import os
import threading
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()   # read agent/.env here too: this module is imported before llm.py

PREDICTOR_URL = os.getenv("PREDICTOR_URL", "").rstrip("/")
WORKFLOW_ID = os.getenv("WORKFLOW_ID") or None
TIMEOUT_S = float(os.getenv("HINT_TIMEOUT_S", "0.2"))
HINT_LOG = os.getenv("HINT_LOG", "runs/hints.jsonl")
_log_lock = threading.Lock()


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


def send_hint(episode_id: str, trajectory: list[str]) -> None:
    if not PREDICTOR_URL:
        return
    hint = {"workflow_id": WORKFLOW_ID, "episode_id": episode_id,
            "trajectory": list(trajectory)[-8:],     # recent trajectory is enough
            "current_tool": trajectory[-1] if trajectory else "START",
            "ts": time.time()}
    threading.Thread(target=_post, args=(hint,), daemon=True).start()
