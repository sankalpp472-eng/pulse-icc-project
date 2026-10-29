"""
Step 2: ask a fast LLM (Groq) for the agent's next action at every sampled held-out step.

One request per step, sequential, temperature 0, JSON answer (prompt.py). Each row records the
ranked guess, the raw answer, tokens, and the latency of the SUCCESSFUL request only (time spent
waiting on rate limits is kept apart, as in agent/llm.py), because latency is half the result:
a guess that arrives after the tool call is useless for prewarming.

Keys: GROQ_API_KEY_1..N (or GROQ_API_KEY) from agent/.env. Requests go round-robin over the keys
to spread the per-minute limits; a key that hits its DAILY limit is dropped for the run. Rows are
appended as they come, so a run stopped by the quota resumes with --out <same folder>.

    python baselines/llm_speculator/speculate.py --dry-run                 # show one prompt, no API calls
    python baselines/llm_speculator/speculate.py --limit 5                 # quick check
    python baselines/llm_speculator/speculate.py                           # the 300-step sample, gpt-oss-20b
    python baselines/llm_speculator/speculate.py --model qwen/qwen3.8-27b
    python baselines/llm_speculator/speculate.py --out results/<folder>    # resume

Writes results/<date>_<time>_<model>_<sample>/run.json + predictions.jsonl
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, NotFoundError,
                    OpenAI, PermissionDeniedError, RateLimitError)

from common import AGENT_DIR, DATA, HERE, RESULTS, read_jsonl, step_key
from prompt import LAST_N, build_messages, domain_actions, parse

GROQ_URL = "https://api.groq.com/openai/v1"
MAX_RATE_WAIT_S = 120      # longer waits (or "per day") = that key's quota is used up


class QuotaExhausted(RuntimeError):
    pass


def groq_keys():
    load_dotenv(AGENT_DIR / ".env")
    numbered = sorted((k for k in os.environ if re.fullmatch(r"GROQ_API_KEY_\d+", k)),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
    keys = [os.environ[k] for k in numbered if os.environ[k].strip()]
    if not keys and os.getenv("GROQ_API_KEY"):
        keys = [os.environ["GROQ_API_KEY"]]
    if not keys:
        raise ValueError("No Groq key: set GROQ_API_KEY_1 (..._2, ...) or GROQ_API_KEY in agent/.env")
    return keys


class KeyPool:
    """Round-robin over keys; per-minute 429s cool a key down, daily 429s drop it."""

    def __init__(self, keys, timeout=60):
        self.clients = [OpenAI(base_url=GROQ_URL, api_key=k, max_retries=0, timeout=timeout) for k in keys]
        self.cool_until = [0.0] * len(keys)
        self.dead, self.next, self.log = set(), 0, []

    def pick(self):
        """-> (key index, seconds slept waiting for a key)."""
        live = [i for i in range(len(self.clients)) if i not in self.dead]
        if not live:
            raise QuotaExhausted(f"all {len(self.clients)} key(s) hit their daily limit")
        slept = 0.0
        now = time.time()
        if all(self.cool_until[i] > now for i in live):
            slept = min(self.cool_until[i] for i in live) - now
            time.sleep(slept)
        now = time.time()
        for off in range(len(self.clients)):
            i = (self.next + off) % len(self.clients)
            if i in live and self.cool_until[i] <= now:
                self.next = i + 1
                return i, slept
        raise AssertionError("no key ready after waiting")

    def rate_limited(self, i, err):
        retry_after = err.response.headers.get("retry-after")
        wait = float(retry_after) if retry_after else 10.0
        msg = str(err).lower()
        if wait > MAX_RATE_WAIT_S or "per day" in msg or "per-day" in msg:
            self.dead.add(i)
            self.log.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "key": i + 1, "event": "daily limit",
                             "error": str(err)[:300]})
            print(f"  [groq] key {i + 1} hit its daily limit; {len(self.clients) - len(self.dead)} key(s) left")
        else:
            self.cool_until[i] = time.time() + wait


def ask(pool, model, messages, max_tokens, json_mode, max_errors=3):
    """-> dict(text, latency_s, rate_wait_s, prompt_tokens, completion_tokens, groq_key)."""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    if model.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"
    waited, errors = 0.0, 0
    while True:
        i, slept = pool.pick()
        waited += slept
        t0 = time.perf_counter()
        try:
            r = pool.clients[i].chat.completions.create(model=model, messages=messages, temperature=0.0,
                                                        max_tokens=max_tokens, **extra)
        except RateLimitError as e:
            pool.rate_limited(i, e)
            continue
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code < 500:
                raise                                        # bad model name, bad request: retrying won't help
            errors += 1
            if errors > max_errors:
                raise
            print(f"  [groq] {type(e).__name__} ({errors}/{max_errors}): {str(e)[:120]}")
            time.sleep(2 ** errors)
            waited += 2 ** errors
            continue
        lat = time.perf_counter() - t0
        u = r.usage
        return {"text": r.choices[0].message.content, "latency_s": lat, "rate_wait_s": waited,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None), "groq_key": i + 1}


def load_sample(sample_path):
    sample = json.loads(Path(sample_path).read_text())
    dom = sample["domain"]
    steps = {step_key(s): s for s in read_jsonl(DATA / f"{dom}_test_steps.jsonl")}
    runs = {r["run_id"]: r for r in read_jsonl(DATA / f"{dom}_test_runs.jsonl")}
    return dom, [steps[k] for k in sample["keys"]], runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default=str(DATA / "airline_sample_300_seed252.json"))
    p.add_argument("--model", default="openai/gpt-oss-20b",
                   help="a Groq model id (also usable on these keys: openai/gpt-oss-120b, qwen/qwen3.8-27b)")
    p.add_argument("--last-n", type=int, default=LAST_N, help="conversation messages shown")
    p.add_argument("--max-tokens", type=int, default=None, help="default 150, 1024 for gpt-oss (reasoning)")
    p.add_argument("--no-json-mode", action="store_true", help="don't send response_format=json_object")
    p.add_argument("--limit", type=int, default=0, help="only the first N sampled steps (0 = all)")
    p.add_argument("--out", default=None, help="existing results folder to resume")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and a token estimate")
    a = p.parse_args()

    dom, steps, runs = load_sample(a.sample)
    actions = domain_actions(dom)
    names = [n for n, _ in actions]
    if a.limit:
        steps = steps[:a.limit]
    max_tokens = a.max_tokens or (1024 if a.model.startswith("openai/gpt-oss") else 150)

    def prompt_for(s):
        hist = runs[s["run_id"]]["messages"][:s["msg_index"]]
        return build_messages(hist, s["trajectory"], actions, a.last_n)

    if a.dry_run:
        msgs = prompt_for(steps[0])
        print(f"--- step {step_key(steps[0])} (true action: {steps[0]['action']})")
        for m in msgs:
            print(f"[{m['role']}]\n{m['content']}\n")
        est = statistics.mean(len(json.dumps(prompt_for(s))) / 3.5 for s in steps)
        print(f"~{est:.0f} prompt tokens per step (chars/3.5) x {len(steps)} steps = "
              f"~{est * len(steps) / 1000:.0f}K prompt tokens, + up to {max_tokens} output tokens each")
        return

    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", a.model)
    out = Path(a.out) if a.out else RESULTS / f"{time.strftime('%Y-%m-%d_%H%M')}_{slug}_{Path(a.sample).stem}"
    if not out.is_absolute() and not out.exists():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {r["key"] for r in read_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [s for s in steps if step_key(s) not in done]
    cfg = {"model": a.model, "sample": str(a.sample), "domain": dom, "last_n": a.last_n,
           "max_tokens": max_tokens, "json_mode": not a.no_json_mode, "steps": len(steps)}
    print(f"{a.model}: {len(todo)} steps to run ({len(done)} already done) -> {out}")

    pool = KeyPool(groq_keys())
    t_start, n_ok = time.time(), 0
    try:
        with open(pred_path, "a", encoding="utf-8") as f:
            for n, s in enumerate(todo, 1):
                try:
                    r = ask(pool, a.model, prompt_for(s), max_tokens, not a.no_json_mode)
                    ranked = parse(r["text"], names)
                    row = {"key": step_key(s), "action": s["action"], "pred": ranked, "ok": ranked is not None, **r}
                except (QuotaExhausted, AuthenticationError, NotFoundError, PermissionDeniedError):
                    raise                                    # stop the run; nothing per-step about these
                except Exception as e:                       # noqa: BLE001  (record and move on)
                    row = {"key": step_key(s), "action": s["action"], "pred": None, "ok": False,
                           "error": f"{type(e).__name__}: {str(e)[:300]}"}
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_ok += row["ok"]
                if n % 10 == 0 or n == len(todo):
                    top = max(row["pred"], key=row["pred"].get) if row["pred"] else "-"
                    print(f"  {n}/{len(todo)}  ok {n_ok}  last: true {row['action']}, guess {top}, "
                          f"{row.get('latency_s', 0):.2f}s")
    except QuotaExhausted as e:
        print(f"stopped: {e}. Resume tomorrow with --out {out}")
    finally:
        run_path = out / "run.json"                          # one entry per session (resumes add one)
        sessions = json.loads(run_path.read_text()).get("sessions", []) if run_path.exists() else []
        sessions.append({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "steps_ok": n_ok,
                         "wall_s": round(time.time() - t_start, 1), "key_log": pool.log})
        run_path.write_text(json.dumps({**cfg, "sessions": sessions}, indent=1))


if __name__ == "__main__":
    main()
