"""
LLM client -- Groq, OpenRouter or Ollama, all through the OpenAI-compatible API.

Switch with LLM_BACKEND=groq|openrouter|ollama in .env (keys: GROQ_API_KEY_1..N
or GROQ_API_KEY, OPEN_ROUTER_API_KEY). Native tool calling is used, so no regex
parsing of "Action:" lines is needed.

Groq key rotation: with several keys (GROQ_API_KEY_1, _2, ...), a key that hits its
daily limit for a model is skipped for that model and the next key takes over (even
mid-conversation). QuotaExhausted is raised only when every key is used up.
KEY_LOG records every switch, including the Groq organization named in the error
(Groq limits are per organization: keys from the same organization share one quota).

`chat()` returns the latency of the *successful* request only. Time spent
waiting on rate limits is reported separately, so it never inflates the
"think time" we later replay as the prewarm window.
"""

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))  # agent/.env, whatever the cwd

BACKEND = os.getenv("LLM_BACKEND", "groq").lower()


def _groq_keys() -> list[str]:
    numbered = sorted((k for k in os.environ if re.fullmatch(r"GROQ_API_KEY_\d+", k)),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
    keys = [os.environ[k] for k in numbered if os.environ[k].strip()]
    if not keys and os.getenv("GROQ_API_KEY"):
        keys = [os.environ["GROQ_API_KEY"]]
    if not keys:
        raise ValueError("No Groq key: set GROQ_API_KEY_1 (..._2, ...) or GROQ_API_KEY in agent/.env")
    return keys


if BACKEND == "groq":
    KEYS = _groq_keys()
    BASE_URL, API_KEY, TIMEOUT = "https://api.groq.com/openai/v1", KEYS[0], 120
    DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
elif BACKEND == "openrouter":
    BASE_URL, API_KEY, TIMEOUT = "https://openrouter.ai/api/v1", os.environ["OPEN_ROUTER_API_KEY"], 180
    KEYS = [API_KEY]
    DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
elif BACKEND == "ollama":
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    API_KEY, TIMEOUT = "ollama", 600
    KEYS = [API_KEY]
    DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
else:
    raise ValueError(f"Unknown LLM_BACKEND: {BACKEND!r}. Use 'groq', 'openrouter' or 'ollama'.")

# OpenRouter routes a model to one of several providers; only use providers that
# support every parameter we send (tools), so tool calling never silently breaks.
EXTRA_BODY = {"provider": {"require_parameters": True}} if BACKEND == "openrouter" else None


def _make_client(key: str):
    c = OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=TIMEOUT)
    # Optional LangSmith tracing of raw OpenAI-SDK calls (plain agent + user simulator).
    # Only records anything when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
    try:
        from langsmith.wrappers import wrap_openai
        c = wrap_openai(c)
    except ImportError:
        pass
    return c


_clients = [_make_client(k) for k in KEYS]
_active: dict[str, int] = {}             # model -> index of the key in use
_exhausted: set[tuple[int, str]] = set()  # (key index, model) pairs that hit their daily limit
KEY_LOG: list[dict] = []                  # every key switch / exhaustion, for run records


# ---- Context budget -------------------------------------------------------------
# Groq's free tier rejects any single request above 8K prompt tokens (its tokens-per-minute
# limit). Measured on Groq (gpt-oss): policy + 14 tool definitions = ~2,790 tokens; tool
# results (JSON) ~2.7 characters per token. Before each request fit_context() shrinks the
# copy that is SENT (traces keep everything), oldest first, until the estimate fits:
#   1. old tool results -> compact JSON (lossless, ~8% smaller)
#   2. old tool results -> a neutral note (no invitation to call the tool again: that made
#      agents loop)
#   3. old long agent/customer messages -> their first 300 characters
# The newest messages (KEEP_LAST) are never touched.
CONTEXT_BUDGET = int(os.getenv("LLM_CONTEXT_BUDGET", "6000" if BACKEND == "groq" else "0"))  # 0 = off; Groq rejects ~7K+
FIXED_TOKENS = int(os.getenv("LLM_FIXED_TOKENS", "2800"))   # system prompt + tool definitions (measured)
CHARS_PER_TOKEN = 2.7                                       # conversation (measured on tool JSON; conservative)
KEEP_LAST = 4                                               # newest messages never shrunk
OMITTED = "[older tool result removed to save space]"


def est_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Prompt-token estimate: measured fixed part + the conversation (everything after the system prompt)."""
    return FIXED_TOKENS + int(len(json.dumps(messages[1:])) / CHARS_PER_TOKEN)


def _compact(text: str) -> str:
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except (ValueError, TypeError):
        return text


def fit_context(messages: list[dict], tools: list[dict] | None = None,
                budget: int | None = None, keep_last: int = KEEP_LAST,
                scale: float = 1.0) -> tuple[list[dict], int]:
    """Returns (messages to send, number of messages shrunk).

    `scale` calibrates the estimate to the provider's real count: the caller passes
    real prompt_tokens / est_tokens of its previous request (see agent._chat_within_limit),
    so trimming starts only when the REAL size approaches the budget."""
    budget = CONTEXT_BUDGET if budget is None else budget
    budget = budget / max(0.3, min(2.0, scale)) if budget else budget   # compare in estimate units
    if not budget or est_tokens(messages, tools) <= budget:
        return messages, 0
    out = [dict(m) for m in messages]
    old = range(1, max(1, len(out) - keep_last))          # never the system prompt, never the newest
    changed = set()
    stages = [
        lambda m: m.get("role") == "tool" and _compact(m["content"]),
        lambda m: m.get("role") == "tool" and OMITTED,
        lambda m: m.get("role") in ("assistant", "user") and len(m.get("content") or "") > 300
                  and m["content"][:300] + " [...]",
    ]
    for stage in stages:
        for i in old:
            if est_tokens(out, tools) <= budget:
                return out, len(changed)
            new = stage(out[i])
            if new and new != out[i].get("content"):
                out[i]["content"] = new
                changed.add(i)
    return out, len(changed)


# Waits longer than this (or a "per day" limit) mean the quota is used up: stop instead of sleeping for hours.
MAX_RATE_WAIT_S = float(os.getenv("LLM_MAX_RATE_WAIT", "120"))


class QuotaExhausted(RuntimeError):
    """Rate limit that won't clear soon (e.g. Groq's tokens-per-day limit)."""


@dataclass
class LLMResult:
    message: object       # openai ChatCompletionMessage (.content, .tool_calls)
    latency_s: float      # duration of the successful request = think time
    rate_wait_s: float    # time spent sleeping on 429s (excluded from latency)
    prompt_tokens: int | None = None   # as counted by the provider (for the context budget)


def _org(err: str) -> str | None:
    m = re.search(r"organization `([^`]+)`", err)
    return m.group(1) if m else None


def _next_key(model: str, err: str) -> bool:
    """Mark the current key exhausted for `model`; switch to the next unused key. False if none left."""
    cur = _active.get(model, 0)
    _exhausted.add((cur, model))
    nxt = next((i for i in range(len(_clients)) if (i, model) not in _exhausted), None)
    KEY_LOG.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "exhausted_key": cur + 1,
                    "organization": _org(err), "switched_to": None if nxt is None else nxt + 1})
    if nxt is None:
        return False
    _active[model] = nxt
    print(f"  [llm] key {cur + 1} hit its daily limit for {model} (org {_org(err)}); switching to key {nxt + 1}")
    return True


def chat(messages: list[dict], tools: list[dict] | None = None,
         model: str | None = None, max_retries: int = 8) -> LLMResult:
    model = model or DEFAULT_MODEL
    waited, attempt = 0.0, 0
    while True:
        t0 = time.perf_counter()
        try:
            resp = _clients[_active.get(model, 0)].chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or None,
                temperature=0.0,
                extra_body=EXTRA_BODY,
            )
            usage = getattr(resp, "usage", None)
            return LLMResult(resp.choices[0].message, time.perf_counter() - t0, waited,
                             getattr(usage, "prompt_tokens", None))
        except RateLimitError as e:
            # Groq tells us how long to wait; fall back to exponential backoff
            retry_after = e.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
            msg = str(e).lower()   # Groq: "tokens per day"; OpenRouter free models: "free-models-per-day"
            if wait > MAX_RATE_WAIT_S or "per day" in msg or "per-day" in msg:
                if _next_key(model, str(e)):
                    continue                     # retry right away on the next key
                raise QuotaExhausted(f"all {len(_clients)} key(s) used up for {model}; last: "
                                     f"{wait:.0f}s wait: {str(e)[:200]}") from e
            if attempt == max_retries:
                raise
            print(f"  [llm] rate limited, waiting {wait:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            waited += wait
            attempt += 1
