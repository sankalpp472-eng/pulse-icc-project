"""
LLM client -- Groq, OpenRouter or Ollama, all through the OpenAI-compatible API.

Switch with LLM_BACKEND=groq|openrouter|ollama in .env (keys: GROQ_API_KEY,
OPEN_ROUTER_API_KEY). Native tool calling is used, so no regex parsing of
"Action:" lines is needed.

`chat()` returns the latency of the *successful* request only. Time spent
waiting on rate limits is reported separately, so it never inflates the
"think time" we later replay as the prewarm window.
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))  # agent/.env, whatever the cwd

BACKEND = os.getenv("LLM_BACKEND", "groq").lower()

if BACKEND == "groq":
    BASE_URL, API_KEY, TIMEOUT = "https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"], 120
    DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
elif BACKEND == "openrouter":
    BASE_URL, API_KEY, TIMEOUT = "https://openrouter.ai/api/v1", os.environ["OPEN_ROUTER_API_KEY"], 180
    DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
elif BACKEND == "ollama":
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    API_KEY, TIMEOUT = "ollama", 600
    DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
else:
    raise ValueError(f"Unknown LLM_BACKEND: {BACKEND!r}. Use 'groq', 'openrouter' or 'ollama'.")

# OpenRouter routes a model to one of several providers; only use providers that
# support every parameter we send (tools), so tool calling never silently breaks.
EXTRA_BODY = {"provider": {"require_parameters": True}} if BACKEND == "openrouter" else None

_client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0, timeout=TIMEOUT)

# Optional LangSmith tracing of raw OpenAI-SDK calls (plain agent + user simulator).
# Only records anything when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
try:
    from langsmith.wrappers import wrap_openai
    _client = wrap_openai(_client)
except ImportError:
    pass


# Waits longer than this (or a "per day" limit) mean the quota is used up: stop instead of sleeping for hours.
MAX_RATE_WAIT_S = float(os.getenv("LLM_MAX_RATE_WAIT", "120"))


class QuotaExhausted(RuntimeError):
    """Rate limit that won't clear soon (e.g. Groq's tokens-per-day limit)."""


@dataclass
class LLMResult:
    message: object       # openai ChatCompletionMessage (.content, .tool_calls)
    latency_s: float      # duration of the successful request = think time
    rate_wait_s: float    # time spent sleeping on 429s (excluded from latency)


def chat(messages: list[dict], tools: list[dict] | None = None,
         model: str | None = None, max_retries: int = 8) -> LLMResult:
    waited = 0.0
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = _client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=messages,
                tools=tools or None,
                temperature=0.0,
                extra_body=EXTRA_BODY,
            )
            return LLMResult(resp.choices[0].message, time.perf_counter() - t0, waited)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            # Groq tells us how long to wait; fall back to exponential backoff
            retry_after = e.response.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
            msg = str(e).lower()   # Groq: "tokens per day"; OpenRouter free models: "free-models-per-day"
            if wait > MAX_RATE_WAIT_S or "per day" in msg or "per-day" in msg:
                raise QuotaExhausted(f"rate limit needs {wait:.0f}s wait: {str(e)[:200]}") from e
            print(f"  [llm] rate limited, waiting {wait:.1f}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            waited += wait
