"""
speculator: a fast LLM reads the recent conversation and lists the agent's likely next actions with
probabilities (the Speculative Actions baseline; prompt and parsing in pulse/speculator/).

Settings: SPEC_MODEL (default openai/gpt-oss-20b), SPEC_MAX_TOKENS (1024 for gpt-oss, else 150),
SPEC_TIMEOUT_S (5: a later guess is useless), SPEC_K (4 candidates per answer).
Keys: GROQ_API_KEY_1..N (or GROQ_API_KEY) in .env, round-robin; or SPEC_BASE_URL + SPEC_API_KEY for any
OpenAI-compatible server (a local Ollama / vLLM). A key cooling down after a 429 is skipped; with no key free the
hint gets no guess (empty ranking) instead of waiting, since the answer would arrive after the call.
"""

import os
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError  # noqa: E402

from pulse.predictor.contract import PredictRequest  # noqa: E402
from pulse.predictor.predictors.base import Predictor  # noqa: E402
from pulse.speculator.prompt import build_messages, domain_actions, parse  # noqa: E402
from pulse.speculator.speculate import KeyPool, api_keys  # noqa: E402

MODEL = os.getenv("SPEC_MODEL", "openai/gpt-oss-20b")
TIMEOUT_S = float(os.getenv("SPEC_TIMEOUT_S", "5"))
MAX_TOKENS = int(os.getenv("SPEC_MAX_TOKENS", "1024" if MODEL.startswith("openai/gpt-oss") else "150"))
K = int(os.getenv("SPEC_K", "4"))


class Speculator(Predictor):
    name = "speculator"
    description = f"LLM speculator ({MODEL}) reading the recent conversation"
    inputs = ("trajectory", "history")
    domains = ("airline",)               # prompt.domain_actions wires up the airline tools only

    def __init__(self):
        self.actions = {d: domain_actions(d) for d in self.domains}
        self.pool = KeyPool(api_keys(), timeout=TIMEOUT_S)
        self.stats = {"hints": 0, "answered": 0, "no_key_free": 0, "errors": 0, "unparsed": 0}
        self.note = ""

    def _free_key(self):
        """A key that is neither dead nor cooling down, or None (never sleeps)."""
        with self.pool.lock:
            now, n = time.time(), len(self.pool.clients)
            for off in range(n):
                i = (self.pool.next + off) % n
                if i not in self.pool.dead and self.pool.cool_until[i] <= now:
                    self.pool.next = i + 1
                    return i
        return None

    def _guess(self, messages, names):
        extra = {"response_format": {"type": "json_object"}}
        if MODEL.startswith("openai/gpt-oss"):
            extra["reasoning_effort"] = "low"
        for _ in range(len(self.pool.clients)):
            i = self._free_key()
            if i is None:
                break
            try:
                r = self.pool.clients[i].chat.completions.create(model=MODEL, messages=messages, temperature=0.0,
                                                                 max_tokens=MAX_TOKENS, **extra)
            except RateLimitError as e:
                self.pool.rate_limited(i, e)
                continue
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                self.stats["errors"] += 1
                return {}, f"{type(e).__name__}: {str(e)[:120]}"
            p = parse(r.choices[0].message.content, names)
            if p is None:
                self.stats["unparsed"] += 1
                return {}, "unparsable answer"
            self.stats["answered"] += 1
            return p, ""
        self.stats["no_key_free"] += 1
        return {}, "no LLM key free (rate limited)"

    def predict(self, req: PredictRequest) -> dict[str, float]:
        self.stats["hints"] += 1
        actions = self.actions.get(req.domain) or self.actions["airline"]
        p, self.note = self._guess(build_messages(req.history, req.tools(), actions, k=K),
                                        [n for n, _ in actions])
        return p

    def health(self) -> dict:
        return {"llm": MODEL, "keys": len(self.pool.clients), "dead_keys": len(self.pool.dead), "stats": self.stats}
