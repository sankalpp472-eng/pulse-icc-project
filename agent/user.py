"""
The customer side of the conversation.

SimulatedUser -- an LLM plays the customer from the task instruction (tau-bench style).
HumanUser     -- you type the customer's messages yourself (for debugging).

A person needs seconds to read the agent's message and type an answer; the LLM answers in ~0.75 s.
USER_DELAY_MEDIAN_S > 0 makes every simulated reply take at least a delay drawn from a lognormal
(median USER_DELAY_MEDIAN_S, sigma USER_DELAY_SIGMA, clipped to 1-60 s, seeded by USER_DELAY_SEED),
the same shape as a customer turn in test/test_knative/w1_replay.py. Default 0: no delay.
"""

import math
import os
import random
import re
import time

from openai import APIStatusError

from llm import chat
from prompts import USER_SIM_PROMPT

STOP = "###STOP###"

DELAY_MEDIAN_S = float(os.getenv("USER_DELAY_MEDIAN_S", "0"))
DELAY_SIGMA = float(os.getenv("USER_DELAY_SIGMA", "0.5"))
_DELAY_RNG = random.Random(int(os.getenv("USER_DELAY_SEED", "252")))


def reply_delay() -> float:
    """How long a person takes to read and answer (0 when USER_DELAY_MEDIAN_S is not set)."""
    if DELAY_MEDIAN_S <= 0:
        return 0.0
    return min(60.0, max(1.0, _DELAY_RNG.lognormvariate(math.log(DELAY_MEDIAN_S), DELAY_SIGMA)))


def _failed_text(e: APIStatusError) -> str | None:
    """gpt-oss on Groq sometimes wraps the customer's line in a fake tool call
    ({"name": "assistant<|channel|>final", "arguments": <the line>}); recover the line."""
    body = e.body if isinstance(e.body, dict) else {}
    gen = (body.get("error", body) or {}).get("failed_generation") or ""
    m = re.search(r'"arguments":\s*(.*)\}\s*$', gen, re.S)
    return m.group(1).strip().strip('"').strip() if m else None


class SimulatedUser:
    def __init__(self, instruction: str):
        self.model = os.getenv("USER_MODEL") or None  # default: same model as the agent
        # Roles are flipped: the simulator is the "assistant", the agent is the "user".
        self.messages = [{"role": "system", "content": USER_SIM_PROMPT.format(instruction=instruction)},
                         {"role": "user", "content": "Hi! How can I help you today?"}]

    def _next(self) -> str:
        # The simulator has no tools, yet gpt-oss on Groq may still emit a malformed tool call
        # (HTTP 400 tool_use_failed): retry, then fall back to the text inside the failed call.
        for attempt in range(1, 4):
            t0 = time.perf_counter()
            try:
                r = chat(self.messages, model=self.model)
                self.last_latency_s = r.latency_s  # customer "think" time (rate-limit waits excluded)
                text = (r.message.content or "").strip()
                break
            except APIStatusError as e:
                if e.status_code != 400 or "tool_use_failed" not in str(e):
                    raise
                recovered = _failed_text(e)
                print(f"  [user sim] malformed tool call from the model ({attempt}/3)")
                if attempt == 3 and recovered:
                    self.last_latency_s = time.perf_counter() - t0
                    text = recovered
                    break
                if attempt == 3:
                    raise
        self.messages.append({"role": "assistant", "content": text})
        return text

    def start(self) -> str:
        return self._next()

    def reply(self, agent_text: str) -> str:
        self.messages.append({"role": "user", "content": agent_text})
        t0 = time.perf_counter()
        text = self._next()
        llm_s = self.last_latency_s
        # a person's reading + typing time; the LLM call counts towards it (total = the longer of the two)
        self.last_delay_s = 0.0 if text.strip() == STOP else reply_delay()
        time.sleep(max(0.0, self.last_delay_s - (time.perf_counter() - t0)))
        self.last_latency_s = max(llm_s, self.last_delay_s)
        return text


class HumanUser:
    def __init__(self, instruction: str):
        print(f"\nYou are the customer. Your goal:\n  {instruction}\n(type {STOP} to end)\n")

    def start(self) -> str:
        return input("you> ")

    def reply(self, agent_text: str) -> str:
        t0 = time.perf_counter()
        text = input("you> ")
        self.last_latency_s = time.perf_counter() - t0
        return text
