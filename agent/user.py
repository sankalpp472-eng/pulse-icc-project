"""
The customer side of the conversation.

SimulatedUser -- an LLM plays the customer from the task instruction (tau-bench style).
HumanUser     -- you type the customer's messages yourself (for debugging).
"""

import os
import re
import time

from openai import APIStatusError

from llm import chat
from prompts import USER_SIM_PROMPT

STOP = "###STOP###"


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
        return self._next()


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
