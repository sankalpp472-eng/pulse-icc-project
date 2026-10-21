"""
The customer side of the conversation.

SimulatedUser -- an LLM plays the customer from the task instruction (tau-bench style).
HumanUser     -- you type the customer's messages yourself (for debugging).
"""

import os
import time

from llm import chat
from prompts import USER_SIM_PROMPT

STOP = "###STOP###"


class SimulatedUser:
    def __init__(self, instruction: str):
        self.model = os.getenv("USER_MODEL") or None  # default: same model as the agent
        # Roles are flipped: the simulator is the "assistant", the agent is the "user".
        self.messages = [{"role": "system", "content": USER_SIM_PROMPT.format(instruction=instruction)},
                         {"role": "user", "content": "Hi! How can I help you today?"}]

    def _next(self) -> str:
        r = chat(self.messages, model=self.model)
        self.last_latency_s = r.latency_s          # customer "think" time (rate-limit waits excluded)
        text = (r.message.content or "").strip()
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
