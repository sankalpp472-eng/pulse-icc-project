"""
Tool backend for the agent.

Right now tools run in-process against a local copy of the tau-bench airline
database. Later, an MCP backend with the same interface (`schemas`, `call`)
will send each call to its OpenFaaS function instead -- the agent loop does
not change.
"""

import hashlib
import json
import time

from tau_airline import ALL_TOOLS, load_data

# Tools that end the conversation when called (same as tau-bench)
TERMINATE_TOOLS = {"transfer_to_human_agents"}


class LocalToolBackend:
    def __init__(self):
        self.data = load_data()  # each episode gets its own fresh DB copy
        self._tools = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
        self.schemas = [t.get_info() for t in ALL_TOOLS]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> tuple[str, float]:
        """Run a tool. Returns (observation, latency_seconds)."""
        t0 = time.perf_counter()
        if name not in self._tools:
            out = f"Error: unknown tool {name}"
        else:
            try:
                out = self._tools[name].invoke(data=self.data, **args)
            except Exception as e:  # bad args from the LLM -> tell it, don't crash
                out = f"Error: {type(e).__name__}: {e}"
        return out, time.perf_counter() - t0

    def data_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()
