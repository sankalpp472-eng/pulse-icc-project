"""
The one contract every predictor serves (predictor/service.py), whatever it is inside.

Input (POST /predict): a PredictRequest. trajectory always; signals and history only when the predictor reads
them. Every predictor says what it reads in GET /info ("inputs"), and the agent, the replay runner and the UI
send exactly that (agent/hints.py), so the conversation leaves the agent only for predictors that need it.

  trajectory  tools called so far, oldest first (the agent sends its last 8)
  signals     numbers and yes/no flags about the conversation, no raw text (predictor/signals.py)
  history     the recent conversation as compact messages {role, content, calls?, name?} (agent/hints.py)

Output: a PredictionOut. ranked = every action of the domain with its probability (the domain's tools,
RESPOND = the agent answers the customer, END = the session ends); prewarm = the tools the shared policy picked
(predictor/policy.py), each with k pods.
"""

from functools import cache
from typing import Annotated, Optional

from pydantic import BaseModel, Field

INPUTS = ("trajectory", "signals", "history")
RESPOND, END = "RESPOND", "END"
NO_PREWARM = {RESPOND, END}            # not functions: never prewarmed
START = "START"                        # current_tool before the first call

# Request size limits (issue #24): a tool name is a short identifier, and senders keep only the last 8 calls
MAX_NAME = 64
MAX_TRAJECTORY = 64
MAX_HISTORY = 100
Name = Annotated[str, Field(max_length=MAX_NAME)]


@cache
def domain_tools(domain: str) -> frozenset[str] | None:
    """The tool names of a tau-bench domain (None = not a domain). RESPOND and END are actions, not tools."""
    if domain == "airline":
        from pulse.agent.tau_airline import ALL_TOOLS
    elif domain == "retail":
        from pulse.agent.tau_retail import ALL_TOOLS
    else:
        return None
    return frozenset(t.get_info()["function"]["name"] for t in ALL_TOOLS)


def unknown_tools(domain: str, names: list[str]) -> list[str]:
    """The names that are not tools of `domain` (in order, once each)."""
    known = domain_tools(domain) or frozenset()
    return list(dict.fromkeys(n for n in names if n not in known))


class PredictRequest(BaseModel):
    workflow_id: Optional[Name] = Field(None, description="W1, W2 ... (unknown / missing -> the domain's pooled model)")
    episode_id: str = Field("", max_length=256)
    domain: Name = Field("airline", description="must be one of the predictor's domains (GET /info), else 422")
    trajectory: list[Name] = Field(default_factory=list, max_length=MAX_TRAJECTORY,
                                   description="tools called so far, in order: tools of the domain only "
                                               "(not RESPOND / END), else 422")
    current_tool: Optional[Name] = Field(None, description="optional; used if trajectory is empty")
    ts: Optional[float] = None
    signals: Optional[dict[str, float]] = Field(None, description="predictor/signals.py (inputs: signals)")
    history: list[dict] = Field(default_factory=list, max_length=MAX_HISTORY,
                                description="compact messages (inputs: history)")

    def tools(self) -> list[str]:
        if self.trajectory:
            return list(self.trajectory)
        return [self.current_tool] if self.current_tool and self.current_tool != START else []


class ActionP(BaseModel):
    tool: str                 # a tool, RESPOND or END
    p: float
    k: Optional[int] = None   # prewarm list only: pods to start (fan-out width = desiredReplicas)


class PredictionOut(BaseModel):
    workflow_id: str
    episode_id: str
    model: str                # the predictor that answered
    ranked: list[ActionP]
    prewarm: list[ActionP]    # never END or RESPOND
    lookahead: list[ActionP] = []    # the part of `prewarm` the lookahead chose (predictors that support it)
    theta: float
    ttl_s: int
    latency_ms: float
    inputs_used: list[str] = []      # what this answer was computed from
    note: str = ""
