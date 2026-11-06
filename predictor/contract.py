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

from typing import Optional

from pydantic import BaseModel, Field

INPUTS = ("trajectory", "signals", "history")
RESPOND, END = "RESPOND", "END"
NO_PREWARM = {RESPOND, END}            # not functions: never prewarmed


class PredictRequest(BaseModel):
    workflow_id: Optional[str] = Field(None, description="W1, W2 ... (unknown / missing -> the domain's pooled model)")
    episode_id: str = ""
    domain: str = "airline"
    trajectory: list[str] = Field(default_factory=list, description="tools called so far, in order")
    current_tool: Optional[str] = Field(None, description="optional; used if trajectory is empty")
    ts: Optional[float] = None
    signals: Optional[dict[str, float]] = Field(None, description="predictor/signals.py (inputs: signals)")
    history: list[dict] = Field(default_factory=list, description="compact messages (inputs: history)")

    def tools(self) -> list[str]:
        if self.trajectory:
            return list(self.trajectory)
        return [self.current_tool] if self.current_tool and self.current_tool != "START" else []


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
