"""
The predictor service's request schema: what the agent (or the harness around it) reports during a
session. Every predictor consumes these, whatever it does with them.
"""

from pydantic import BaseModel, Field


class Call(BaseModel):
    tool: str                                   # tool name as the agent calls it, e.g. "get_order_details"
    args: dict = Field(default_factory=dict)


class Hint(BaseModel):
    session_id: str
    domain: str                                 # "airline" | "retail"
    phase: str                                  # "start" | "call" | "result"
    agent_model: str = "default"                # predictors may keep per-agent-model state (fixed at "start")
    user_text: str | None = None                # the user's latest message, when it changed
    agent_text: str | None = None
    calls: list[Call] = Field(default_factory=list)       # phase "call": the step's tool calls
    results: list[str] = Field(default_factory=list)      # phase "result": one result per call
    selfhint: list[tuple[str, float]] | None = None       # optional: the agent's own next-tool guess


class End(BaseModel):
    session_id: str
