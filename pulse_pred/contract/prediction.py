"""
Prediction: what pulse-v3 returns for one hint.

  p1     {tool: P(the next step calls it)}          tool names are domain-prefixed ("airline.think");
  p2     {tool: P(the step after that calls it)}    END is not a tool: its mass is p_end
  info   extra fields for the reply (e.g. {"embedding": "ok"})
"""

from dataclasses import dataclass, field


@dataclass
class Prediction:
    session_id: str
    domain: str
    agent_model: str
    step: int                                   # tool steps seen so far in the session
    phase: str                                  # the hint's phase: "start" | "call" | "result"
    last_calls: list[str]                       # phase "call": the step's tools, else []
    p1: dict[str, float]
    p2: dict[str, float] = field(default_factory=dict)
    p_end: float = 0.0
    info: dict = field(default_factory=dict)
