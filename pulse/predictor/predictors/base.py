"""
The interface every predictor implements. service.py does the rest the same way for all of them: the prewarm
policy, PrewarmRequests, metrics, the audit log.

  name, description   shown by the UI
  inputs              what it reads from a PredictRequest: ("trajectory",) / + "signals" / + "history"
  domains             tau-bench domains it can answer for
  predict(req)        -> {action: p} over the domain's actions (tools, RESPOND, END), summing to ~1
  fanout(tool, req)   -> pods to start for a prewarmed tool (default 1)
  lookahead(req, ranked) -> {tool: p} to prewarm for the step AFTER the customer's reply, or {} (default)
  health()            -> extra fields for GET /healthz (e.g. which model file, embedding server state)
"""

from pulse.predictor.contract import PredictRequest


class Predictor:
    name = "predictor"
    description = ""
    inputs: tuple = ("trajectory",)
    domains: tuple = ("airline",)
    supports_fanout = False
    supports_lookahead = False
    note = ""                        # shown with the latest answer (e.g. "LLM rate limited")

    def predict(self, req: PredictRequest) -> dict[str, float]:
        raise NotImplementedError

    def fanout(self, tool: str, req: PredictRequest) -> int:
        return 1

    def lookahead(self, req: PredictRequest, ranked: dict[str, float]) -> dict[str, float]:
        return {}

    def health(self) -> dict:
        return {}

    def info(self) -> dict:
        return {"name": self.name, "description": self.description, "inputs": list(self.inputs),
                "domains": list(self.domains),
                "supports": {"fanout": self.supports_fanout, "lookahead": self.supports_lookahead}}
