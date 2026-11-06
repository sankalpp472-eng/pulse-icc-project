"""lightgbm: gradient-boosted trees over (workflow, last two tools) + conversation signals. The main model."""

from pulse.predictor.contract import PredictRequest
from pulse.predictor.predictors.action import ActionPickle


class LightGBM(ActionPickle):
    name = "lightgbm"
    key = "gbt"
    description = "LightGBM next-action model: workflow, last two tools and conversation signals"
    inputs = ("trajectory", "signals")

    def predict(self, req: PredictRequest) -> dict[str, float]:
        # without signals the trees would see every signal as 0; the order-2 Markov chain is the honest answer
        if req.signals is None:
            return self.am.proba("ngram", self.workflow(req), req.tools(), {})
        return super().predict(req)
