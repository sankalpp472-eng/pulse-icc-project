"""
The interface of a session-based next-tool predictor (pulse-v3 implements it).

A predictor turns a session's hints into Predictions. It owns its per-session state and any learning;
it does not decide what to warm or write PrewarmRequests (predictor/service.py does, through predictor/predictors/pulse_v3.py).

  hint(h)  -> Prediction    one per hint; raise BadHint / OutOfOrder for hints it cannot use
  end(e)   -> dict          the session finished (learn from it, drop its state); raise UnknownSession
  stats()  -> dict          anything worth reporting on GET /healthz
  load(**kw)                classmethod: build a ready predictor (load model files etc.)

Tool names in a Prediction are domain-prefixed ("retail.get_order_details", see pulse.pulse_v3.tools).
"""

from abc import ABC, abstractmethod

from ..contract.hints import End, Hint
from ..contract.prediction import Prediction


class PredictorError(Exception):
    status = 422


class BadHint(PredictorError):
    status = 422


class OutOfOrder(PredictorError):
    status = 409


class UnknownSession(PredictorError):
    status = 404


class NextToolPredictor(ABC):
    name = "predictor"

    @classmethod
    def load(cls, **kw):
        return cls(**kw)

    @abstractmethod
    def hint(self, h: Hint) -> Prediction: ...

    @abstractmethod
    def end(self, e: End) -> dict: ...

    def stats(self) -> dict:
        return {}
