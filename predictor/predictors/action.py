"""
Shared loading for the two predictors that live in one action_model.pkl (train_action.py):
lightgbm (the "gbt" model) and markov (the "ngram" model). Both answer over the domain's tools + RESPOND,
use the pending-ID fan-out rule, and the pickle's second ActionPredictor (.lookahead) predicts the step
after the customer's reply.

Where the model comes from:
  ACTION_MODEL_URI      s3://<bucket>/action-model = the registry's latest version (pulse_pred/model_store.py),
                        downloaded once into predictor/data/models/<version>/
  ACTION_MODEL_VERSION  pin a version (GitOps: pulse-config), instead of latest
  otherwise             the file ACTION_MODEL_PATH (default predictor/data/action_model.pkl, also in the image).
If the registry can't be read the predictor still starts, from that file, and /healthz says why.
"""

import os
import pickle
import sys
from pathlib import Path

import action_model  # noqa: F401  (pickle needs the classes)
from contract import PredictRequest
from predictors.base import Predictor

HERE = Path(__file__).resolve().parents[1]
ACTION_PATH = Path(os.getenv("ACTION_MODEL_PATH", HERE / "data" / "action_model.pkl"))
ACTION_URI = os.getenv("ACTION_MODEL_URI", "")
ACTION_VERSION = os.getenv("ACTION_MODEL_VERSION", "")
_CACHE = {}


def model_file():
    """(path, source, error): the registry's version if ACTION_MODEL_URI is set and reachable, else the local file."""
    if not ACTION_URI:
        return ACTION_PATH, "file", None
    uri = f"{ACTION_URI.rstrip('/')}/{ACTION_VERSION}" if ACTION_VERSION else ACTION_URI
    try:
        root = str(HERE.parent)
        if root not in sys.path:
            sys.path.append(root)
        from pulse_pred import model_store
        dest = Path(os.getenv("ACTION_MODELS_DIR") or HERE / "data" / "models")    # pods: a writable folder
        return Path(model_store.fetch(uri, dest=dest)), "registry", None
    except Exception as e:                    # no credentials / empty registry: serve the file, say so on /healthz
        print(f"action model: {uri} unavailable ({type(e).__name__}: {e}); using {ACTION_PATH}", flush=True)
        return ACTION_PATH, "file (registry unavailable)", f"{type(e).__name__}: {str(e)[:200]}"


def load_action_model():
    if "m" not in _CACHE:
        path, source, error = model_file()
        if not path.exists():
            raise SystemExit(f"{path} is missing: run predictor/build_action_data.py and predictor/train_action.py "
                             "(or set ACTION_MODEL_URI)")
        with open(path, "rb") as f:
            _CACHE["m"] = pickle.load(f)
        _CACHE["info"] = {"model_file": str(path), "model_source": source, "model_uri": ACTION_URI or None,
                          **({"registry_error": error} if error else {})}
    return _CACHE["m"]


class ActionPickle(Predictor):
    """A predictor backed by one model inside action_model.pkl."""
    key = ""                       # model name inside the pickle
    supports_fanout = True

    def __init__(self):
        self.am = load_action_model()
        if self.key not in self.am.models:
            raise SystemExit(f"{_CACHE['info']['model_file']} has no {self.key!r} model (has {self.am.names}); "
                             "retrain it")
        self.look = getattr(self.am, "lookahead", None)
        self.supports_lookahead = self.look is not None and self.key in self.look.models

    def workflow(self, req: PredictRequest) -> str:
        """The workflow id the model knows, else the domain's pooled rows, else unknown (-1 in the encoder)."""
        known = set(self.am.encoder.vocab.get("wf", []))
        for wf in (req.workflow_id, req.domain):
            if wf and wf in known:
                return wf
        return req.workflow_id or req.domain

    def predict(self, req: PredictRequest) -> dict[str, float]:
        return self.am.proba(self.key, self.workflow(req), req.tools(), req.signals or {})

    def fanout(self, tool: str, req: PredictRequest) -> int:
        return self.am.fanout.k(tool, req.signals or {})

    def lookahead(self, req: PredictRequest, ranked: dict[str, float]) -> dict[str, float]:
        if not self.supports_lookahead:
            return {}
        return self.look.proba(self.key, self.workflow(req), req.tools(), req.signals or {})

    def health(self) -> dict:
        return {**_CACHE["info"], "models_in_file": self.am.names,
                "workflows": sorted(self.am.encoder.vocab.get("wf", []))}
