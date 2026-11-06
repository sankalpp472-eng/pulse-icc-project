"""
The four predictors, behind one interface (base.Predictor), loaded by name (PREDICTOR=<name>):

  lightgbm    LightGBM next-action model over the trajectory + signals (predictor/action_model.py, the main model)
  markov      order-2 Markov chain over actions with back-off: P(next | last two tools) (action_model.NGramModel)
  speculator  an LLM reads the recent conversation and guesses the next action (baselines/llm_speculator/)
  pulse_v3    conversation text + tool graph + state flags, online adapter, S3 model registry (pulse_pred/)

Old names still accepted: gbt / lgbm -> lightgbm, ngram_action -> markov.
"""

import importlib

REGISTRY = {
    "lightgbm": "predictors.lightgbm:LightGBM",
    "markov": "predictors.markov:Markov",
    "speculator": "predictors.speculator:Speculator",
    "pulse_v3": "predictors.pulse_v3:PulseV3",
}
ALIASES = {"gbt": "lightgbm", "lgbm": "lightgbm", "ngram_action": "markov", "ngram": "markov", "pulse-v3": "pulse_v3"}
DEFAULT = "lightgbm"


def canonical(name: str) -> str:
    name = (name or DEFAULT).strip()
    name = ALIASES.get(name, name)
    if name not in REGISTRY:
        raise KeyError(f"unknown predictor {name!r}; use one of {sorted(REGISTRY)}")
    return name


def load(name: str):
    """The predictor called `name`, ready to answer (model files loaded)."""
    module, cls = REGISTRY[canonical(name)].split(":")
    return getattr(importlib.import_module(module), cls)()
