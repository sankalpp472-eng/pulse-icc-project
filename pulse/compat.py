"""
Loading model files pickled under older module names. A pickle stores each class by its module path, so models
trained before the code moved into the pulse package (e.g. the pulse-v3 versions already in the S3 registry) name
modules that no longer exist. load() maps them to where the classes live now.
"""

import pickle

_EXACT = {   # pulse-v3, before it was split into subpackages
    "pulse_pred.predictors": "pulse.pulse_v3.predictors.pulse_v3.models",
    "pulse_pred.graph": "pulse.pulse_v3.predictors.pulse_v3.graph",
    "pulse_pred.state": "pulse.pulse_v3.predictors.pulse_v3.state",
    "pulse_pred.online": "pulse.pulse_v3.predictors.pulse_v3.online",
    "pulse_pred.embed": "pulse.pulse_v3.predictors.pulse_v3.embed",
    "pulse_pred.tau": "pulse.pulse_v3.traces.tau",
    # lightgbm + markov, from when predictor/ was a flat folder on sys.path
    "action_model": "pulse.predictor.action_model",
    "signals": "pulse.predictor.signals",
}


def module_name(module: str) -> str:
    if module in _EXACT:
        return _EXACT[module]
    if module.startswith("pulse_pred."):
        return "pulse.pulse_v3." + module[len("pulse_pred."):]
    return module


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module_name(module), name)


def load(path):
    """The object pickled in `path`, whatever module names it was saved under."""
    with open(path, "rb") as f:
        return Unpickler(f).load()
