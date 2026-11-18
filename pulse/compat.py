"""
Loading model files safely, including ones pickled under older module names.

Safety (issue #30): a pickle can name any importable callable, and unpickling calls it, so an unrestricted load of a
tampered model file runs arbitrary code. load() only lets a model file name what our models are made of:
  * classes from Pulse's model packages and the parts of the libraries the models are built with (scikit-learn's
    text features, linear models and preprocessing, LightGBM's Booster, SciPy's sparse matrices, NumPy's array,
    dtype, scalar and random-generator types, collections, datetime, pathlib), and
  * a short list of named helper functions those classes pickle through (NumPy's array and random-generator
    reconstructors, copyreg's).
Anything else (os.system, builtins.eval, subprocess.Popen, a function or a dotted attribute path) raises
UnsafeModelFile before its module is even imported. A model that needs a new class fails with a message naming it;
add it below only if it is plain data.

Old names: a pickle stores each class by its module path, so models trained before the code moved into the pulse
package (e.g. the pulse-v3 versions already in the S3 registry) name modules that no longer exist. module_name()
maps them to where the classes live now.
"""

import _compat_pickle
import importlib
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

# modules (and their submodules) whose CLASSES a model file may name. Narrow on purpose: unpickling can CALL a class
# (cls(*args)), so a class whose constructor does work (pulse.ui.backend.BatchRun runs commands, numpy.memmap writes
# files) must stay out. Pulse: only the model packages; NumPy: only its array, dtype and scalar types (_numpy_ok).
CLASS_MODULES = ("pulse.predictor.action_model", "pulse.predictor.signals", "pulse.pulse_v3",
                 "sklearn.feature_extraction", "sklearn.linear_model", "sklearn.preprocessing",
                 "lightgbm.basic", "lightgbm.sklearn", "scipy.sparse", "numpy", "collections", "datetime", "pathlib")
NOT_MODULES = ("pulse.pulse_v3.train",)                 # scripts, not model code
# builtins a model file may name: data types only (never eval, exec, getattr, __import__, ...)
BUILTINS = {"set", "frozenset", "slice", "range", "complex", "bytearray", "object", "list", "dict", "tuple",
            "bytes", "str", "int", "float", "bool"}
# the only functions a model file may name: how NumPy arrays, scalars and random generators, and plain objects, pickle
FUNCTIONS = {
    ("copyreg", "_reconstructor"),
    ("_codecs", "encode"),
    *((f"numpy.{core}.multiarray", f) for core in ("core", "_core") for f in ("_reconstruct", "scalar")),
    *((f"numpy.{core}.numeric", "_frombuffer") for core in ("core", "_core")),
    *(("numpy.random._pickle", f) for f in ("__generator_ctor", "__bit_generator_ctor", "__randomstate_ctor")),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
}


class UnsafeModelFile(pickle.UnpicklingError):
    """The model file names something a model is not made of: refused (it could run code when loaded)."""


def module_name(module: str) -> str:
    if module in _EXACT:
        return _EXACT[module]
    if module.startswith("pulse_pred."):
        return "pulse.pulse_v3." + module[len("pulse_pred."):]
    return module


def _in(module: str, roots) -> bool:
    return any(module == r or module.startswith(r + ".") for r in roots)


def _numpy_ok(cls) -> bool:
    """NumPy classes a model is made of: the plain ndarray (not memmap or other subclasses), dtypes, scalar types,
    and the random generators."""
    import numpy as np
    return (cls is np.ndarray or issubclass(cls, (np.dtype, np.generic))
            or issubclass(cls, (np.random.Generator, np.random.BitGenerator, np.random.RandomState,
                                  np.random.SeedSequence)))


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Python 2 names that pickle protocols 0-2 may use (__builtin__.long -> builtins.int), as pickle itself maps them
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        elif module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]
        module = module_name(module)
        what = f"{module}.{name}"
        if (module, name) in FUNCTIONS:
            return getattr(importlib.import_module(module), name)
        if module == "builtins":
            if name in BUILTINS:
                return getattr(importlib.import_module("builtins"), name)
            raise UnsafeModelFile(f"model file refused: it names {what}")
        # a plain class name in an allowed library; no dotted paths (x.__init__.__globals__) and no dunders
        if (not _in(module, CLASS_MODULES) or _in(module, NOT_MODULES)
                or "." in name or name.startswith("__")):
            raise UnsafeModelFile(f"model file refused: it names {what}, which a model is not made of")
        obj = getattr(importlib.import_module(module), name)
        if not isinstance(obj, type):
            raise UnsafeModelFile(f"model file refused: {what} is a function, not a model class")
        if _in(module, ("numpy",)) and not _numpy_ok(obj):
            raise UnsafeModelFile(f"model file refused: it names {what}, which a model is not made of")
        return obj


def load(path):
    """The model pickled in `path`, whatever module names it was saved under. Raises UnsafeModelFile if the file
    names anything a model is not made of (issue #30)."""
    with open(path, "rb") as f:
        return Unpickler(f).load()
