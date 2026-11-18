"""
Regression tests for issue #30: loading a model file must not run code named inside it. compat.load() now only lets
a model file name the classes our models are made of (and a few named NumPy / copyreg helpers); anything else is
refused before its module is imported.

The hostile files here are built by hand and harmless (the payload would only create a marker file); the tests check
that it is never created. No cluster, Docker or LLM needed.
"""

import pickle
from collections import Counter, OrderedDict, defaultdict

import pytest

from pulse import compat, paths


def load_bytes(tmp_path, data: bytes):
    f = tmp_path / "model.pkl"
    f.write_bytes(data)
    return compat.load(f)


def calls(module: str, name: str, *args: str) -> bytes:
    """A protocol-0 pickle that calls module.name(*args) when loaded: the classic malicious model file."""
    return (f"c{module}\n{name}\n(" + "".join(f"V{a}\n" for a in args) + "tR.").encode()


@pytest.mark.parametrize("module,name", [
    ("os", "system"), ("posix", "system"), ("nt", "system"), ("subprocess", "Popen"), ("builtins", "eval"),
    ("builtins", "exec"), ("builtins", "getattr"), ("builtins", "__import__"), ("pulse.ui.backend", "BatchRun"),
    ("pulse.compat", "load"), ("pulse.predictor.action_model", "Encoder.__init__"), ("numpy", "load"),
    ("numpy", "memmap"), ("numpy.testing", "assert_equal"), ("pulse.pulse_v3.train", "main"),
    # classes defined elsewhere that an allowed module merely imports
    ("sklearn.feature_extraction.text", "partial"), ("numpy.ctypeslib._ctypeslib", "_ndptr_base"),
])
def test_hostile_names_are_refused(tmp_path, module, name):
    marker = tmp_path / "pwned"
    with pytest.raises(compat.UnsafeModelFile):
        load_bytes(tmp_path, calls(module, name, f"touch {marker}"))
    assert not marker.exists()


def test_refusal_names_what_was_refused(tmp_path):
    with pytest.raises(compat.UnsafeModelFile, match=r"os\.system"):
        load_bytes(tmp_path, calls("os", "system", "true"))


def test_plain_data_still_loads(tmp_path):
    obj = {"counts": Counter("aab"), "order": OrderedDict(a=1), "dd": defaultdict(int, {"x": 2}),
           "s": {1, 2}, "fs": frozenset({3}), "t": (1.5, "x", None, b"\x00")}
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        assert load_bytes(tmp_path, pickle.dumps(obj, protocol=proto)) == obj


def test_numpy_and_sklearn_objects_load(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    X = TfidfVectorizer().fit(["book a flight", "cancel my reservation"])
    lr = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
    obj = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "f": np.float64(2.5), "sp": sparse.csr_matrix(np.eye(3)),
           "tfidf": X, "lr": lr, "rng": np.random.default_rng(0)}
    out = load_bytes(tmp_path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    assert (out["a"] == obj["a"]).all() and out["f"] == 2.5 and (out["sp"] != obj["sp"]).nnz == 0
    assert out["tfidf"].vocabulary_ == X.vocabulary_
    assert out["lr"].predict([[3.0]])[0] == 1
    assert out["rng"].random() == np.random.default_rng(0).random()


def test_old_module_names_still_map(tmp_path):
    pytest.importorskip("numpy")
    from pulse.predictor.action_model import Encoder
    assert load_bytes(tmp_path, b"caction_model\nEncoder\n.") is Encoder


def test_shipped_model_loads():
    """models/action_model.pkl (lightgbm + markov) still loads through the restricted loader."""
    pytest.importorskip("lightgbm")
    pred = compat.load(paths.MODELS / "action_model.pkl")
    assert type(pred).__name__ == "ActionPredictor"
