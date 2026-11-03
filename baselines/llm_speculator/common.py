"""
Shared paths and imports for the LLM-speculator baseline.

The step data, split and scoring come from the v2 predictor (tau_data.py, models.py, evaluate.py),
so the speculator and the n-gram / lr / gbdt models are scored on exactly the same steps.
v2 lives in predictor/ once the patch is applied, in predictor_v2/pulse_predictor_v2_files/predictor/
until then; PREDICTOR_DIR overrides both.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA, RESULTS = HERE / "data", HERE / "results"
AGENT_DIR = ROOT / "agent"


def predictor_dir() -> Path:
    if os.getenv("PREDICTOR_DIR"):
        return Path(os.environ["PREDICTOR_DIR"]).resolve()
    for d in (ROOT / "predictor", ROOT / "predictor_v2" / "pulse_predictor_v2_files" / "predictor"):
        if (d / "tau_data.py").exists():
            return d
    raise FileNotFoundError("v2 predictor not found (no tau_data.py in predictor/ or predictor_v2/...); "
                            "set PREDICTOR_DIR")


PREDICTOR_DIR = predictor_dir()
sys.path.insert(0, str(PREDICTOR_DIR))     # tau_data, features, models, evaluate

RESPOND = "RESPOND"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def step_key(s):
    return f"{s['run_id']}#{s['step']}"
