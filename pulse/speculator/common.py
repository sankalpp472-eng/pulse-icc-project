"""
Shared paths and helpers for the LLM speculator's offline evaluation (steps.py, speculate.py, evaluate.py).

  DATA      the held-out steps the speculator is scored on (steps.py writes them), data/speculator/
  RESULTS   its runs and comparisons, results/speculator/
"""

import json

from pulse import paths

ROOT = paths.ROOT
DATA = paths.DATA / "speculator"
RESULTS = paths.RESULTS / "speculator"
RESPOND = "RESPOND"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def step_key(s):
    return f"{s['run_id']}#{s['step']}"
