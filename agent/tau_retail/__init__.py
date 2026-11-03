"""
tau-bench retail environment, vendored from https://github.com/sierra-research/tau-bench
(MIT License, Copyright (c) 2024 Sierra -- see LICENSE in this folder).

Only changes from upstream: tool imports point at the local `tool.py`, and the 115 test tasks
(tasks_test.py) are stored as tasks.json, in the same shape as tau_airline/tasks.json.
"""

import json
import os

from .tools import ALL_TOOLS

FOLDER = os.path.dirname(__file__)


def load_data() -> dict:
    """Fresh copy of the retail database (orders, products, users)."""
    data = {}
    for name in ("orders", "products", "users"):
        with open(os.path.join(FOLDER, "data", f"{name}.json"), encoding="utf-8") as f:
            data[name] = json.load(f)
    return data


def load_tasks() -> list[dict]:
    with open(os.path.join(FOLDER, "tasks.json"), encoding="utf-8") as f:
        return json.load(f)


with open(os.path.join(FOLDER, "wiki.md"), encoding="utf-8") as f:
    WIKI = f.read()
