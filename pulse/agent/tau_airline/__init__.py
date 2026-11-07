"""
tau-bench airline environment, vendored from https://github.com/sierra-research/tau-bench
(MIT License, Copyright (c) 2024 Sierra -- see LICENSE in this folder).

Only change from upstream: tool imports point at the local `tool.py`.
"""

import json
import os

from .tools import ALL_TOOLS

FOLDER = os.path.dirname(__file__)


def load_data() -> dict:
    """Fresh copy of the airline database (flights, reservations, users)."""
    data = {}
    for name in ("flights", "reservations", "users"):
        with open(os.path.join(FOLDER, "data", f"{name}.json"), encoding="utf-8") as f:
            data[name] = json.load(f)
    return data


def load_tasks() -> list[dict]:
    with open(os.path.join(FOLDER, "tasks.json"), encoding="utf-8") as f:
        return json.load(f)


with open(os.path.join(FOLDER, "wiki.md"), encoding="utf-8") as f:
    WIKI = f.read()
