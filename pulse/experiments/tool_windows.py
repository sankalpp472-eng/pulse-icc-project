"""
Per-tool Knative idle windows from measured traffic (pulse/controller/refresh.py, lever 3).

For every tool: the gaps between its successive calls in the given runs (all episodes together, as the cluster
sees them; calls of one parallel burst, under BURST_S apart, are one use), and the window = the q-quantile of
those gaps clamped to [lo, hi] (refresh.tool_windows). Use runs WITHOUT prewarming (C0): their traffic is the
agents' own, not the predictor's.

  python -m pulse.experiments.tool_windows <run dir> [<run dir> ...] --domain airline [--q 0.8 --lo 20 --hi 60]

Writes the domain's entry of data/tool_windows.json (the other domain's is kept), which run_cloud.sh
(TOOL_WINDOWS=per-tool) and the UI (Run: idle window "per tool") apply to the Knative services.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from pulse import paths
from pulse.controller.refresh import tool_windows

OUT = paths.DATA / "tool_windows.json"
BURST_S = 1.0


def gaps_from_runs(runs: list[Path]) -> dict[str, list[float]]:
    gaps: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        starts: dict[str, list[float]] = defaultdict(list)
        with open(Path(run) / "calls.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                starts[row["tool"]].append(float(row["t_start"]))
        for tool, ts in starts.items():
            ts.sort()
            gaps[tool] += [b - a for a, b in zip(ts, ts[1:]) if b - a >= BURST_S]
            gaps.setdefault(tool, [])
    return gaps


def load(domain: str) -> dict[str, int]:
    """The domain's per-tool windows (seconds), {} if none were made."""
    try:
        return json.loads(OUT.read_text())[domain]["windows"]
    except (OSError, KeyError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--domain", choices=["airline", "retail"], required=True)
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--lo", type=int, default=20)
    ap.add_argument("--hi", type=int, default=60)
    a = ap.parse_args()
    gaps = gaps_from_runs(a.runs)
    win = tool_windows(gaps, a.q, a.lo, a.hi)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[a.domain] = {"windows": dict(sorted(win.items())), "rule": f"q{a.q:g} of re-use gaps, clamped to "
                      f"[{a.lo}, {a.hi}] s (over {a.hi} s -> {a.lo})", "runs": [Path(r).name for r in a.runs],
                      "reuses": {t: len(g) for t, g in sorted(gaps.items())}}
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    for t, w in sorted(win.items()):
        print(f"  {t:32} {w:>3} s   ({len(gaps[t])} re-uses)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
