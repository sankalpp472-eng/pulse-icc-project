"""
Step 2: build one workflow graph per workflow (FR-3).

A workflow graph says, for each tool, which tools are ALLOWED to come next.
It is what the agent developer hands us ("the workflow definition"). We take it
from the Edges sheet of the workload spreadsheet (the team's declared graph).

    python predictor/build_graphs.py             # run from the repo root

Writes predictor/graphs/W1.json, W2.json and airline.json (= W1 and W2 merged,
used when a hint has an unknown / missing workflow id). Format:

    {"workflow_id": "W1",
     "nodes": ["START", "get_user_details", ..., "END"],
     "edges": {"START": ["get_user_details", ...], "get_user_details": [...], ...}}
"""

import json
from collections import defaultdict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "workloads" / "Pulse_Workloads.xlsx"
OUT = Path(__file__).resolve().parent / "graphs"

WORKLOADS = ("W1", "W2")
POOLED = "airline"


def to_json(wf_id, edges):
    nodes = sorted({a for a in edges} | {b for v in edges.values() for b in v})
    return {"workflow_id": wf_id, "nodes": nodes,
            "edges": {a: sorted(v) for a, v in sorted(edges.items())}}


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    rows = wb["Edges"].iter_rows(min_row=2, values_only=True)

    graphs = {w: defaultdict(set) for w in (*WORKLOADS, POOLED)}
    for row in rows:
        wf, src, dst = row[0], row[1], row[2]
        if wf in WORKLOADS and src and dst:
            graphs[wf][src].add(dst)
            graphs[POOLED][src].add(dst)

    OUT.mkdir(parents=True, exist_ok=True)
    for wf, edges in graphs.items():
        g = to_json(wf, edges)
        (OUT / f"{wf}.json").write_text(json.dumps(g, indent=2))
        n_edges = sum(len(v) for v in g["edges"].values())
        print(f"{wf:8}: {len(g['nodes']):2} nodes, {n_edges:3} allowed transitions -> graphs/{wf}.json")


if __name__ == "__main__":
    main()
