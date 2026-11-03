"""
Build Pulse_Concurrent_Workloads.xlsx: the tau-bench airline tasks whose agent runs look up several
items at once (parallel fan-out), with their workflow graphs.

A parallel step = back-to-back calls of the same read-only tool with different arguments, no customer
turn in between, and no call using a value that first appeared in another call of the step
(test/test_knative/w1_replay.py: fanout_groups, the grouping `w1_replay.py --fanout` replays).

    agent/.venv/Scripts/python workloads/build_concurrent_workloads_xlsx.py     # from the project root
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

ROOT = Path(__file__).resolve().parents[1]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(ROOT / "test" / "test_knative"))
import w1_replay as w  # noqa: E402

OUT = ROOT / "workloads" / "Pulse_Concurrent_Workloads.xlsx"

# ordered by share of calls inside parallel steps
TASKS = {
    33: "Cancel every future reservation that has a flight over 3 h; upgrade the others to business where possible.",
    31: "Booked two flights on May 17 by mistake: cancel the ATL -> JFK one (or else the other one).",
    30: "Find bookings on the same day in the profile and cancel one duplicate for each such day.",
    2: "Downgrade all business-class reservations to economy and report the total money saved.",
    3: "Change the Houston -> Denver return to the fastest same-day trip, add a checked bag, pay with the smallest gift card.",
    28: "Cancel all upcoming flights, even those without a refund.",
    34: "Cancel reservations XEHM4B and 59XX6W (upgrade basic economy first), then total the cost of the other upcoming flights.",
}
# why the agent fans out with this tool
WHY = {
    "get_reservation_details": "look up every reservation in the user's profile at once",
    "search_direct_flight": "search several dates / routes at once",
    "search_onestop_flight": "search several dates / routes at once",
    "calculate": "price several options at once",
    "get_user_details": "look up several users at once",
}


def task_graph(tid: int) -> dict:
    trs = [t for t in w.load_traces(None) if t["task_id"] == tid and t["calls"]]
    edges, widths, nodes_order = defaultdict(Counter), defaultdict(list), ["START"]
    runs_with = Counter()
    calls = fan = 0
    for tr in trs:
        gs = w.fanout_groups(tr)
        seq = ["START"]
        for g in gs:
            tool = tr["calls"][g[0]]["tool"]
            if len(g) > 1:
                widths[tool].append(len(g))
                seq.append(f"{tool} ×N (parallel)")
            else:
                seq.append(tool)
        seq.append("END")
        for s in seq:
            if s not in nodes_order:
                nodes_order.append(s)
        for a, b in zip(seq, seq[1:]):
            edges[a][b] += 1
        for t in {tr["calls"][g[0]]["tool"] for g in gs if len(g) > 1}:
            runs_with[t] += 1
        calls += len(tr["calls"])
        fan += sum(len(g) for g in gs if len(g) > 1)
    return {"runs": len(trs), "calls": calls, "fan": fan, "edges": edges, "widths": widths,
            "runs_with": runs_with, "nodes": [n for n in nodes_order if n != "END"],
            "models": Counter(t["source_model"] for t in trs)}


def label(node: str, g: dict) -> str:
    if node.endswith("(parallel)"):
        ws = g["widths"][node.split(" ")[0]]
        return f"{node.split(' ')[0]} ×N in parallel (N = {min(ws)}–{max(ws)})" if min(ws) != max(ws) \
            else f"{node.split(' ')[0]} ×{ws[0]} in parallel"
    return node


def when(node_to: str, k: int, total: int, g: dict) -> str:
    share = f"{k} of {total}"
    if node_to == "END":
        return f"END: conversation ends ({share})"
    tool = node_to.split(" ")[0]
    if node_to.endswith("(parallel)"):
        return f"{label(node_to, g)}: {WHY.get(tool, 'independent calls at once')} ({share})"
    if node_to == "transfer_to_human_agents":
        return f"{tool}: request outside policy ({share})"
    return f"{tool} ({share})"


def main():
    wb = Workbook()
    head_font, head_fill = Font(name="Arial", bold=True, size=11, color="FFFFFFFF"), PatternFill("solid", fgColor="FF1F3864")
    body, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    fills = [PatternFill("solid", fgColor="FFEEF3FA"), PatternFill("solid", fgColor="FFFFFFFF")]
    par_font = Font(name="Arial", size=10, bold=True, color="FFC00000")
    wrap = Alignment(wrap_text=True, vertical="top")
    thin = Border(bottom=Side(style="thin", color="FFBFBFBF"))

    def header(ws, cols, widths):
        for i, (c, wd) in enumerate(zip(cols, widths), 1):
            cell = ws.cell(row=1, column=i, value=c)
            cell.font, cell.fill = head_font, head_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            ws.column_dimensions[cell.column_letter].width = wd
        ws.freeze_panes = "A2"

    graphs = {tid: task_graph(tid) for tid in TASKS}

    # ---- sheet 1: one row per (task, step) of the workflow graph
    ws = wb.active
    ws.title = "1 Concurrent tasks"
    header(ws, ["Task no", "Small description", "After this tool", "Agent may call next", "When each branch happens"],
           [9, 48, 40, 62, 80])
    r = 2
    for k, (tid, desc) in enumerate(TASKS.items()):
        g, fill = graphs[tid], fills[k % 2]
        first = r
        for node in g["nodes"]:
            nxt = g["edges"][node].most_common()
            total = sum(c for _, c in nxt)
            row = [tid, desc, label(node, g), ", ".join(label(n, g) for n, _ in nxt),
                   "\n".join(when(n, c, total, g) for n, c in nxt)]
            for i, v in enumerate(row, 1):
                cell = ws.cell(row=r, column=i, value=v)
                cell.font, cell.fill, cell.alignment, cell.border = body, fill, wrap, thin
            ws.cell(row=r, column=1).font = bold
            if "parallel" in str(row[2]):
                ws.cell(row=r, column=3).font = par_font
            r += 1
        if r - first > 1:   # task no and description once per task block
            ws.merge_cells(start_row=first, end_row=r - 1, start_column=1, end_column=1)
            ws.merge_cells(start_row=first, end_row=r - 1, start_column=2, end_column=2)
    ws.auto_filter.ref = f"A1:E{r - 1}"
    note = r + 1
    notes = [
        "How to read: each row is one step of the task's workflow graph, built from the 12 recorded tau-bench runs "
        "of the task (4 gpt-4o + 8 claude-3.5-sonnet). '(k of n)' = how often that branch was taken from this step.",
        "Red 'xN in parallel' steps: the agent calls the same read-only tool N times at once (one per reservation, date, ...). "
        "w1_replay.py --fanout fires these calls together; Knative (containerConcurrency 1) then starts N pods.",
        "Parallel step = back-to-back calls of one read-only tool, no customer turn between them, different arguments, and no "
        "call using a value that first appeared in another call's result (checked by re-running each trace on a copy of the database).",
        "Source: tau-bench historical trajectories (sierra-research/tau-bench), airline domain. "
        "Built by workloads/build_concurrent_workloads_xlsx.py.",
    ]
    for i, t in enumerate(notes):
        c = ws.cell(row=note + i, column=1, value=t)
        c.font = Font(name="Arial", size=9, italic=True)
        ws.merge_cells(start_row=note + i, end_row=note + i, start_column=1, end_column=5)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[note + i].height = 28

    # ---- sheet 2: fan-out summary per task
    ws2 = wb.create_sheet("2 Fan-out summary")
    header(ws2, ["Task no", "Small description", "Recorded runs", "Tool calls (all runs)", "Calls inside parallel steps",
                 "Share of calls in parallel steps", "Parallel steps (tool: widths seen)", "Runs with a parallel step"],
           [9, 48, 10, 12, 13, 13, 46, 30])
    for k, (tid, desc) in enumerate(TASKS.items()):
        g, rr = graphs[tid], k + 2
        par = "\n".join(f"{t}: {', '.join(map(str, sorted(ws_)))}" for t, ws_ in g["widths"].items())
        runs = "\n".join(f"{t}: {n} of {g['runs']}" for t, n in g["runs_with"].items())
        row = [tid, desc, g["runs"], g["calls"], g["fan"], f"=IF(D{rr}=0,0,E{rr}/D{rr})", par, runs]
        for i, v in enumerate(row, 1):
            c = ws2.cell(row=rr, column=i, value=v)
            c.font, c.fill, c.alignment, c.border = body, fills[k % 2], wrap, thin
        ws2.cell(row=rr, column=1).font = bold
        ws2.cell(row=rr, column=6).number_format = "0%"
    tr_ = len(TASKS) + 2
    ws2.cell(row=tr_, column=2, value="All tasks").font = bold
    for col in "CDE":
        c = ws2[f"{col}{tr_}"]
        c.value, c.font = f"=SUM({col}2:{col}{tr_ - 1})", bold
    ws2[f"F{tr_}"].value, ws2[f"F{tr_}"].font = f"=IF(D{tr_}=0,0,E{tr_}/D{tr_})", bold
    ws2[f"F{tr_}"].number_format = "0%"
    c = ws2.cell(row=tr_ + 2, column=1, value="Widths: how many calls one parallel step fires at once (one entry per step seen). "
                 "Counts come from the recorded traces (hardcoded by the build script); the shares are formulas.")
    c.font = Font(name="Arial", size=9, italic=True)
    ws2.merge_cells(start_row=tr_ + 2, end_row=tr_ + 2, start_column=1, end_column=8)

    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
