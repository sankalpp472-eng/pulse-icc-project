"""
Plots for one run folder (results/<date>_<config>_W1-replay_* or _W1-live_*). Saves PNGs into <run>/plots/.

  python test/test_knative/plot_c0.py test/test_knative/results/2026-10-19_0015_C0_W1-replay_taubench-46ep

  1_latency_cdf.png          latency distribution: all / warm / cold, with p50 p95 p99
  2_tool_calls_cold_warm.png calls per tool, split warm / cold
  3_timeline.png             every call's latency over the run + tool pods alive
  4_episode_time.png         where each conversation's time goes (waiting / warm tools / cold starts)
  5_cold_vs_idle_gap.png     chance a call is cold, by time since that tool was last used
  6_example_episodes.png     three conversations drawn as timelines
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Reference palette (dataviz skill): slot 1 blue = warm, slot 2 orange = cold; text inks; surface
WARM, COLD = "#2a78d6", "#eb6834"
NEUTRAL, INK, INK2, MUTED, SURFACE, GRID = "#b9b8b2", "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e8e7e3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "text.color": INK,
})


def heading(fig, title, sub):
    """Title + subtitle at figure level (never clipped, never squeezes the plot)."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.28 / h, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.58 / h, sub, fontsize=9.5, color=INK2, va="top")
    return 1 - 0.85 / h        # top of the plotting area, for tight_layout(rect=...)


def legend_above(ax, **kw):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=kw.pop("ncol", 3), borderaxespad=0, **kw)


def load(run: Path):
    rows = list(csv.DictReader(open(run / "calls.csv", encoding="utf-8")))
    for r in rows:
        r["lat"] = float(r["latency_ms"]) / 1000
        r["cold"] = r["cold"] == "True"
        r["t"] = float(r["t_start"])
        r["gap"] = float(r["idle_gap_s"]) if r.get("idle_gap_s") else None
    eps = [json.loads(l) for l in open(run / "episodes.jsonl", encoding="utf-8")]
    pods = list(csv.DictReader(open(run / "pods.csv", encoding="utf-8"))) if (run / "pods.csv").exists() else []
    cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    return rows, eps, pods, cfg


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def tag(cfg) -> str:
    k = cfg.get("knative", {})
    return f"{cfg.get('config', 'C0')} · idle window {', '.join(k.get('tool_window', ['?']))} · " \
           f"{cfg.get('workload', 'W1')}"


def fig_cdf(rows, cfg, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [("All calls", [r["lat"] for r in rows], INK2, 1.6, "--"),
              ("Warm", [r["lat"] for r in rows if not r["cold"]], WARM, 2.0, "-"),
              ("Cold", [r["lat"] for r in rows if r["cold"]], COLD, 2.0, "-")]
    for name, xs, color, lw, ls in series:
        xs = sorted(xs)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", color=color, lw=lw, ls=ls, label=f"{name} ({len(xs)})")
    allx = series[0][1]
    for q in (50, 95, 99):
        v = pct(allx, q)
        ax.axvline(v, color=MUTED, lw=0.8, ls=":")
        ax.text(v, 0.02 + (q - 50) / 400, f" p{q} {v * 1000:.0f} ms" if v < 1 else f" p{q} {v:.1f} s",
                color=INK2, fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Tool-call latency seen by the agent (seconds, log scale)")
    ax.set_ylabel("Share of calls at or below this latency")
    ax.set_ylim(0, 1.02)
    legend_above(ax)
    cold_share = sum(r["cold"] for r in rows) / len(rows)
    top = heading(fig, "Cold starts make the tail: p50 stays fast, p95/p99 jump to seconds",
                  f"{tag(cfg)}  ·  {len(rows)} calls, {cold_share:.0%} cold")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_tools(rows, cfg, out):
    by = {}
    for r in rows:
        d = by.setdefault(r["tool"], [0, 0])
        d[1 if r["cold"] else 0] += 1
    tools = sorted(by, key=lambda t: sum(by[t]))
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(tools) + 1.6))
    y = range(len(tools))
    warm = [by[t][0] for t in tools]
    cold = [by[t][1] for t in tools]
    ax.barh(y, warm, color=WARM, height=0.68, edgecolor=SURFACE, linewidth=2, label="Warm")
    ax.barh(y, cold, left=warm, color=COLD, height=0.68, edgecolor=SURFACE, linewidth=2, label="Cold")
    for i, t in enumerate(tools):
        n = sum(by[t])
        ax.text(n + max(warm) * 0.01, i, f"{by[t][1] / n:.0%} cold  ({n})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), tools)
    ax.set_xlabel("Tool calls")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sum(v) for v in by.values()) * 1.22)
    legend_above(ax, ncol=2)
    top = heading(fig, "Frequent tools stay warm; rare write and branch tools are mostly cold",
                  f"{tag(cfg)}  ·  calls per tool, split by warm / cold")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_timeline(rows, pods, cfg, out):
    t0 = min(r["t"] for r in rows)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for cold, color, name in ((False, WARM, "Warm"), (True, COLD, "Cold")):
        pts = [r for r in rows if r["cold"] == cold]
        a1.scatter([(r["t"] - t0) / 60 for r in pts], [r["lat"] for r in pts], s=22, color=color,
                   edgecolor=SURFACE, linewidth=0.8, label=f"{name} ({len(pts)})", zorder=3)
    a1.set_yscale("log")
    a1.set_ylabel("Call latency (s, log)")
    legend_above(a1, ncol=2)
    top = heading(fig, "Cold starts happen throughout the run, whenever a tool has gone idle",
                  f"{tag(cfg)}  ·  each dot is one tool call")
    if pods:
        t_end = max(float(p["last_seen"]) for p in pods)
        steps = [t0 + i for i in range(int(t_end - t0) + 2)]
        alive = [sum(float(p["first_seen"]) <= s <= float(p["last_seen"]) for p in pods) for s in steps]
        a2.fill_between([(s - t0) / 60 for s in steps], alive, step="post", color=WARM, alpha=0.25, lw=0)
        a2.step([(s - t0) / 60 for s in steps], alive, where="post", color=WARM, lw=1.6)
        a2.set_ylabel("Tool pods alive")
        a2.text(0.995, 0.92, f"{len(pods)} pods started · "
                f"{sum(float(p['last_seen']) - float(p['first_seen']) for p in pods):,.0f} pod-seconds",
                transform=a2.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    a2.set_xlabel("Minutes since the run started")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_episodes(rows, eps, cfg, out):
    data = []
    for e in eps:
        mine = [r for r in rows if r["episode"] == e["episode"]]
        cold = sum(r["lat"] for r in mine if r["cold"])
        warm = sum(r["lat"] for r in mine if not r["cold"])
        dur = e.get("duration_s") or (cold + warm)
        data.append((e["episode"], e.get("task_id"), max(0.0, dur - cold - warm), warm, cold))
    data.sort(key=lambda d: d[4])
    fig, ax = plt.subplots(figsize=(9.5, 0.2 * len(data) + 1.8))
    y = range(len(data))
    wait = [d[2] for d in data]
    warm = [d[3] for d in data]
    cold = [d[4] for d in data]
    ax.barh(y, wait, color=NEUTRAL, height=0.72, edgecolor=SURFACE, linewidth=1.5, label="Waiting between calls (LLM + customer)")
    ax.barh(y, warm, left=wait, color=WARM, height=0.72, edgecolor=SURFACE, linewidth=1.5, label="Warm tool calls")
    ax.barh(y, cold, left=[a + b for a, b in zip(wait, warm)], color=COLD, height=0.72, edgecolor=SURFACE,
            linewidth=1.5, label="Cold tool calls")
    ax.set_yticks(list(y), [f"{d[0]} · task {d[1]}" for d in data], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seconds per conversation")
    legend_above(ax, fontsize=9)
    shares = [d[4] / (d[2] + d[3] + d[4]) for d in data if d[2] + d[3] + d[4] > 0]
    top = heading(fig, "Cold starts take a visible bite out of most conversations",
                  f"{sum(c > 0 for c in cold)} of {len(data)} conversations had a cold call · time spent on cold "
                  f"calls: median {statistics.median(shares):.0%}, max {max(shares):.0%} of the conversation"
                  + "\n" + tag(cfg))
    fig.tight_layout(rect=(0, 0, 1, top - 0.15 / fig.get_figheight()))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_gap(rows, cfg, out):
    bins = [(0, 15), (15, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 300), (300, 1e9)]
    labels = ["<15s", "15-30s", "30-60s", "60-90s", "90-120s", "2-3 min", "3-5 min", ">5 min", "first use\nin run"]
    groups = [[r for r in rows if r["gap"] is not None and lo <= r["gap"] < hi] for lo, hi in bins]
    groups.append([r for r in rows if r["gap"] is None])
    share = [sum(r["cold"] for r in g) / len(g) if g else 0 for g in groups]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(groups))
    ax.bar(x, share, color=[COLD if i < len(bins) else NEUTRAL for i in x], width=0.7, edgecolor=SURFACE, linewidth=2)
    for i, (s, g) in enumerate(zip(share, groups)):
        ax.text(i, s + 0.02, f"{s:.0%}\nn={len(g)}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Calls that were cold")
    ax.set_xlabel("Time since that tool's previous call ended (idle gap)")
    ax.grid(axis="x", visible=False)
    window = ", ".join(cfg.get("knative", {}).get("tool_window", ["?"]))
    top = heading(fig, f"A tool goes cold once it has been idle longer than the idle window ({window})",
                  f"{tag(cfg)}  ·  share of cold calls per idle-gap range")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_examples(rows, eps, cfg, out):
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode"], []).append(r)
    ranked = sorted(by_ep, key=lambda e: (sum(r["cold"] for r in by_ep[e]), len(by_ep[e])))
    picks = [ranked[-1], ranked[len(ranked) // 2], next((e for e in ranked if not any(r["cold"] for r in by_ep[e])), ranked[0])]
    task = {e["episode"]: e.get("task_id") for e in eps}
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 1.55 * len(picks) + 1.2), sharex=True)
    for ax, ep in zip(axes, picks):
        calls = sorted(by_ep[ep], key=lambda r: r["t"])
        start = calls[0]["t"] - 2
        k = 0
        for r in calls:
            if r["cold"]:
                ax.barh(0, r["lat"], left=r["t"] - start, height=0.5, color=COLD, edgecolor=SURFACE, linewidth=1)
                ax.text(r["t"] - start + r["lat"] / 2, 0.42 if k % 2 == 0 else -0.62, r["tool"].replace("_", " "),
                        ha="center", fontsize=7.5, color=INK2)
                k += 1
            else:   # warm calls are ~0.05 s: draw a fixed-width tick so they stay visible
                ax.barh(0, 0.35, left=r["t"] - start, height=0.5, color=WARM, linewidth=0)
        n_cold = sum(r["cold"] for r in calls)
        ax.set_yticks([])
        ax.set_ylim(-0.9, 0.9)
        ax.grid(axis="y", visible=False)
        ax.text(0, 1.0, f"{ep} · task {task.get(ep)} · {len(calls)} calls: {len(calls) - n_cold} warm, {n_cold} cold "
                        f"({sum(r['lat'] for r in calls if r['cold']):.1f} s waiting on cold starts)",
                transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")
    axes[-1].set_xlabel("Seconds since the conversation's first tool call (orange width = cold-call latency; "
                        "blue ticks = warm calls; gaps = LLM / customer)")
    fig.legend(handles=[Patch(color=WARM, label="Warm call"), Patch(color=COLD, label="Cold call (labelled)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    top = heading(fig, "Inside a conversation: cold starts hit the first use of each tool",
                  f"{tag(cfg)}  ·  most cold calls, a median one, and one with none")
    fig.tight_layout(rect=(0, 0.35 / fig.get_figheight(), 1, top - 0.1 / fig.get_figheight()))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((Path(__file__).parent / "results").glob("*_W1-replay_*"))[-1]
    rows, eps, pods, cfg = load(run)
    out = run / "plots"
    out.mkdir(exist_ok=True)
    fig_cdf(rows, cfg, out / "1_latency_cdf.png")
    fig_tools(rows, cfg, out / "2_tool_calls_cold_warm.png")
    fig_timeline(rows, pods, cfg, out / "3_timeline.png")
    fig_episodes(rows, eps, cfg, out / "4_episode_time.png")
    fig_gap(rows, cfg, out / "5_cold_vs_idle_gap.png")
    fig_examples(rows, eps, cfg, out / "6_example_episodes.png")
    print(f"saved {len(list(out.glob('*.png')))} plots to {out}")


if __name__ == "__main__":
    main()
