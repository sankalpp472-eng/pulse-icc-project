"""
Compare runs of the SAME workload under different configs (C0, C1, later C3).

  python test/test_knative/plot_compare.py results/w1_replay_A results/w1_replay_B [...] --out results/compare_C0_C1

  1_latency_cdf.png     latency distribution of every config on one chart
  2_percentiles.png     p50 / p95 / p99 per config
  3_cost.png            pod-seconds (resource cost) per config
  4_cold_starts.png     real cold starts (a new pod appeared) per config
  comparison.json       the numbers behind the charts

Configs are read from each run's run_config.json; runs should share the seed (same request sequence).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_c0 import GRID, INK, INK2, SURFACE, heading, legend_above, pct  # noqa: E402,F401

# Fixed colour per config (reference palette slots): C0 orange, C1 blue, C3 aqua
CONFIG_COLOR = {"C0": "#eb6834", "C1": "#2a78d6", "C3": "#1baf7a"}
EXTRA = ["#eda100", "#e87ba4", "#008300"]


def load(run: Path) -> dict:
    cfg = json.loads((run / "run_config.json").read_text())
    rows = list(csv.DictReader(open(run / "calls.csv", encoding="utf-8")))
    lat = [float(r["latency_ms"]) / 1000 for r in rows if r.get("transport_ok", "True") == "True"]
    pods = list(csv.DictReader(open(run / "pods.csv", encoding="utf-8"))) if (run / "pods.csv").exists() else []
    real_cold = sum(r.get("cold_by_pod") == "True" for r in rows)
    slow_warm = sum(r.get("cold_by_pod") != "True" and float(r["latency_ms"]) > 1000 for r in rows)
    return {"run": run.name, "config": cfg.get("config", "?"), "seed": cfg.get("replay", {}).get("seed"),
            "window": ", ".join(cfg.get("knative", {}).get("tool_window", [])), "lat": lat,
            "calls": len(rows), "real_cold_starts": real_cold, "slow_warm_calls_over_1s": slow_warm,
            "pods_started": len(pods),
            "pod_seconds": round(sum(float(p["last_seen"]) - float(p["first_seen"]) for p in pods))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [load(r) for r in a.runs]
    a.out.mkdir(parents=True, exist_ok=True)
    color = {}
    for r in runs:
        color[r["run"]] = CONFIG_COLOR.get(r["config"]) or EXTRA[len(color) % len(EXTRA)]
    label = lambda r: f"{r['config']}"
    seeds = {r["seed"] for r in runs}
    sub = (f"W1 replay · same request sequence (seed {', '.join(map(str, seeds))}) · idle window "
           f"{runs[0]['window']} · {runs[0]['calls']} tool calls each")

    # 1. latency CDF
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        xs = sorted(r["lat"])
        ax.step(xs, [(i + 1) / len(xs) for i in range(len(xs))], where="post", lw=2, color=color[r["run"]],
                label=f"{label(r)}: p50 {pct(xs, 50) * 1000:.0f} ms, p99 {pct(xs, 99):.2f} s")
    ax.set_xscale("log")
    ax.set_xlabel("Tool-call latency (seconds, log scale)")
    ax.set_ylabel("Share of calls at or below this latency")
    ax.set_ylim(0, 1.02)
    legend_above(ax, ncol=len(runs))
    top = heading(fig, "Keeping tools warm removes the multi-second tail", sub)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(a.out / "1_latency_cdf.png", dpi=160)
    plt.close(fig)

    # 2. percentiles
    qs = (50, 95, 99)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        vals = [pct(r["lat"], q) for q in qs]
        xs = [i + (k - (len(runs) - 1) / 2) * w for i in range(len(qs))]
        ax.bar(xs, vals, width=w * 0.92, color=color[r["run"]], edgecolor=SURFACE, linewidth=2, label=label(r))
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.12, f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s", ha="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(qs)), [f"p{q}" for q in qs])
    ax.set_ylabel("Latency (s, log scale)")
    ax.grid(axis="x", visible=False)
    legend_above(ax, ncol=len(runs))
    top = heading(fig, "Same median, very different tail: p95 and p99 per config", sub)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(a.out / "2_percentiles.png", dpi=160)
    plt.close(fig)

    # 3. cost and 4. cold starts (simple bars, one value per config)
    for fname, key, ylabel, title, fmt in [
        ("3_cost.png", "pod_seconds", "Tool pod-seconds during the run", "Always-warm costs about twice the pod time", "{:,.0f}"),
        ("4_cold_starts.png", "real_cold_starts", "Calls served by a newly started pod", "Cold starts per config", "{:,.0f}")]:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        names = [label(r) for r in runs]
        vals = [r[key] for r in runs]
        ax.bar(names, vals, color=[color[r["run"]] for r in runs], width=0.55, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, fmt.format(v), ha="center", fontsize=10, color=INK)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.15 or 1)
        ax.grid(axis="x", visible=False)
        top = heading(fig, title, sub)
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(a.out / fname, dpi=160)
        plt.close(fig)

    summary = [{k: v for k, v in r.items() if k != "lat"} |
               {"p50_ms": round(pct(r["lat"], 50) * 1000, 1), "p95_ms": round(pct(r["lat"], 95) * 1000, 1),
                "p99_ms": round(pct(r["lat"], 99) * 1000, 1), "mean_ms": round(sum(r["lat"]) / len(r["lat"]) * 1000, 1)}
               for r in runs]
    (a.out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved 4 plots + comparison.json to {a.out}")


if __name__ == "__main__":
    main()
