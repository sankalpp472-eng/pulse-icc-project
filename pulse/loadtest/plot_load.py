"""
Plots for load runs (results/load_runs/*): how latency, throughput, failures and cold share change with load,
one line per run (e.g. C0 vs C3, or gbt vs pulse_v3).

    python -m pulse.loadtest.plot_load results/load_runs/<run> [<run> ...] --out results/load_runs/compare_<...>

Writes 1_latency_vs_load.png, 2_throughput_vs_load.png, 3_failures_cold_vs_load.png and comparison.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

XLABEL = {"predictor": "load (users = hints/s, or hints/s)", "tools": "load (users, or burst size)",
          "episodes": "concurrent agents"}


def label(run: Path, s: dict) -> str:
    pred = (s.get("params") or {}).get("predictor_url") and "+hints" or ""
    return f"{s['config']} {s['engine']}{pred} · {run.name[:15]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [(Path(r), json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))) for r in a.runs]
    bad = [r.name for r, s in runs if s.get("kind") != "load"]
    if bad:
        raise SystemExit(f"not load runs: {bad} (experiment runs: pulse.experiments.plot_compare)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = sorted({s["target"] for _, s in runs})
    xl = XLABEL.get(targets[0], "load") if len(targets) == 1 else "load"

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, s in runs:
        x = [st["load"] for st in s["stages"]]
        line, = ax.plot(x, [st["p95"] for st in s["stages"]], marker="o", label=f"p95 · {label(r, s)}")
        ax.plot(x, [st["p50"] for st in s["stages"]], ls="--", color=line.get_color(), alpha=0.6, label=f"p50 · {label(r, s)}")
    if "predictor" in targets:
        ax.axhline(50, color="red", lw=1, ls=":", label="NFR-4: p95 < 50 ms")
    ax.set(xlabel=xl, ylabel="latency (ms)", title="Latency vs load")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "1_latency_vs_load.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, s in runs:
        ax.plot([st["load"] for st in s["stages"]], [st["rps"] for st in s["stages"]], marker="o", label=label(r, s))
    ax.set(xlabel=xl, ylabel="requests / s", title="Throughput vs load")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "2_throughput_vs_load.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, s in runs:
        x = [st["load"] for st in s["stages"]]
        line, = ax.plot(x, [st["failures"] / max(1, st["requests"]) for st in s["stages"]], marker="x",
                        label=f"failed · {label(r, s)}")
        if s["target"] != "predictor":
            ax.plot(x, [st["cold_share"] for st in s["stages"]], marker="o", ls="--", color=line.get_color(),
                    label=f"cold (> 1 s) · {label(r, s)}")
    ax.set(xlabel=xl, ylabel="share of requests", title="Failures and cold calls vs load", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "3_failures_cold_vs_load.png", dpi=130)

    (out / "comparison.json").write_text(json.dumps(
        {r.name: {k: s.get(k) for k in ("target", "engine", "config", "requests", "failures", "rps", "latency_ms",
                                        "cold_share", "stages")} for r, s in runs}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
