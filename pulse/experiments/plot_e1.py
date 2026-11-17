"""
E1, tool-call latency: p50 / p95 / p99 per configuration, and where a cold start's time goes.

  python -m pulse.experiments.plot_e1 results/cloud_runs/<run> [<run> ...] --out results/cloud_runs/e1_<date>

Give it the runs of one workload (same tasks and seed): C1, C0 and C3 with any predictors, any number of trials
each. Runs of the same configuration (same config + predictor + policy) are trials: the bars show their median,
the whiskers min and max (the protocol: 3 trials, median with min/max).

  e1_latency.png   p50 / p95 / p99 per configuration (log scale), C1 = lower bound, C0 = upper bound
  e1_startup.png   startup breakdown of the tool pods started during the runs: scheduling, image pull,
                   init (sandbox + containers), readiness (the tool server until its readiness probe passes);
                   mean seconds per new pod, next to the median cold call the agent saw
  e1.json, e1.md   the numbers (also versus C0 and C1)

The startup breakdown needs runs made after the pod watcher recorded it (pods.csv with scheduled / ready columns).
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pulse.experiments.common import startup_breakdown  # noqa: E402
from pulse.experiments.plot_c0 import INK2, heading, legend_above, pct  # noqa: E402

PCTS = [(50, "p50", "#9ec5f0"), (95, "p95", "#2a78d6"), (99, "p99", "#123a6b")]
PARTS = [("scheduling_s", "scheduling", "#b9b8b2"), ("image_pull_s", "image pull", "#eda100"),
         ("init_s", "init (sandbox + containers)", "#e87ba4"), ("readiness_s", "readiness (tool server start-up)", "#1baf7a")]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(run: Path) -> dict:
    cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    calls = list(csv.DictReader(open(run / "calls.csv", encoding="utf-8")))
    ok = [c for c in calls if c.get("transport_ok", "True") == "True"]
    lat = sorted(float(c["latency_ms"]) for c in ok)
    cold = sorted(float(c["latency_ms"]) for c in ok if c.get("cold") == "True")
    pods = list(csv.DictReader(open(run / "pods.csv", encoding="utf-8"))) if (run / "pods.csv").exists() else []
    t_first = min((float(c["t_start"]) for c in ok), default=None)
    new = []                                 # pods created during the run, with a complete startup record
    for p in pods:
        row = {k: _f(p.get(k)) for k in ("created", "scheduled", "containers_started", "ready", "image_pull_s")}
        b = startup_breakdown(row)
        if b and t_first is not None and row["created"] >= t_first - 120:
            new.append(b)
    pred = (cfg.get("prewarm") or {}).get("predictor") or {}
    pol = pred.get("policy") or {}
    config = cfg.get("config") or summ.get("config") or "?"
    name = pred.get("model") or ""
    if pol.get("name") == "topk" or pol.get("prewarm_policy") == "topk":
        name += f" top-{pol.get('top_k')}"
    if pol.get("lookahead") in (True, 1, "1") or (isinstance(pol.get("lookahead"), dict) and pol["lookahead"].get("on")):
        name += " +look"
    label = f"{config} ({name.strip()})" if config.startswith("C3") and name else config
    return {"run": run.name, "label": label, "config": config, "seed": (cfg.get("replay") or {}).get("seed"),
            "calls": len(ok), "lat": lat, "cold_lat": cold,
            "p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "mean": st.mean(lat) if lat else None,
            "cold_calls": len(cold), "cold_p50": pct(cold, 50) if cold else None,
            "pod_seconds": sum((_f(p.get("last_seen")) or 0) - (_f(p.get("first_seen")) or 0) for p in pods),
            "pods_started": len(pods), "startup": new}


def order(label: str) -> tuple:
    """Lower bound first, upper bound last: C1, the C3 variants, C0."""
    return ({"C1": 0, "C0": 2}.get(label.split(" ")[0], 1), label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [load(r) for r in a.runs]
    groups: dict[str, list] = {}
    for r in runs:
        groups.setdefault(r["label"], []).append(r)
    labels = sorted(groups, key=order)
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = sorted({str(r["seed"]) for r in runs if r["seed"] is not None})
    trials = sorted({len(v) for v in groups.values()})
    sub = (f"W1 replay · same request sequence (seed {', '.join(seeds) or '?'}) · {runs[0]['calls']} tool calls per run"
           f" · {'/'.join(map(str, trials))} trial(s) per configuration (bars: median, whiskers: min-max)")

    agg = {}
    for lab in labels:
        g = groups[lab]
        rec = {"trials": len(g), "runs": [r["run"] for r in g]}
        for key in ("p50", "p95", "p99", "mean", "cold_calls", "pod_seconds", "cold_p50"):
            vals = [r[key] for r in g if r[key] is not None]
            rec[key] = {"median": st.median(vals), "min": min(vals), "max": max(vals)} if vals else None
        pods = [b for r in g for b in r["startup"]]
        rec["startup_pods"] = len(pods)
        rec["startup_mean_s"] = {k: round(st.mean(b[k] for b in pods), 3) for k, *_ in PARTS + [("startup_s",)]} if pods else None
        agg[lab] = rec

    # 1. latency percentiles
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels) + 3), 5.2))
    width = 0.26
    for j, (q, name, colr) in enumerate(PCTS):
        xs = [i + (j - 1) * width for i in range(len(labels))]
        med = [agg[lab][name]["median"] for lab in labels]
        lo = [agg[lab][name]["median"] - agg[lab][name]["min"] for lab in labels]
        hi = [agg[lab][name]["max"] - agg[lab][name]["median"] for lab in labels]
        ax.bar(xs, med, width, color=colr, label=name, yerr=[lo, hi] if max(trials) > 1 else None,
               error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
        for x, v in zip(xs, med):
            ax.text(x, v * 1.08, f"{v:.0f}" if v < 1000 else f"{v / 1000:.2f}s", ha="center", va="bottom", fontsize=7.5,
                    color=INK2, rotation=90 if len(labels) > 5 else 0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels, rotation=15 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_ylabel("Tool-call latency (ms, log scale)")
    legend_above(ax, ncol=3)
    top = heading(fig, "E1 · tool-call latency per configuration (C1 = lower bound, C0 = upper bound)", sub)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(a.out / "e1_latency.png", dpi=160)
    plt.close(fig)

    # 2. startup breakdown (configurations that started pods during the run)
    with_pods = [lab for lab in labels if agg[lab]["startup_mean_s"]]
    if with_pods:
        fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(with_pods) + 4), 5.6))
        bottom = [0.0] * len(with_pods)
        for key, name, colr in PARTS:
            vals = [agg[lab]["startup_mean_s"][key] for lab in with_pods]
            ax.bar(range(len(with_pods)), vals, 0.55, bottom=bottom, color=colr, label=name)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for i, lab in enumerate(with_pods):
            cp = agg[lab]["cold_p50"]
            ax.text(i, bottom[i] + 0.05, f"{bottom[i]:.2f} s\n{agg[lab]['startup_pods']} pods", ha="center", va="bottom",
                    fontsize=8, color=INK2)
            if cp:
                ax.plot([i - 0.32, i + 0.32], [cp["median"] / 1000] * 2, color="#eb6834", lw=2.2,
                        label="median cold call (agent's view)" if i == 0 else None)
        ax.set_xticks(range(len(with_pods)), with_pods, rotation=15 if len(with_pods) > 4 else 0,
                      ha="right" if len(with_pods) > 4 else "center")
        ax.set_ylabel("Seconds per new tool pod (mean)")
        legend_above(ax, ncol=2)
        top = heading(fig, "E1 · where a tool pod's start-up time goes",
                      "mean per new pod (Kubernetes keeps these times to the second) · image pull = 0 if the image is "
                      "already on the node")
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(a.out / "e1_startup.png", dpi=160)
        plt.close(fig)

    # 3. numbers
    base0, base1 = next((agg[l] for l in labels if l == "C0"), None), next((agg[l] for l in labels if l == "C1"), None)
    rel = lambda v, b: None if not (v and b) else round(100 * (v - b) / b, 1)
    lines = ["| Configuration | trials | p50 ms | p95 ms | p99 ms | mean ms | cold calls | pod-s | vs C0 p50 / p99 | startup s (new pods) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        r = agg[lab]
        m = lambda k: r[k]["median"] if r[k] else None
        vs0 = (f"{rel(m('p50'), base0['p50']['median']):+}% / {rel(m('p99'), base0['p99']['median']):+}%"
               if base0 and lab != "C0" else "-")
        s = r["startup_mean_s"]
        startup = f"{s['startup_s']:.2f} ({r['startup_pods']} pods)" if s else "-"
        lines.append(f"| {lab} | {r['trials']} | {m('p50'):.0f} | {m('p95'):.0f} | {m('p99'):.0f} | {m('mean'):.0f} | "
                     f"{m('cold_calls'):.0f} | {m('pod_seconds'):.0f} | {vs0} | {startup} |")
    if base1:
        lines += ["", f"C1 (always warm) pod-seconds: {base1['pod_seconds']['median']:.0f}; NFR: C3 at least 30% fewer."]
    (a.out / "e1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.out / "e1.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved e1_latency.png{', e1_startup.png' if with_pods else ''}, e1.md, e1.json to {a.out}")


if __name__ == "__main__":
    main()
