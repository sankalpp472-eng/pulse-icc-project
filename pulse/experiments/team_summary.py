"""
Summary tables and plots of one experiment of the team scripts (experiments/team/), from its manifest:
median over trials (with min / max) per domain and arm.

    python -m pulse.experiments.team_summary --manifest results/team/<id>/manifest.tsv --exp topk --out <dir>

The manifest's columns: exp domain arm trial status result_dir start end. The kind of run is read from its folder:
  replay     summary.json + calls.csv (+ controller.jsonl for C3): latency p50/p95/p99, cold calls, pod-seconds,
             new pods and their start-up, concurrent episodes, prewarms (warmed / skipped / wasted)
  load       summary.json with "stages" (pulse.loadtest): the predictor's latency per hint rate
  coldstart  coldstart.csv: per tool cold / warm latency and the new pod's start-up breakdown
Writes <exp>.md, <exp>.json and <exp>_<domain>.png.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def json_events(path: Path) -> list[dict]:
    """Every JSON object in a log, also when two threads printed onto one line ({...}{...}): the controller's
    log is several threads' print() calls, and kubectl logs can join them."""
    dec, out = json.JSONDecoder(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        s, i = line.strip(), 0
        while i < len(s):
            try:
                obj, i = dec.raw_decode(s, i)
            except ValueError:
                break
            if isinstance(obj, dict):
                out.append(obj)
            while i < len(s) and s[i].isspace():
                i += 1
    return out


def prewarm_stats(run: Path) -> dict:
    """Prewarms of a C3 run (controller.jsonl): warmed, skipped, and wasted = warmed for an episode that did not
    call the tool within the prewarm's effective TTL (or 60 s)."""
    ctl = run / "controller.jsonl"
    if not ctl.exists():
        return {}
    calls = defaultdict(list)
    with open(run / "calls.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            calls[(r["episode"], r["tool"])].append(float(r["t_start"]))
    warmed = skipped = wasted = pods = 0
    for e in json_events(ctl):
        if e.get("event") == "skipped":
            skipped += 1
        if e.get("event") != "warmed" or not e.get("requested_at"):
            continue
        warmed += 1
        pods += int(e.get("pings") or 1)
        t0, ttl = ts(e["requested_at"]), float(e.get("effective_ttl_s") or 60)
        if not any(t0 <= t <= t0 + ttl for t in calls.get((e.get("episode"), e["tool"]), [])):
            wasted += 1
    return {"prewarms": warmed, "prewarm_pings": pods, "skipped": skipped, "wasted": wasted,
            "wasted_pct": round(100 * wasted / warmed, 1) if warmed else None}


def replay_metrics(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    lat = s.get("latency_ms", {}).get("all", {})
    conc = []
    with open(run / "calls.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("concurrent_episodes"):
                conc.append(float(r["concurrent_episodes"]))
    st = (s.get("startup") or {}).get("startup_s") or {}
    return {"p50_ms": lat.get("p50"), "p95_ms": lat.get("p95"), "p99_ms": lat.get("p99"), "mean_ms": lat.get("mean"),
            "calls": s.get("tool_calls"), "cold_calls": s.get("cold_calls"),
            "cold_pct": round(100 * s["cold_share"], 1) if s.get("cold_share") is not None else None,
            "pod_s": s.get("pod_seconds_seen"), "pods_started": s.get("pods_started"),
            "startup_s": st.get("median") if isinstance(st, dict) else None,
            "concurrent_mean": round(sum(conc) / len(conc), 1) if conc else None,
            "concurrent_max": max(conc) if conc else None, **prewarm_stats(run)}


def resource_metrics(prom: Path, run_key: str) -> dict:
    """Resource use of one run from the Prometheus export (prom_export.py, step 15 s): tool pods' CPU core-seconds
    and mean memory, the Pulse services' (predictor + controller) CPU core-seconds, mean node CPU busy share and
    the most tool pods at once."""
    def rows(name):
        f = prom / f"{name}.csv"
        if not f.exists():
            return []
        with open(f, newline="", encoding="utf-8") as fh:
            return [(int(r["time"]), json.loads(r["labels"]), float(r["value"])) for r in csv.DictReader(fh)
                    if r["run"] == run_key and r["value"] not in ("NaN", "+Inf", "-Inf")]
    step = 15
    cpu, mem = rows("pod_cpu_cores"), rows("pod_memory_bytes")
    if not cpu and not mem:
        return {}
    tool = lambda lb: "-deployment-" in lb.get("pod", "")                               # noqa: E731 (Knative pods)
    pulse = lambda lb: lb.get("pod", "").startswith(("pulse-predictor", "pulse-controller"))  # noqa: E731
    mem_t = defaultdict(float)
    for t, lb, v in mem:
        if tool(lb):
            mem_t[t] += v
    pods_t = defaultdict(float)
    for t, _, v in rows("tool_pods"):
        pods_t[t] += v
    node = [v for _, _, v in rows("node_cpu_busy")]
    return {"tool_cpu_core_s": round(sum(v for _, lb, v in cpu if tool(lb)) * step, 1),
            "tool_mem_mib": round(sum(mem_t.values()) / len(mem_t) / 2**20, 1) if mem_t else None,
            "pulse_cpu_core_s": round(sum(v for _, lb, v in cpu if pulse(lb)) * step, 1),
            "node_cpu_busy_pct": round(100 * sum(node) / len(node), 1) if node else None,
            "tool_pods_peak": max(pods_t.values()) if pods_t else None}


def load_metrics(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    out = {}
    for st in s.get("stages", []):
        out[f"{float(st['load']):g}"] = {k: st.get(k) for k in ("requests", "failures", "rps", "p50", "p95", "p99", "max")}
    return {"rates": out}


def coldstart_metrics(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return {"cold_ms": s["cold_ms"]["median"], "cold_p95_ms": s["cold_ms"].get("p95"),
            "warm_ms": s["warm_ms"]["median"], "per_tool": s["per_tool"], "startup": s.get("startup")}


def kind_of(run: Path) -> str:
    if (run / "coldstart.csv").exists():
        return "coldstart"
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return "load" if "stages" in s else "replay"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    groups, order, kinds = defaultdict(list), [], {}
    for r in csv.reader(open(a.manifest, encoding="utf-8"), delimiter="\t"):
        if len(r) < 6 or r[0] != a.exp or r[4] != "done" or not Path(r[5]).is_dir():
            continue
        run = Path(r[5])
        kind = kind_of(run)
        m = {"replay": replay_metrics, "load": load_metrics, "coldstart": coldstart_metrics}[kind](run)
        if kind == "replay":
            m.update(resource_metrics(out / "prometheus", f"{r[1]}/{r[2]}/t{r[3]}"))
        key = (r[1], r[2])
        if key not in groups:
            order.append(key)
        groups[key].append({"trial": r[3], "run": run.name, **m})
        kinds[key] = kind
    if not groups:
        print(f"no finished runs of {a.exp}")
        return 0

    table, lines = {}, [f"# {a.exp}", ""]
    for domain in dict.fromkeys(d for d, _ in order):
        keys = [k for k in order if k[0] == domain]
        kind = kinds[keys[0]]
        lines += [f"## {domain}", ""]
        if kind == "replay":
            cols = ["p50_ms", "p95_ms", "p99_ms", "cold_calls", "cold_pct", "pod_s", "pods_started", "startup_s",
                    "concurrent_mean", "prewarms", "wasted_pct", "skipped"]
            res = ["tool_cpu_core_s", "tool_mem_mib", "pulse_cpu_core_s", "node_cpu_busy_pct", "tool_pods_peak"]
            if any(x.get(c) is not None for k in keys for x in groups[k] for c in res):
                cols += res
            lines += ["| arm | trials | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
            for k in keys:
                runs = groups[k]
                agg = {c: {"median": med([x.get(c) for x in runs]), "min": min([x[c] for x in runs if x.get(c) is not None], default=None),
                           "max": max([x[c] for x in runs if x.get(c) is not None], default=None)} for c in cols}
                table[f"{domain}/{k[1]}"] = {"trials": len(runs), "metrics": agg, "runs": runs}
                lines.append(f"| {k[1]} | {len(runs)} | " + " | ".join(str(agg[c]["median"]) for c in cols) + " |")
            fig, axes = plt.subplots(1, 4, figsize=(17, 4))
            arms = [k[1] for k in keys]
            for ax, c in zip(axes, ["p50_ms", "p95_ms", "p99_ms", "pod_s"]):
                mids = [table[f"{domain}/{x}"]["metrics"][c]["median"] or 0 for x in arms]
                lo = [m - (table[f"{domain}/{x}"]["metrics"][c]["min"] or m) for m, x in zip(mids, arms)]
                hi = [(table[f"{domain}/{x}"]["metrics"][c]["max"] or m) - m for m, x in zip(mids, arms)]
                ax.bar(range(len(arms)), mids, yerr=[lo, hi], capsize=3, color="#4C72B0")
                ax.set_xticks(range(len(arms)))
                ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
                ax.set_title(c.replace("_", " "))
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle(f"{a.exp} | {domain} | median over trials (bars: min-max)")
        elif kind == "load":
            lines += ["| arm | rate/s | trials | p50 ms | p95 ms | p99 ms | max ms | failures |", "|---|---|---|---|---|---|---|---|"]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for k in keys:
                runs = groups[k]
                rates = sorted({float(x) for r in runs for x in r["rates"]})
                agg = {}
                for rate in rates:
                    st = [r["rates"].get(f"{rate:g}") for r in runs if r["rates"].get(f"{rate:g}")]
                    agg[f"{rate:g}"] = {q: med([x.get(q) for x in st]) for q in ("p50", "p95", "p99", "max")}
                    agg[f"{rate:g}"]["failures"] = sum(x.get("failures") or 0 for x in st)
                    lines.append(f"| {k[1]} | {rate:g} | {len(st)} | {agg[f'{rate:g}']['p50']} | {agg[f'{rate:g}']['p95']} | "
                                 f"{agg[f'{rate:g}']['p99']} | {agg[f'{rate:g}']['max']} | {agg[f'{rate:g}']['failures']} |")
                table[f"{domain}/{k[1]}"] = {"trials": len(runs), "rates": agg, "runs": runs}
                for ax, q in zip(axes, ("p50", "p95", "p99")):
                    ax.plot(rates, [agg[f"{r:g}"][q] for r in rates], marker="o", label=k[1])
            for ax, q in zip(axes, ("p50", "p95", "p99")):
                ax.set_title(f"{q} (ms)")
                ax.set_xlabel("hints per second")
                ax.set_yscale("log")
                ax.grid(alpha=0.3, which="both")
            axes[1].axhline(50, ls="--", c="grey", lw=1)
            axes[0].legend()
            fig.suptitle(f"{a.exp} | {domain} | predictor API latency (median over trials; dashed: 50 ms)")
        else:   # coldstart
            runs = [r for k in keys for r in groups[k]]
            tools = sorted({t for r in runs for t in r["per_tool"]})
            per = {t: {q: med([r["per_tool"][t].get(q) for r in runs if t in r["per_tool"]])
                       for q in ("cold_ms", "warm_ms", "scheduling_s", "image_pull_s", "init_s", "readiness_s", "startup_s")}
                   for t in tools}
            table[domain] = {"runs": runs, "per_tool": per, "cold_ms": med([r["cold_ms"] for r in runs]),
                             "warm_ms": med([r["warm_ms"] for r in runs])}
            lines += [f"Median cold {table[domain]['cold_ms']} ms, warm {table[domain]['warm_ms']} ms "
                      f"({sum(len(r['per_tool']) for r in runs)} tool rows).", "",
                      "| tool | cold ms | warm ms | scheduling s | image pull s | init s | readiness s |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {t} | {per[t]['cold_ms']} | {per[t]['warm_ms']} | {per[t]['scheduling_s']} | "
                      f"{per[t]['image_pull_s']} | {per[t]['init_s']} | {per[t]['readiness_s']} |" for t in tools]
            fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
            x = range(len(tools))
            axes[0].bar([i - 0.2 for i in x], [per[t]["cold_ms"] or 0 for t in tools], 0.4, label="cold")
            axes[0].bar([i + 0.2 for i in x], [per[t]["warm_ms"] or 0 for t in tools], 0.4, label="warm")
            axes[0].set_title("call latency (ms, median)")
            axes[0].legend()
            bottom = [0.0] * len(tools)
            for part in ("scheduling_s", "image_pull_s", "init_s", "readiness_s"):
                vals = [per[t][part] or 0 for t in tools]
                axes[1].bar(x, vals, bottom=bottom, label=part.replace("_s", ""))
                bottom = [b + v for b, v in zip(bottom, vals)]
            axes[1].set_title("new pod start-up (s, median)")
            axes[1].legend(fontsize=8)
            for ax in axes:
                ax.set_xticks(list(x))
                ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=7)
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle(f"{a.exp} | {domain} | per-tool cold start (median over rounds)")
        fig.tight_layout()
        fig.savefig(out / f"{a.exp}_{domain}.png", dpi=120)
        plt.close(fig)
        lines.append("")

    md = "\n".join(lines) + "\n"
    (out / f"{a.exp}.md").write_text(md, encoding="utf-8")
    (out / f"{a.exp}.json").write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    print(md.encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
