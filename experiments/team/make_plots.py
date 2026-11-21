"""Plots for the four team scripts + the crowding extra: python experiments/team/make_plots.py (from the repo root) -> experiments/results/plots/."""
import csv, json, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("experiments/results")
EXTRA = Path("results/team/extra_adaptive_crowded/adaptive_crowded/adaptive_crowded.json")
OUT = R / "plots"
OUT.mkdir(exist_ok=True)

# reference palette (validated: light surface), fixed order, entity -> colour never changes
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CFG = {"C0": BLUE, "C3-markov": ORANGE, "C3-lightgbm_adapt": AQUA, "C1": YELLOW}
CFG_LABEL = {"C0": "C0 scale-to-zero", "C3-markov": "C3 Pulse (markov)",
             "C3-lightgbm_adapt": "C3 Pulse (lightgbm_adapt)", "C1": "C1 min-scale 1"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "legend.frameon": False, "figure.dpi": 110,
})


def load(p):
    return json.load(open(p, encoding="utf-8"))


def runs(d, key, m):
    return [r[m] for r in d[key]["runs"] if r.get(m) is not None]


def med(xs):
    return st.median(xs) if xs else float("nan")


def bar_with_trials(ax, labels, values_per, color, fmt="{:.0f}", ylabel=None, colors=None):
    """Median bar + every trial as a dot (n=3 is too small to hide)."""
    x = range(len(labels))
    for i, vals in enumerate(values_per):
        c = colors[i] if colors else color
        m = med(vals)
        ax.bar(i, m, width=0.6, color=c, edgecolor=SURF, linewidth=2, zorder=2)
        ax.scatter([i] * len(vals), vals, s=18, color=INK, zorder=3, linewidths=0)
        ax.annotate(fmt.format(m), (i, max(vals + [m])), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(list(x), labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)


def save(fig, name, note=None, note_y=-0.03):
    if note:
        fig.text(0.01, note_y, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(OUT / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


N3 = "Bars: median of 3 trials; dots: each trial."

# ---------------------------------------------------------------- script 1
s1 = load(R / "script1_latency_resources/latency_resources/latency_resources.json")
cfgs = ["C0", "C3-markov", "C3-lightgbm_adapt", "C1"]

# 1a. trade-off: pod-seconds vs cold calls
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, dom in zip(axs, ["airline", "retail"]):
    for c in cfgs:
        k = f"{dom}/{c}"
        xs, ys = runs(s1, k, "pod_s"), runs(s1, k, "cold_calls")
        ax.scatter(xs, ys, s=60, color=CFG[c], edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(CFG_LABEL[c], (med(xs), med(ys)), xytext=(8, 6), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    ax.set_title(f"{dom}: cost vs cold calls (46 conversations)")
    ax.set_xlabel("pod-seconds during run (cost)")
    ax.set_ylabel("cold calls")
    ax.set_ylim(bottom=-5)
    ax.margins(x=0.2)
save(fig, "s1_cost_vs_cold.png", "Script 1. Each dot is one trial. Lower-left is better.")

# 1b. latency percentiles
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
marks = [("p50_ms", "o", "p50"), ("p95_ms", "s", "p95"), ("p99_ms", "^", "p99")]
for ax, dom in zip(axs, ["airline", "retail"]):
    for i, c in enumerate(cfgs):
        for m, mk, _ in marks:
            v = med(runs(s1, f"{dom}/{c}", m))
            ax.scatter(v, i, marker=mk, s=70, color=CFG[c], edgecolor=SURF, linewidth=1.5, zorder=3)
        lo, hi = med(runs(s1, f"{dom}/{c}", "p50_ms")), med(runs(s1, f"{dom}/{c}", "p99_ms"))
        ax.plot([lo, hi], [i, i], color=CFG[c], lw=2, zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(cfgs)), [CFG_LABEL[c] for c in cfgs])
    ax.invert_yaxis()
    ax.set_title(f"{dom}: tool-call latency")
    ax.set_xlabel("ms (log scale)")
    ax.grid(axis="y", visible=False)
axs[1].legend([Line2D([], [], marker=mk, ls="", color=MUTED, ms=7) for _, mk, _ in marks],
              [l for *_, l in marks], loc="upper left", bbox_to_anchor=(1.0, 1.0), title="median of 3 trials")
save(fig, "s1_latency.png", "Script 1. ~1.5 s cold start vs ~50 ms warm: p95/p99 depend on whether >5% / >1% of calls are cold.")

# 1c. CPU: tools vs Pulse itself
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, dom in zip(axs, ["airline", "retail"]):
    tool = [med(runs(s1, f"{dom}/{c}", "tool_cpu_core_s")) for c in cfgs]
    pulse = [med(runs(s1, f"{dom}/{c}", "pulse_cpu_core_s")) for c in cfgs]
    y = range(len(cfgs))
    ax.barh(y, tool, color=BLUE, height=0.6, edgecolor=SURF, linewidth=2, label="tool pods")
    ax.barh(y, pulse, left=tool, color=ORANGE, height=0.6, edgecolor=SURF, linewidth=2, label="Pulse (predictor + controller)")
    for i, (t, p) in enumerate(zip(tool, pulse)):
        ax.annotate(f"{t:.0f} + {p:.0f}", (t + p, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), [CFG_LABEL[c] for c in cfgs])
    ax.invert_yaxis()
    ax.set_title(f"{dom}: CPU used during the run")
    ax.set_xlabel("CPU core-seconds")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.25)
axs[0].legend(loc="lower right")
save(fig, "s1_cpu.png", "Script 1, from Prometheus. Median of 3 trials.")

# ---------------------------------------------------------------- script 2: top-k
s2 = load(R / "script2_topk_ttl/topk_adaptive/topk_adaptive.json")
arms = ["top1", "top2", "top3", "adaptive", "adaptive-kadapt"]
lab = ["top-1", "top-2", "top-3", "adaptive", "adaptive\n+ k-adapt"]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (m, t, f) in zip(axs, [("cold_calls", "cold calls", "{:.0f}"), ("p95_ms", "p95 latency (ms)", "{:.0f}"),
                               ("pod_s", "pod-seconds", "{:.0f}"), ("prewarms", "prewarms issued", "{:.0f}")]):
    bar_with_trials(ax, lab, [runs(s2, f"airline/{a}", m) for a in arms], AQUA, f)
    ax.set_title(t)
    if m == "pod_s":
        ax.set_ylim(9000, 11000)
        ax.set_title("pod-seconds (axis from 9000)")
save(fig, "s2_topk.png", "Script 2, airline, C3 lightgbm_adapt. " + N3)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
bar_with_trials(ax, lab, [runs(s2, f"airline/{a}", "wasted_pct") for a in arms], AQUA, "{:.0f}%")
ax.set_title("Prewarms never used by a call (wasted)")
ax.set_ylabel("% of prewarms")
save(fig, "s2_wasted.png", "Script 2. " + N3)

# ---------------------------------------------------------------- script 2: TTL
s2t = load(R / "script2_topk_ttl/ttl/ttl.json")
ttls = [30, 60, 120]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (m, t) in zip(axs, [("cold_calls", "cold calls"), ("p95_ms", "p95 latency (ms)"), ("pod_s", "pod-seconds")]):
    ys = [runs(s2t, f"airline/ttl{x}", m) for x in ttls]
    ax.plot(ttls, [med(y) for y in ys], color=MAGENTA, lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3,
            label="TTL arms (Knative window 20 s)")
    for x, y in zip(ttls, ys):
        ax.scatter([x] * len(y), y, s=14, color=INK, zorder=4)
    ref = med(runs(s2, "airline/adaptive", m))
    ax.axhline(ref, color=AQUA, lw=2, ls="--", zorder=2, label="default adaptive (window 60 s, no TTL)")
    ax.set_xticks(ttls, [f"{x} s" for x in ttls])
    ax.set_xlabel("prewarm TTL")
    ax.set_title(t)
axs[0].legend(loc="upper right")
save(fig, "s2_ttl.png", "Script 2, airline. Line: median of 3 trials; dots: trials.")

# ---------------------------------------------------------------- script 3: concurrency
s3 = load(R / "script3_kfanout_concurrency/concurrency/concurrency.json")
rates = [3, 7, 14]
fig, axs = plt.subplots(1, 4, figsize=(16, 3.7))
for ax, (m, t) in zip(axs, [("p95_ms", "p95 latency (ms)"), ("cold_pct", "cold calls (% of calls)"),
                            ("transport_failures", "failed calls (no answer)"), ("pod_s", "pod-seconds")]):
    for c, col, l in [("C0", BLUE, "C0"), ("C3-lightgbm_adapt", AQUA, "C3 Pulse")]:
        ys = [runs(s3, f"airline/{c}-rate{r}", m) for r in rates]
        ax.plot(rates, [med(y) for y in ys], color=col, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=l, zorder=3)
        for r, y in zip(rates, ys):
            ax.scatter([r] * len(y), y, s=12, color=INK, zorder=4)
    ax.set_xticks(rates)
    ax.set_xlabel("new conversations per minute")
    ax.set_title(t)
    ax.set_ylim(bottom=0)
axs[0].legend()
save(fig, "s3_concurrency.png", "Script 3, airline. ~4 / 7 / 12 conversations at once. Line: median; dots: trials.")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
w = 0.38
for i, r in enumerate(rates):
    t = med(runs(s3, f"airline/C3-lightgbm_adapt-rate{r}", "tool_cpu_core_s"))
    p = med(runs(s3, f"airline/C3-lightgbm_adapt-rate{r}", "pulse_cpu_core_s"))
    ax.bar(i - w / 2, t, w, color=BLUE, edgecolor=SURF, linewidth=2, label="tool pods" if i == 0 else None)
    ax.bar(i + w / 2, p, w, color=ORANGE, edgecolor=SURF, linewidth=2, label="Pulse (predictor + controller)" if i == 0 else None)
    for x, v in [(i - w / 2, t), (i + w / 2, p)]:
        ax.annotate(f"{v:.0f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(3), [f"{r}/min" for r in rates])
ax.set_ylabel("CPU core-seconds")
ax.set_title("C3: Pulse's own CPU overtakes the tools' CPU")
ax.grid(axis="x", visible=False)
ax.legend()
save(fig, "s3_pulse_overhead.png", "Script 3, airline, C3 lightgbm_adapt, from Prometheus. Median of 3 trials.")

# k fan-out
s3k = load(R / "script3_kfanout_concurrency/k_fanout/k_fanout.json")
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, (m, t) in zip(axs, [("cold_calls", "cold calls"), ("p95_ms", "p95 latency (ms)"), ("pod_s", "pod-seconds")]):
    bar_with_trials(ax, ["k rule", "k static"], [runs(s3k, f"airline/k-{a}", m) for a in ["rule", "static"]], AQUA)
    ax.set_title(t)
save(fig, "s3_kfanout.png", "Script 3, airline, C3 lightgbm_adapt. " + N3)

# ---------------------------------------------------------------- script 4: cold start
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
fig.subplots_adjust(wspace=0.8)
for ax, dom in zip(axs, ["airline", "retail"]):
    man = [l.rstrip("\n").split("\t") for l in open(R / "script4_coldstart_api/manifest.tsv", encoding="utf-8")]
    rd = next(R / "script4_coldstart_api" / m[5] for m in man if m[0] == "coldstart" and m[1] == dom and m[4] == "done")
    rows = list(csv.DictReader(open(rd / "coldstart.csv", encoding="utf-8")))
    tools = sorted({r["tool"] for r in rows}, key=lambda t: med([float(r["cold_ms"]) for r in rows if r["tool"] == t]))
    for i, t in enumerate(tools):
        c = [float(r["cold_ms"]) for r in rows if r["tool"] == t and r["cold_ms"]]
        wv = [float(r["warm_ms"]) for r in rows if r["tool"] == t and r["warm_ms"]]
        ax.scatter(c, [i] * len(c), s=22, color=BLUE, alpha=0.85, linewidths=0, zorder=3)
        ax.scatter(wv, [i] * len(wv), s=22, color=AQUA, alpha=0.85, linewidths=0, zorder=3)
    ax.set_yticks(range(len(tools)), [t.replace("retail-", "") for t in tools])
    ax.set_xscale("log")
    ax.set_xlabel("ms (log scale)")
    ax.set_title(f"{dom}: first call to a tool at zero vs warm")
    ax.grid(axis="y", visible=False)
axs[1].legend([Line2D([], [], marker="o", ls="", color=BLUE), Line2D([], [], marker="o", ls="", color=AQUA)],
              ["cold (pod at zero)", "warm"], loc="upper left", bbox_to_anchor=(1.0, 1.0))
save(fig, "s4_coldstart.png", "Script 4. 5 rounds per tool; each dot is one round.")

pa = load(R / "script4_coldstart_api/predictor_api/predictor_api.json")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, dom in zip(axs, ["airline", "retail"]):
    for model, col in [("markov", ORANGE), ("lightgbm_adapt", AQUA)]:
        rr = pa[f"{dom}/{model}"]["rates"]
        xs = sorted(rr, key=float)
        for m, ls in [("p50", "-"), ("p99", ":")]:
            ax.plot([float(x) for x in xs], [rr[x][m] for x in xs], color=col, ls=ls, lw=2, marker="o", ms=6,
                    mec=SURF, label=f"{model} {m}")
    ax.set_xlabel("requests per second")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_title(f"{dom}: predictor /predict latency")
    ax.set_ylim(0, 20)
axs[0].set_ylabel("ms")
axs[0].legend(ncol=2)
save(fig, "s4_predictor_api.png", "Script 4. Median of 3 trials. Next-tool prediction costs ~8 ms vs ~1500 ms cold start.")

# ---------------------------------------------------------------- extra: crowding
if EXTRA.exists():
    ex = load(EXTRA)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (m, t) in zip(axs, [("cold_calls", "cold calls"), ("p95_ms", "p95 latency (ms)"),
                                ("pod_s", "pod-seconds"), ("prewarms", "prewarms issued")]):
        bar_with_trials(ax, ["adaptive", "top-3"], [runs(ex, f"airline/{a}-rate14", m) for a in ["adaptive", "top3"]], AQUA)
        ax.set_title(t)
    save(fig, "extra_adaptive_vs_top3_rate14.png",
         "Extra run, airline, 14 conversations/min. " + N3 + " Adaptive trial 1 is the outlier dot.")

# ---------------------------------------------------------------- consistency across clusters
same = [("Script 1\nSankalp's\ncluster", s1, "airline/C3-lightgbm_adapt"),
        ("Script 2\nAniruth's\ncluster", s2, "airline/adaptive"),
        ("Script 3\nNithish's\ncluster", s3k, "airline/k-rule")]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (m, t) in zip(axs, [("cold_calls", "cold calls"), ("p95_ms", "p95 latency (ms)"), ("pod_s", "pod-seconds")]):
    bar_with_trials(ax, [s[0] for s in same], [runs(d, k, m) for _, d, k in same], AQUA)
    ax.set_title(t)
    if m == "pod_s":
        ax.set_ylim(9000, 11000)
        ax.set_title("pod-seconds (axis from 9000)")
save(fig, "consistency_same_config.png",
     "Identical configuration (C3, lightgbm_adapt, adaptive top-k, 46 airline conversations) on three EKS clusters. " + N3,
     note_y=-0.1)
