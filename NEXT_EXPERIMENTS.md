# Next experiments: top-k prewarming, a decision model, and lookahead

Three ideas to try on top of the LightGBM (`gbt`) predictor. Each section has:
- the idea and why it might help
- an **offline check** first: minutes on the laptop, no cluster
- the **code changes**, with snippets ready to paste
- **cloud runs** that use [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md), Path A
- what to compare, and what counts as a win

## Where we stand (28 Oct 2026 cloud runs)

Held-out tasks 6 20 22 26 30, 46 episodes, 293 calls, fan-out on, fair hints, seed 252.

| | Cold calls | Mean latency | p95 / p99 | Pod-seconds |
|---|---|---|---|---|
| C0 | 87 | 515 ms | 1826 / 2029 ms | 9,080 |
| C3 gbt, θ = 0.3 | 37 | 241 ms | 1505 / 1844 ms | 8,923 |
| C3 LLM speculator | 60 | 326 ms | 1659 / 1709 ms | 9,833 |

Offline, on the held-out steps of `predictor/data/action_rows.csv`, gbt scores:
- top-1 = 0.637, top-3 = 0.875
- **hit@0.3 = 0.561**: the true tool was prewarmed on 56% of tool steps
- **wasted@0.3 = 0.18** prewarms per step

The gap between top-1 (0.64) and top-3 (0.88) is the opportunity that top-k goes after.
- **Main lever for p95/p99:** the cold share must fall below 5% (for p95) or 1% (for p99). Today it's 12.6%.
- **Second lever:** lead time. Under fair hints most prewarms only get the agent's think time (~1.5 s), which is about one cold start.

Run order: **1 → 2 → 3**. Experiment 1 is the smallest change, and 2 and 3 reuse its plumbing.

---

## Experiment 1: top-k prewarming

**Idea.** Prewarm the k most likely tools at each step instead of every tool with p ≥ θ.
RESPOND and END are still never prewarmed, and a small floor `p_min` avoids waking tools on steps that are almost surely RESPOND.
Variants:

| Policy | Rule |
|---|---|
| `theta` (today) | All tools with p ≥ θ |
| `topk` | The k most likely tools, each only if p ≥ p_min |
| `mass` | The fewest tools whose probabilities add up to ≥ `mass` × (total tool probability), each with p ≥ p_min |

**Hypothesis.** top-2 raises the hit rate from about 0.56 toward 0.75 for roughly double the prewarms.
A wasted prewarm is cheap when the tool is already warm (the ping just restarts its 60 s window).
When the tool is cold, a wasted prewarm costs about 60–65 pod-seconds.

### 1.1 Code: the policy function (`predictor/models.py`, next to `prewarm_set`)

```python
def prewarm_select(ranked, policy="theta", theta=0.3, top_k=2, p_min=0.0, mass=0.8):
    """Which tools to prewarm, from [(action, p), ...] highest first. RESPOND / END never.
      theta  every tool with p >= theta (= prewarm_set)
      topk   the top_k tools, each with p >= p_min
      mass   the fewest tools covering `mass` of the tools' total probability, each with p >= p_min"""
    tools = [(t, p) for t, p in ranked if t not in NO_PREWARM]
    if policy == "theta":
        return [t for t, p in tools if p >= theta]
    if policy == "topk":
        return [t for t, p in tools[:top_k] if p >= p_min]
    if policy == "mass":
        total, out, acc = sum(p for _, p in tools), [], 0.0
        for t, p in tools:
            if p < p_min or (total and acc >= mass * total):
                break
            out.append(t)
            acc += p
        return out
    raise ValueError(f"unknown PREWARM_POLICY {policy}")
```

### 1.2 Offline check (~1 min, laptop)

Save as `predictor/eval_policies.py` and run `PYTHONIOENCODING=utf-8 "$PY" predictor/eval_policies.py` from the repo root:

```python
"""Offline: hit rate vs prewarms per step for each prewarm policy, gbt on the held-out airline steps."""
import pickle
import sys

import pandas as pd

sys.path.insert(0, "predictor")
from models import prewarm_select          # noqa: E402
from train_action import NO_WARM, predict_rows  # noqa: E402

pred = pickle.load(open("predictor/data/action_model.pkl", "rb"))
rows = pd.read_csv("predictor/data/action_rows.csv")
test = rows[rows.split == "test"]
probs, y = predict_rows(pred, "gbt", test), test.target.tolist()

POLICIES = [("theta", dict(theta=0.5)), ("theta", dict(theta=0.3)), ("theta", dict(theta=0.2)),
            ("topk", dict(top_k=1)), ("topk", dict(top_k=2)), ("topk", dict(top_k=3)),
            ("topk", dict(top_k=2, p_min=0.1)), ("topk", dict(top_k=3, p_min=0.05)),
            ("mass", dict(mass=0.8, p_min=0.05))]
print("policy                          hit   prewarms/step  wasted/step")
for name, kw in POLICIES:
    hit = tool_n = warmed = wasted = 0
    for p, t in zip(probs, y):
        ranked = sorted(p.items(), key=lambda kv: -kv[1])
        w = prewarm_select(ranked, name, **kw)
        warmed += len(w)
        wasted += sum(a != t for a in w)
        if t not in NO_WARM:
            tool_n += 1
            hit += t in w
    print(f"{name:6} {str(kw):24} {hit / tool_n:.3f}   {warmed / len(y):.2f}           {wasted / len(y):.2f}")
```

Check: the `theta 0.3` line must reproduce hit 0.561 and wasted 0.18. If it doesn't, the script has a bug.

Output on 29 Oct 2026 (current `action_model.pkl`, 1,417 held-out steps):

| Policy | Hit (tool steps prewarmed) | Prewarms / step | Wasted / step |
|---|---|---|---|
| theta 0.5 | 0.409 | 0.29 | 0.07 |
| **theta 0.3 (today)** | **0.561** | **0.49** | **0.18** |
| theta 0.2 | 0.615 | 0.64 | 0.31 |
| topk 1 | 0.685 | 1.00 | 0.63 |
| topk 2 | 0.772 | 2.00 | 1.58 |
| topk 3 | 0.817 | 3.00 | 2.55 |
| topk 2, p_min 0.1 | 0.716 | 1.07 | 0.67 |
| topk 3, p_min 0.05 | 0.774 | 1.63 | 1.20 |
| mass 0.8, p_min 0.05 | 0.783 | 1.62 | 1.19 |

- **Plain top-k without a floor** prewarms on every RESPOND step too, so it wastes a lot.
- **The floors keep most of the gain:** `topk 2, p_min 0.1` gets +15 points of hit for about 2× the prewarms of θ = 0.3.
- **A wasted prewarm on an already-warm tool costs ~nothing.** Pod-seconds grow much less than "wasted/step" suggests, which is why the cloud run is the real test.
**Choose 2–3 policies to run in the cloud:** the ones that clearly raise the hit rate without more than about doubling prewarms per step. Good first picks are `topk 2 p_min 0.1` and `topk 3 p_min 0.05`.

### 1.3 Code: use it in the API (`predictor/service.py`)

```python
from models import ACTION_MODELS, MODEL_NAMES, POOLED, prewarm_select   # instead of prewarm_set

PREWARM_POLICY = os.getenv("PREWARM_POLICY", "theta")     # theta | topk | mass
TOP_K = int(os.getenv("TOP_K", "2"))
P_MIN = float(os.getenv("P_MIN", "0.0"))
MASS = float(os.getenv("MASS", "0.8"))

# in predict(): replace   warm = set(prewarm_set(ranked, THETA))   with
    warm = set(prewarm_select(ranked, PREWARM_POLICY, theta=THETA, top_k=TOP_K, p_min=P_MIN, mass=MASS))
```

Also add these settings to the docstring at the top of `service.py`, and to the `/healthz` answer so every run records its policy.
Find where `/healthz` builds its dict and add `"policy": {"name": PREWARM_POLICY, "top_k": TOP_K, "p_min": P_MIN, "mass": MASS}`.
`run_cloud.sh` already saves `/healthz` as `predictor_health.json` in each run folder.

Unit test (`predictor/tests/test_predictor.py`): `prewarm_select([("RESPOND", .6), ("a", .2), ("b", .1), ("c", .05)], "topk", top_k=2) == ["a", "b"]`, and with `p_min=0.15` the result is `["a"]`.
Run it with `"$PY" -m pytest predictor/tests -q`.

### 1.4 Cloud runs

```bash
bash EKS_deployment/cloud/build_image.sh           # new code into the image
PYTHON="$PY" bash EKS_deployment/eks_up.sh
bash EKS_deployment/cloud/cloud_up.sh
CONFIG=C0 bash EKS_deployment/cloud/run_cloud.sh --fanout                      # fresh baseline, same day

CONFIG=C3 PREDICTOR=gbt LABEL=cloud-theta03 bash EKS_deployment/cloud/run_cloud.sh --fanout   # today's policy

kubectl set env deploy/pulse-predictor PREWARM_POLICY=topk TOP_K=2 P_MIN=0.1
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-top2 bash EKS_deployment/cloud/run_cloud.sh --fanout

kubectl set env deploy/pulse-predictor PREWARM_POLICY=topk TOP_K=3 P_MIN=0.05
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-top3 bash EKS_deployment/cloud/run_cloud.sh --fanout

kubectl set env deploy/pulse-predictor PREWARM_POLICY=theta                     # back to normal
```

- `kubectl set env` restarts the predictor with the new settings, and `run_cloud.sh` keeps them; it only restarts the pod.
- Check before each run with `kubectl set env deploy/pulse-predictor --list`, and afterwards in the run's `predictor_health.json`.
- Each run takes ~30 min. Folder names end in `_cloud-top2-gbt` and so on.

### 1.5 What to compare

```bash
/c/Python311/python.exe test/test_knative/plot_compare.py results/cloud_runs/<C0> results/cloud_runs/<theta03> \
  results/cloud_runs/<top2> results/cloud_runs/<top3> --out results/cloud_runs/compare_<date>_topk_cloud
```

| Metric | Where |
|---|---|
| Cold calls, mean, p95, p99 | `comparison.json` |
| Pod-seconds (cost) | `comparison.json`; vs C0 and vs C1 (NFR: ≥30% fewer than C1) |
| Prewarms sent, how many were wasted | `controller.jsonl`: count `warmed` events and `was_cold: true` |

**Win:** fewer cold calls than θ = 0.3, with pod-seconds still well under C1's.
Plot cold calls (x) against pod-seconds (y), one point per policy. That trade-off curve is the result.

---

## Experiment 2: a decision model (cost-aware prewarming)

**Idea.** θ = 0.3 is one fixed number for every tool and every moment. A decision model asks, for each tool:
*is the latency I expect to save worth the pod-seconds I risk?*

```
saved_s(t)  = min(lead_s, cold_s(t))           latency a prewarm can hide (lead = time until the call)
benefit(t)  = p(t) × saved_s(t)                 expected seconds of latency saved
cost(t)     = (1 − p(t)) × (window_s + cold_s(t))   expected pod-seconds wasted if the guess is wrong
prewarm  if  tool has fewer pods than k   and   benefit(t) ≥ λ × cost(t)
```

- **λ** is the price of one pod-second, in seconds of latency.
  - λ = 0.01 with lead 1.5 s, cold 1.5 s and window 60 s gives p ≥ 0.29, which is about today's θ = 0.3.
  - Smaller λ prewarms more; larger λ prewarms less.
- **What it adds over θ:**
  1. **Skip tools that are already warm.** No benefit, and no point in a ping. This saves CRD writes and pings, not pods.
  2. **Lead time and per-tool cold starts together.** A prewarm only hides `min(lead, cold)`.
     - With today's ~1.5 s lead, every tool saves at most 1.5 s, so the rule is close to one θ.
     - With a long lead (experiment 3), slow tools save their whole cold start. `book_reservation` takes 2.5 s cold vs 1.6 s for most, so it's worth prewarming at lower p (λ = 0.01: p ≥ 0.20 instead of 0.28).
  3. **Many guesses at once.** It weighs each tool on its own, so it can prewarm several cheap, likely-enough tools (like top-k) while skipping unlikely ones (like θ).

There are two levels. Do 2a first.

### 2a. Analytical decision rule (no training)

**Inputs the predictor needs:**
- `cold_s(t)`: from `deploy/knative/results/2026-10-28_0217_C0_tools-cold-warm.csv` (columns `tool,cold_ms,...`), which is EKS. Copy it into `predictor/data/tool_cold_ms.csv` so it gets into the image.
- `pods(t)`: the tool's current pod count. Add a pod cache to the predictor that lists pods every 2 s.
  The pod label is `serving.knative.dev/service`; the service name is the tool name with `_` → `-`.
  The `pulse` service account may already list pods (see `EKS_deployment/cloud/pulse-cloud.yaml`).
- `lead_s`: a setting (`LEAD_S`, default 1.5 = think time). Experiment 3 makes it larger for lookahead predictions.

**Code sketch** (`predictor/decision.py`, new):

```python
"""Cost-aware prewarm decision (NEXT_EXPERIMENTS.md, experiment 2a)."""
import csv
import os
import threading
import time
from pathlib import Path

from models import NO_PREWARM

HERE = Path(__file__).resolve().parent
LAMBDA = float(os.getenv("LAMBDA", "0.01"))       # seconds of latency one pod-second is worth
LEAD_S = float(os.getenv("LEAD_S", "1.5"))
WINDOW_S = float(os.getenv("TTL_S", "60"))
COLD_S = {r["tool"]: float(r["cold_ms"]) / 1000 for r in csv.DictReader(open(HERE / "data" / "tool_cold_ms.csv"))}


class PodCache:
    """Pods per Knative service, refreshed every `every` s in the background (needs pods list RBAC)."""

    def __init__(self, every=2.0):
        from kubernetes import client, config
        config.load_incluster_config() if os.getenv("KUBERNETES_SERVICE_HOST") else config.load_kube_config()
        self.core, self.counts = client.CoreV1Api(), {}
        threading.Thread(target=self._loop, args=(every,), daemon=True).start()

    def _loop(self, every):
        while True:
            try:
                n = {}
                for p in self.core.list_namespaced_pod("default", label_selector="serving.knative.dev/service").items:
                    if p.metadata.deletion_timestamp is None and p.status.phase in ("Running", "Pending"):
                        s = p.metadata.labels["serving.knative.dev/service"]
                        n[s] = n.get(s, 0) + 1
                self.counts = n
            except Exception as e:          # keep the last counts
                print(f"pod cache: {type(e).__name__}: {e}")
            time.sleep(every)

    def pods(self, tool):
        return self.counts.get(tool.replace("_", "-"), 0)


def decide(ranked, k_of, pods, lam=LAMBDA, lead_s=LEAD_S, window_s=WINDOW_S):
    """Tools worth prewarming: [(tool, p, why), ...]. k_of(tool) = fan-out width, pods(tool) = current pods."""
    out = []
    for t, p in ranked:
        if t in NO_PREWARM or pods(t) >= k_of(t):
            continue                                    # nothing to gain: already warm enough
        cold = COLD_S.get(t, 1.5)
        benefit = p * min(lead_s, cold)
        cost = (1 - p) * (window_s + cold)
        if benefit >= lam * cost:
            out.append((t, p, round(benefit - lam * cost, 4)))
    return out
```

**In `service.py`:**
- Add `PREWARM_POLICY=decision`.
- Create `PODS = PodCache()` only when that policy is on.
- In `predict()`, call:
  ```python
  warm = {t for t, _, _ in decide(ranked, lambda t: PRED.action.fanout.k(t, h.signals) if action_used else 1, PODS.pods)}
  ```
- Log λ and the decisions in the `AUDIT_LOG` line.

**Offline check (laptop).** Without live pod state, assume every tool is cold.
`decide` then becomes a per-tool θ; run it through `eval_policies.py` for λ ∈ {0.005, 0.01, 0.02}.
It should land between θ = 0.2 and θ = 0.5 on hit/wasted. The real gain (skipping warm tools) only shows in the cluster.

**Cloud runs:**

```bash
kubectl set env deploy/pulse-predictor PREWARM_POLICY=decision LAMBDA=0.01
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-dec001 bash EKS_deployment/cloud/run_cloud.sh --fanout
kubectl set env deploy/pulse-predictor LAMBDA=0.005
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-dec0005 bash EKS_deployment/cloud/run_cloud.sh --fanout
```

**Win:** compared with θ = 0.3 (and top-k), either the same cold calls with fewer pod-seconds or controller pings, or fewer cold calls for the same pod-seconds.
Count `controller.jsonl` `warmed` events with `was_cold: false`. Those are pings the decision model should have skipped, and this number should drop a lot.

### 2b. Learned decision model (optional, if 2a looks promising)

Train a small classifier that predicts **"prewarming tool t now would hide a cold start"** from more context than p alone.

1. **Collect data in shadow mode.** Run C0 with the predictor answering hints but the controller off, so nothing is prewarmed:
   ```bash
   kubectl scale deploy/pulse-controller --replicas=0
   CONFIG=C0 LABEL=cloud-shadow bash EKS_deployment/cloud/run_cloud.sh --fanout --predictor-url http://pulse-predictor:8000
   kubectl scale deploy/pulse-controller --replicas=1
   ```
   The folder will be labelled C3 (a predictor URL was given), but no prewarming happened, so the tool behaviour is C0.
   `hints.jsonl` holds every prediction; `calls.csv` and `pods.csv` show what really happened. Run it with 2–3 seeds (`--seed 1`, `--seed 2`) for more data.
2. **Build rows**, one per (hint, candidate tool in the top 5):
   - **Features:** p(t), rank of t, p(RESPOND), tool id, pods(t) at the hint (from `pods.csv`), seconds since t was last called, concurrent episodes, the hint's signals.
   - **Label = 1** if the same episode called t within `window_s` after the hint *and* t had no pod at that moment. Otherwise 0.
3. **Train** LightGBM binary (or logistic regression for interpretability), split by task as in `train_action.py`.
   Pick the decision threshold from λ: prewarm if `P(useful) × cold_s ≥ λ × (1 − P(useful)) × window_s`.
4. **Serve** it like 2a, with `PREWARM_POLICY=learned`, and run the same cloud comparison.

Watch for leakage: build features only from what was known at the hint's `ts`.

---

## Experiment 3: lookahead on reply steps (more lead time)

**Idea.** Under fair hints, the hint before a tool call arrives only one think time (~1.5 s) before the call, about one cold start.
At a **reply step** (the agent is about to answer the customer), the next *tool* call usually comes after the customer's reply (~8 s) plus think time.
A model that predicts **the next tool call, skipping RESPOND steps**, can prewarm with ~9 s of lead, enough to finish a cold start.
That is where the remaining cold calls (and p95/p99) can come down.

1. **New target.** In `predictor/data/action_rows.csv`, rows are in step order within each `trace` (checked: 600 traces, 7,128 rows).
   Add a column `next_tool` = the next row's target in the same trace that is not RESPOND (END if none):
   ```python
   rows = pd.read_csv("predictor/data/action_rows.csv")
   nxt = []
   for _, g in rows.groupby("trace", sort=False):
       tg = g.target.tolist()
       nxt += [next((x for x in tg[i + 1:] if x != "RESPOND"), "END") for i in range(len(tg))]
   rows["next_tool"] = nxt     # keep the groupby order aligned with rows (sort rows by trace, step first)
   ```
2. **Train** a second LightGBM on rows whose `target == "RESPOND"`, with `next_tool` as the label and the same features.
   Reuse `fit_all`/`GBTModel` in `train_action.py`, for example with a `--target next_tool` option. Save it inside `action_model.pkl` as `models["gbt_next"]`.
   Offline: top-1/top-3 and hit@θ on held-out RESPOND rows.
3. **Serve.** When gbt's top action is RESPOND (or p(RESPOND) ≥ 0.5), also rank `gbt_next`.
   Prewarm its tools with p ≥ `THETA_NEXT` (start with 0.4), or through the decision rule with `lead_s ≈ 9`.
   With that lead each correct guess hides the whole cold start. Add the lead itself to the cost: the pod idles until the call, even when the guess is right.
4. **TTL check.** A lookahead prewarm must still be warm when the call arrives:
   - the customer reply is 1–60 s, median 8 s
   - Knative's window is 60 s, so this is fine for nearly every reply
   - a reply over ~55 s will miss
5. **Cloud runs.** The same as experiment 1, with `LABEL=cloud-lookahead`. Compare against the best policy from experiments 1 and 2.

**Win:** fewer cold calls on the *first* call after a customer turn. In `calls.csv`, filter to `user_turns_before > 0`; those are the calls lookahead targets.
The tail should also move: p95 drops a lot once the cold share falls below 5%.

---

## Record every run

Copy this table into your notes. Use the same seed, the same workload and the same day's C0 for every row.

| Run folder | Policy / settings | Cold calls | Mean | p95 | p99 | Pod-seconds | Pings (wasted) | Notes |
|---|---|---|---|---|---|---|---|---|
| … `_cloud` (C0) | – | | | | | | – | |
| … `_cloud-theta03-gbt` | theta 0.3 | | | | | | | |
| … `_cloud-top2-gbt` | topk 2, p_min 0.1 | | | | | | | |

Protocol for the final numbers: 3 trials per config (`LABEL=…-t1/-t2/-t3`), report the median with min/max.
Always finish with `bash EKS_deployment/eks_down.sh` and the checks in RUN_EXPERIMENTS.md A8.
