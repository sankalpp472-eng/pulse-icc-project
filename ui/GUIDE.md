# Pulse UI: the complete guide

This guide covers everything you can do in the Pulse demo UI: what to run, how to run it, how to read what you see, and where every result ends up.
For a short technical overview see [README.md](README.md); for running experiments without the UI see [RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md).

**Contents**
1. [What the UI is for](#1-what-the-ui-is-for)
2. [Pick a setup](#2-pick-a-setup)
3. [Start the UI](#3-start-the-ui)
4. [A tour of the screen](#4-a-tour-of-the-screen)
5. [Recipes: what to run](#5-recipes-what-to-run)
6. [Reading what you see](#6-reading-what-you-see)
7. [Where the results are](#7-where-the-results-are)
8. [Stop, clean up, costs](#8-stop-clean-up-costs)
9. [Troubleshooting](#9-troubleshooting)
10. [Quick reference](#10-quick-reference)

---

## 1. What the UI is for

Pulse predicts the next MCP tool an LLM agent will call, and wakes that tool's scaled-to-zero pod before the call arrives.
The UI makes this visible and lets you measure it:

| You want to… | Use |
|---|---|
| **Show** the idea to someone: watch the predictor guess, pods start early, calls come back warm | **Live demo** tab |
| **Talk to the agent yourself** as the customer | Live demo → *Live agent* → *I'll type* |
| **Measure** C0 vs C1 vs C3 properly (seeded replays, live runs) | **Experiments** tab |
| **Compare** runs, make the report plots, download result folders | **Results** tab |
| **Explain** the system in one page | **How it works** tab |

Words used throughout:

| Word | Meaning |
|---|---|
| **Cold start** | A tool had no pod, so the call waited ~1.5–2.5 s while Knative started one |
| **Prewarm** | Starting a tool's pod *before* the agent calls it, because the predictor expects that call |
| **Hint** | What the agent sends the predictor before every LLM step: the tools called so far plus a few numbers about the conversation (never raw text) |
| **θ (theta)** | The confidence threshold: tools with predicted probability ≥ θ are prewarmed |
| **k** | How many pods to prewarm for one tool, for k parallel lookups (fan-out) |
| **RESPOND** | The agent answers the customer instead of calling a tool. It is never prewarmed |
| **C0 / C1 / C3** | No prewarming / every tool always warm / Pulse's prewarming |

---

## 2. Pick a setup

| Setup | Tools run on | Needs | Cost | Use it for |
|---|---|---|---|---|
| **A. Laptop + simulated cluster** | A simulation on the laptop | Only `agent/.venv` | Free | Demos anywhere, trying predictors, talking to the agent |
| **B. UI inside EKS** (recommended for real numbers) | The real Knative cluster on AWS | AWS login, Docker for the image build | EKS + EC2 by the hour, until teardown | Real cold starts, experiment runs, real demos |
| C. Laptop + kind | A local Knative cluster | Docker with plenty of memory | Free | Not recommended on this laptop: it cannot run kind reliably |

How the simulated cluster (A) works:
- **Cold starts** use the per-tool times measured on EKS.
- **Pods** follow the same rules as the real tools: one request per pod, up to 10 pods, back to zero after 60 s idle.
- **Tool results are real:** the tools run on the airline database.

It is the same system, just with simulated pods.

The live agent and the AI customer use Groq. The keys come from `agent/.env` (laptop) or the `pulse-llm-keys` Secret (EKS). Replays never call an LLM.

---

## 3. Start the UI

### Setup A: laptop, simulated cluster (1 minute)

```bash
cd ~/OneDrive/Documents/iisc_ICC/Pulse-project1
agent/.venv/Scripts/python ui/server.py
```

1. Open **http://127.0.0.1:8765**.
2. Wait until the **Predictor** card (right side) shows **ready**. The first load takes about a minute.
3. The top bar shows **Tools run on: Simulated cluster**.

To stop the UI, press Ctrl+C in that terminal. That also stops the predictor it started.

### Setup B: UI inside EKS (~40 min the first time)

Run these in Git Bash, from the repo root. Each step is explained in RUN_EXPERIMENTS.md (A1–A3).

```bash
export PY="$PWD/agent/.venv/Scripts/python"
bash EKS_deployment/cloud/build_image.sh             # 1. image with the UI (Docker running) ~5 min
PYTHON="$PY" bash EKS_deployment/eks_up.sh           # 2. the cluster + Knative + 14 tools   ~25 min
bash EKS_deployment/cloud/cloud_up.sh                # 3. Secret, predictor, speculator, controller ~1 min
bash EKS_deployment/cloud/ui_cloud.sh up             # 4. the UI pod, then a tunnel to it
```

1. When step 4 prints `Pulse UI: http://127.0.0.1:8765`, open that address. **Keep that terminal open**: it is the tunnel.
2. The top bar shows a green **running in EKS** pill and **Tools run on: Knative (pulse-eks (in-cluster))**.
3. The Predictor card shows **ready** after about a minute. The controller is the cluster's `pulse-controller`.
4. If the tunnel drops (laptop sleep, Wi-Fi), the script reconnects by itself. If you closed it, reopen it with `bash EKS_deployment/cloud/ui_cloud.sh open`. The UI kept running in the cluster.

**After changing any code:** rebuild the image (step 1), then run step 4 again.

---

## 4. A tour of the screen

```
┌───────────────────────────── top bar ─────────────────────────────────────────┐
│ Pulse        [running in EKS]   Tools run on: [Simulated cluster | Knative] ↻  │
├──────────── main area (tabs) ─────────────────────┬──────── side panel ───────┤
│ Live demo | Experiments | Results | How it works  │ Predictor   (model, θ)    │
│                                                   │ Tool pods   (14 tools)    │
│                                                   │ Prewarm activity (feed)   │
└───────────────────────────────────────────────────┴───────────────────────────┘
```

**Top bar**

| Control | What it does |
|---|---|
| **Tools run on** | *Simulated cluster* or *Knative*. Knative uses the real cluster: the current kubectl context on the laptop, the cluster itself in EKS. The switch is refused if the cluster is not reachable, or while something is running |
| **↻** | Checks the cluster again (context, tools deployed) |
| **running in EKS** | Shown when the UI itself runs in the cluster |

**Side panel**

| Card | What it shows / does |
|---|---|
| **Predictor** | **Model**: LightGBM (main), Logistic regression, n-gram baseline, Trajectory ensemble (older), LLM speculator (Groq). **Prewarm policy**: *θ threshold* (every tool with P ≥ θ) or *Top-k* (the k most likely tools, each with P ≥ the floor). **Lookahead**: also prewarm, during the customer's turn, the tool the agent will call after the reply. **Start** / **Stop**. Status pill: *stopped*, *loading…*, *ready*, *exited*. The info line shows model, policy, whether PrewarmRequests are written (`CRDs k8s`) or not (`off`, simulated), and the controller state. Top-k and lookahead need the latency patch (LATENCY_IDEAS_RUN.md). An older predictor ignores them, and the card then shows a yellow note. **Changes apply by themselves:** about a second after you change model, policy, k, floor, θ or lookahead, the predictor restarts with the new settings (~1 min to load; the card shows what's running until then). During an episode the change waits until it ends, and a C3 episode always starts with the card's settings |
| **Tool pods** | One row per tool with a dot per pod: 🟢 ready, 🟡 starting (cold start in progress), 🔵 serving a call, grey = zero pods. A row flashes blue when a prewarm hits it. Simulated cluster only: **Scale to zero / Always warm (C1)**, **Reset** (all tools back to zero), and the pod-seconds used so far |
| **Prewarm activity** | Every prewarm (*was cold → starting* or *already warm, window reset*, with k and probability), every scale-down after 60 s idle, and min-scale changes |

---

## 5. Recipes: what to run

### Recipe 0: the end-to-end demo: one click, C0 then C3 (~2 min)

The **"End-to-end time: without vs with prewarming"** card shows how long the whole episode took, start to finish, without and with Pulse. It also shows how much of that time the agent spent waiting on tools.

1. **Live demo** tab:
   - **Episode:** Replay recorded
   - **Speed of gaps:** real time
   - **Fan-out:** on
   - **Task and recorded run**, chosen for the demo (below)
2. Predictor card:
   - **LightGBM**
   - **Prewarm policy: Top-k**, k 2, floor 0.1 (or θ 0.3, the original policy)
   - **Start**, then wait for **ready**
3. Click **⏱ Run C0 then C3 (same episode)**. The UI then:
   - **resets to a cold start:** simulated cluster → every tool to zero pods. Knative → restores the database and waits until every tool pod is gone; that takes 60 s idle plus any scale-down-delay
   - runs the episode **without** prewarming (C0), with the timeline's **⏱ clock** running
   - resets to a cold start again
   - runs the **same** episode **with** the predictor (C3)
4. The card shows both end-to-end times side by side, then a line like:
   > *With prewarming the episode finished 5.9 s faster (−11%): waiting on tools 9.7 s → 3.8 s, cold calls 6 → 2.*

**Which episode to demo.** Pick short episodes with several *different* tools and only 1–2 customer turns, so cold starts are a large share of the total. Measured on the simulated cluster with LightGBM top-2:

| Task / recorded run | C0 → C3 end-to-end | Tool wait | Cold calls |
|---|---|---|---|
| **Task 6, run #9** (held-out) | 52.6 s → 46.6 s (**−5.9 s, −11%**) | 9.7 s → 3.8 s | 6 → 2 |
| Task 30, run #1 (fan-out, many customer turns) | 60.0 s → 57.8 s (−2.2 s, −4%) | 5.7 s → 3.4 s | 9 → 3 |

Other good candidates, by the share of C0 time spent on cold starts: task 31 run #5, task 22 run #11, task 5 run #5/#6/#11.

**Choosing runs with or without fan-out.** In *Replay recorded*, the **Show tasks** filter picks:
- *All*, *Held-out*, or **Held-out, runs without fan-out**: tasks 6 (runs #1–#3), 20 (#1–#7, #12) and 22 (#1, #3, #4, #7, #11, #12). Tasks 26 and 30 have fan-out in every recorded run.
- *Runs with* / *without fan-out* across all tasks.

Each recorded run is listed with its fan-out width (or "no fan-out"), calls, tool steps, customer turns and the model that produced it. You can also untick **Fan-out** to fire every call of any run one at a time.
In the **Experiments** tab, the preset **Held-out, no fan-out** runs the held-out tasks with every call on its own (fan-out off).

**Why replay, not live, for this demo.** A replay with the same seed has *identical* think and customer-reply gaps in both runs, so the end-to-end difference is exactly the tool waiting saved. With the live agent, the LLM's speed and the AI customer's replies differ by seconds between runs, enough to hide a 2–6 s saving. If you do race live episodes, compare **Tool wait**, not E2E (the card says so too).

You can also do it by hand: run the episode with **None (C0)**, press **Reset** (simulated), run it with **Predictor (C3)**. The card pairs the latest C0 and C3 runs of the same episode.

### Recipe 0b: record a live run, then race its exact replay (C0 vs C3)

The most convincing demo: a **real** agent conversation, compared **fairly**.

1. **Record it.** Live demo → **Live agent** → AI customer (or **I'll type**, and chat yourself) → pick a task → **▶ Start episode**. Run it with or without prewarming; it doesn't matter. When it ends, the UI saves the episode's trajectory:
   - every customer message and how long the customer took to reply
   - every agent message and its think time
   - every tool call with its arguments, and which calls ran in parallel
   - the hint each step sent the predictor

   Every live episode with at least one tool call is recorded, including stopped or failed ones (up to where they ended).
2. **Race it.** Click **↻ Replay this exact run: C0 vs C3** at the end of the episode. Or, any time later: **Episode: Replay a live run** → pick it in **Your recorded live run** → **⏱ Run C0 then C3**.
3. **What you see.** Both replays show the original conversation (the real messages) with the original timing, and make the same tool calls. Only the prewarming differs, so the gap between the two end-to-end times is exactly the cold-start time Pulse saved. The card also shows how long the original live run took.

**Notes:**
- The original live run's own time can be longer than its replays. Recordings keep the model's think time but not Groq rate-limit waits or retries, which are not part of the agent's real behaviour.
- **Where recordings are kept:** `ui/.state/recordings/` (laptop), or `/home/app/ui-state/recordings/` in the EKS UI pod. `ui_cloud.sh fetch` copies the pod's recordings to the laptop.
- **Groq's per-minute token limit:** if a live episode fails with *"413 Request too large … tokens per minute"*, the conversation outgrew that limit for the agent model. Use **gpt-oss-20b** as the agent, or set a smaller `LLM_CONTEXT_BUDGET` (e.g. 4000) in `agent/.env`. The partial run is still recorded and replayable.

### Recipe 1: the 5-minute demo: C0 vs C3 on the same episode

The same recorded conversation runs twice: once without prewarming, once with Pulse.

1. **Live demo** tab. **Episode: Replay recorded**. **Task: 30 · fan-out · held-out**. **Recorded run #1**.
2. **Speed of gaps: real time.** Faster speeds shorten think time below a cold start, so prewarms can't win.
3. **Fan-out** on. **Prewarming: None (C0)**.
4. Side panel → **Tool pods → Reset** (simulated). On EKS, wait until the tool pods show grey (60 s idle).
5. **▶ Start episode**. Watch every first call of a tool come back **cold** (~2 s).
6. When it ends, **Reset** again. Switch **Prewarming: Predictor (C3)**. **▶ Start episode**.
7. Watch:
   - the predictor's guesses before each step
   - pods turning 🟡→🟢 *before* the calls
   - calls marked **prewarmed ✓ warm**
   - the `get_reservation_details` fan-out getting several pods at once
8. Scroll to **Episodes so far**: two rows, same task. Compare **Cold**, **Cold wait**, **Mean**, **Pod-s**.

### Recipe 2: watch the live agent with an AI customer

1. **Episode: Live agent**, **Customer: AI customer**.
2. Pick a **Task**. It shows the customer's goal and the expected actions.
3. **Agent model:**
   - **qwen3.8-27b** batches parallel lookups, the best way to show fan-out prewarming.
   - **gpt-oss-20b** / **120b** call one tool at a time.
4. **Customer model:** gpt-oss-20b (default).
5. **Customer reply delay:** how long the customer "takes to type" (0–15 s). Longer delays give the predictor more lead time, as with real people.
6. **Prewarming: Predictor (C3)** (or None for a baseline) → **▶ Start episode**.
7. **■ Stop** ends it at any time.

Each live episode uses Groq quota (agent + customer). The episode ends when the customer is satisfied, the agent transfers to a human, or after 30 steps.

### Recipe 3: be the customer yourself

1. **Episode: Live agent**, **Customer: I'll type**.
2. Pick a task (its goal is your brief) or **No task: free conversation**.
3. **▶ Start episode**. When the timeline says **Your turn**, type in the box at the bottom and press **Send** (or Enter).
4. Watch what the predictor does with your words. Example: give your user id → it expects `get_user_details`. Ask about a booking → `get_reservation_details`.
5. To end, send an empty message or `###STOP###`.

### Recipe 4: compare predictors on the same replay

1. Predictor card → **Model: LightGBM**, θ 0.30 → **Start**. Run a replay (Recipe 1, step 6).
2. **Reset** pods → Predictor card → **Model: n-gram baseline** (or Logistic regression, or LLM speculator) → **Start** → run the same replay.
3. Compare the rows in **Episodes so far**. The **Hint RTT** column shows how fast each predictor answers: LightGBM ~5–30 ms, the LLM speculator ~0.5–1 s.

Try θ too: lower θ (0.2) prewarms more (fewer cold calls, more pods); higher θ (0.5) prewarms less.

### Recipe 5: a proper experiment run (C0 / C1 / C3)

This is the same runner the report numbers come from. It needs **Knative** (Setup B or C).

1. **Experiments** tab.
2. **Mode: Replay** (seeded, no LLM, the fair comparison) or **Live** (the real agent, uses Groq).
3. **Configuration:**
   - **C0 · cold:** scale to zero, no prewarming.
   - **C1 · always warm:** the UI first switches every tool to min-scale 1 and waits until they're ready.
   - **C3 · prewarm:** pick the **Predictor** and **θ**. The UI starts that predictor with PrewarmRequests on.
4. **Tasks:**

   | Preset | Tasks |
   |---|---|
   | Held-out | 6 20 22 26 30: the tasks the predictor never saw, the report's workload |
   | Fan-out tasks | 2 3 28 30 31 33 34 |
   | Task 30 only | 30 |
   | All W1 tasks | every task (replay) |
   | Custom | ids you type |

   **C3 prewarm policy:** *θ threshold*, or *Top-k* with k and a floor, plus an optional **Lookahead**. Top-k and lookahead need the latency patch. The run's folder name gets `-top2` / `-look` added.
5. **Replay options:**

   | Option | Default | Meaning |
   |---|---|---|
   | Episodes | 46 | Number of episodes |
   | Seed | 252 | Keep it the same across C0/C1/C3 |
   | Customer reply median | 8 s | Customer turn time |
   | Fan-out | on | Parallel lookups |

6. **Live options:** agent model, repeats, customer delay median.
7. **Runner:**

   | Runner | Where the run happens | Where results go |
   |---|---|---|
   | In EKS: **This UI pod** | The UI pod, on the client node | The pod's own folder |
   | On the laptop: **Laptop** | The laptop | `results/…` |
   | On the laptop: **Run pod (run_cloud.sh)** | A separate pod in EKS | Copied back to the laptop |

8. **Label:** the end of the result folder name. Use `t1`, `t2`, `t3` for the 3 trials.
9. **▶ Start run.** The **Run log** streams everything:
   - the min-scale switch, waiting for the predictor
   - "all 14 tools at zero, database reset; starting"
   - one line per finished episode
   - the final summary

   A held-out replay takes ~25–35 min.
10. When it finishes, the **Result** card shows calls, cold calls, cold share, mean / p50 / p95 / p99, pod-seconds and hints answered, plus the result folder.

**The full comparison:** run C0, then C3 (LightGBM), then C3 (LLM speculator), all with the same tasks and seed. Optionally add C1. Then use Recipe 6.

Only one thing runs at a time: an experiment run and a demo episode share the tools, so the UI refuses to start one while the other is running.

### Recipe 6: compare runs and make the report plots

1. **Results** tab → the runs table, newest first. Filter by folder with the dropdown.
2. Tick the runs to compare. They must share the workload and seed: same tasks, same episodes.
3. **Compare selected:** bar charts of cold calls, mean, p95, p99 and pod-seconds. Colours: C0 orange, C1 blue, C3 (LightGBM) green, other predictors in other colours.
4. **Make plots (plot_compare.py):** creates the four report figures. They appear under **Plots** and are saved in a `compare_<date>_ui_<n>runs` folder:
   - `1_latency_cdf.png`
   - `2_percentiles.png`
   - `3_cost.png`
   - `4_cold_starts.png`
   - plus `comparison.json`, the numbers behind them.
5. Click an image to open it full size. The **zip** link on a run row downloads that whole run folder.

---

## 6. Reading what you see

### The episode timeline (Live demo)

| Item | Meaning |
|---|---|
| **Customer** bubble (left) | The customer's message, with seconds since the start. In replays: "(customer replies after X s)" |
| **Agent** bubble (right) | The agent's reply to the customer, with its think time |
| *agent thinking…* / *customer is typing…* | An LLM call is in progress |
| **LLM step N** box | One agent decision, with the predictor's line and the tool calls that step made |
| **Predictor** line | Top guesses with probabilities. **Green-outlined** guesses ≥ θ were prewarmed. Then "→ prewarm X ×k" or "nothing to prewarm", and the predictor's response time. *(reply step)* = a step where the agent is about to answer the customer |
| **Tool call** row | Tool name, arguments (grey), first part of the result (↳). Then the badges and the latency |
| **warm** / **cold** | Cold = the call took over 1 s, the same rule as the experiment runners |
| **prewarmed ✓** | The predictor asked for this tool before the call |
| **waited N ms (prewarm landed late)** | The pod was prewarmed but still starting when the call arrived (simulated cluster only). The lead time was shorter than the cold start |
| **∥ N** | N calls fired together (fan-out) |
| **Summary** (end) | Tool calls, cold calls, prewarmed calls, mean latency, seconds lost to cold starts, pod-seconds (simulated), hints answered |

**What good looks like (C3):**
- On reply steps, the predictor says RESPOND (nothing to prewarm).
- On tool steps, the right tool is green-outlined.
- Its call is **prewarmed ✓ warm** at ~40 ms.

A **prewarmed ✓ cold** call means the guess was right but too late: the think time was shorter than the cold start.

### Episodes so far

One row per finished demo episode, newest first:

| Column | Meaning |
|---|---|
| Type | live (ai / human) or replay |
| Task | Task id (replays: and the recorded run #) |
| Prewarming | none (C0), or the predictor and its policy (e.g. `gbt · top-2 (P ≥ 0.1)`) |
| Cluster | simulated or the cluster name |
| **E2E** | End-to-end time: from the conversation's first message to the end of the episode (setup excluded) |
| **Tool wait** | The part of E2E the agent spent blocked on tool calls. A parallel step counts once, as its slowest call. Prewarming can only shorten this part |
| Calls / Cold / Prewarmed | Tool calls, cold calls, calls the predictor had prewarmed |
| Mean | Mean call latency |
| Cold wait | Total seconds the agent spent waiting on cold starts |
| Pod-s | Pod-seconds used during the episode (simulated cluster), the resource cost |
| Hint RTT | Median predictor response time |

These rows live in the UI's memory and reset when the UI restarts. For numbers you keep, use experiment runs (Recipe 5).

### Experiment results: the metrics

| Metric | Meaning | Better |
|---|---|---|
| Cold calls / cold share | Calls that waited for a pod | Lower |
| Mean latency | Average call time: falls with every avoided cold start | Lower |
| p50 / p95 / p99 | Latency percentiles. p95/p99 drop sharply only once the cold share is below 5% / 1% | Lower |
| Pod-seconds | Resource cost: total time pods existed | Lower (C1 is the ceiling) |
| Hints answered | Predictions that came back in time | sent = answered |

Report targets (NFRs):
- **Latency:** C3 p50 ≥ 20% and p99 ≥ 15% lower than C0.
- **Cost:** ≥ 30% fewer pod-seconds than C1.

---

## 7. Where the results are

| What | Setup A (laptop, simulated) | Setup B (UI in EKS) | Setup C (laptop + kind) |
|---|---|---|---|
| Demo episodes | Only in the UI (Episodes so far) | Same | Same |
| Experiment runs | n/a (needs Knative) | In the pod: Results → folder **this pod**. Copy home with `ui_cloud.sh fetch` → `results/cloud_runs/` | `results/local_runs/` |
| Runs from before | Results tab: `cloud`, `local`, `legacy` | Included read-only in the image: `cloud`, `local` | Same as A |
| Comparison plots | Next to the first selected run | In the pod (**this pod**), fetched with the runs | Next to the first run |
| UI / predictor / controller logs | `ui/.state/logs/` | In the pod: `/home/app/ui-state/logs/`; controller: `kubectl logs deploy/pulse-controller` | `ui/.state/logs/` |

A run folder is named `<date>_<time>_<config>_W1-<replay|live>_<details>_<label>`. It contains:

| File | Contents |
|---|---|
| `summary.json` | All the totals and percentiles shown in the UI |
| `calls.csv` | One row per tool call (latency, cold, fan-out group, gap before it, …) |
| `pods.csv` | Every pod seen (for pod-seconds) |
| `episodes.jsonl` | Per episode |
| `run_config.json` | Every setting, code version, image |
| `hints.jsonl` | Every hint + the predictor's answer (C3) |
| `predictor_health.json` | The predictor's model, θ, CRDs written (C3) |
| `controller.jsonl` | The controller's actions. Laptop runs: in `ui/.state/controller.jsonl`. EKS: `kubectl logs deploy/pulse-controller` |

**Getting EKS results home (important).** The UI pod's folder disappears with the pod or the cluster. Before tearing down:

```bash
bash EKS_deployment/cloud/ui_cloud.sh fetch       # copies every new run / plot folder to results/cloud_runs/
```

`eks_down.sh` also runs this automatically if the UI is deployed. For a single run, the **zip** link in the Results tab works too.

---

## 8. Stop, clean up, costs

| To… | Do |
|---|---|
| Stop a demo episode | **■ Stop** (Live demo) |
| Stop an experiment run | **■ Stop** (Experiments). The partial result folder stays |
| Free the predictor | Predictor card → **Stop** |
| Close the UI (laptop) | Ctrl+C in its terminal |
| Close the tunnel (EKS) | Ctrl+C in the `ui_cloud.sh` terminal. The UI keeps running in the cluster |
| Remove the UI from EKS | `bash EKS_deployment/cloud/ui_cloud.sh down` (fetch first) |
| **Stop paying for EKS** | `bash EKS_deployment/eks_down.sh` (fetches the UI's runs first), then the checks in RUN_EXPERIMENTS.md A8 |

EKS costs money every hour it exists, whether or not you use the UI. Tear it down when you're done for the day.

---

## 9. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Predictor stays **loading…** for > 2 min | It's still importing (slow disk). If it turns **exited**: read the last log lines on the card, or `ui/.state/logs/predictor.log` (EKS: `kubectl exec deploy/pulse-ui -- tail -50 /home/app/ui-state/logs/predictor.log`) |
| "Start the predictor first" | The episode is set to C3 but no predictor is running → Predictor card → **Start**, or choose *None (C0)* |
| Switching to **Knative** is refused: "not reachable" | No cluster in the current kubectl context (laptop), or the cluster is down → check `kubectl get nodes`. On EKS, run steps 2–3 of Setup B |
| "Experiment runs need a real cluster" | You are on the simulated cluster → switch to **Knative** (top bar) |
| "an interactive episode is running" / "an experiment run is in progress" | One thing at a time → stop it or wait |
| Every C3 call is cold, even "prewarmed ✓" | Replay **speed** faster than real time, or a 0 s customer delay. Lead time is shorter than the cold start → use real time / a longer delay |
| Live episode fails with a rate-limit / quota error | Groq daily limit on all keys → wait, add keys in `agent/.env` (then rerun `cloud_up.sh` on EKS), or use replays |
| Tool pods board empty (Knative) | Pods are listed by the label `pulse/role=mcp-tool`: are the tools deployed? `kubectl get ksvc` |
| Prewarm activity empty on EKS | The controller deployment isn't running → `kubectl get deploy pulse-controller` (from `cloud_up.sh`) |
| C1 run fails at "set the tools to min-scale 1" with *forbidden* | The cluster's RBAC predates the UI → run `bash EKS_deployment/cloud/cloud_up.sh` again |
| `ui_cloud.sh up` says *No image* | Build it: `bash EKS_deployment/cloud/build_image.sh` |
| Pod `ImagePullBackOff` | Image tag not pushed → rebuild (step 1), then `ui_cloud.sh up` |
| The browser can't reach 127.0.0.1:8765 (EKS) | The tunnel isn't running → `bash EKS_deployment/cloud/ui_cloud.sh open` |
| "Make plots" fails on the laptop | No Python with matplotlib → set `PLOT_PYTHON` to one that has it (e.g. `C:\Python311\python.exe`) |

---

## 10. Quick reference

```bash
# ---- laptop, simulated cluster
agent/.venv/Scripts/python ui/server.py                 # → http://127.0.0.1:8765

# ---- EKS (once per session)
bash EKS_deployment/cloud/build_image.sh                 # after any code change
PYTHON="$PWD/agent/.venv/Scripts/python" bash EKS_deployment/eks_up.sh
bash EKS_deployment/cloud/cloud_up.sh
bash EKS_deployment/cloud/ui_cloud.sh up                 # → http://127.0.0.1:8765 (keep the terminal open)
bash EKS_deployment/cloud/ui_cloud.sh open               # re-open the tunnel
bash EKS_deployment/cloud/ui_cloud.sh fetch              # runs → results/cloud_runs/
bash EKS_deployment/eks_down.sh                          # fetches, then deletes the cluster
```

| I want to… | Go to |
|---|---|
| Show the end-to-end saving in one click | Live demo → Replay recorded → task 6, run #9 → **⏱ Run C0 then C3** (Recipe 0) |
| Race a real conversation fairly | Live agent episode → **↻ Replay this exact run: C0 vs C3** (Recipe 0b) |
| Show prewarming in 5 minutes | Live demo → Replay recorded → task 30 → C0, Reset, C3 (Recipe 1) |
| Talk to the agent | Live demo → Live agent → I'll type (Recipe 3) |
| Try another predictor / θ | Predictor card → Model, θ → Start (Recipe 4) |
| Get report numbers | Experiments → Replay, Held-out, C0 then C3 (Recipe 5) |
| Make the report figures | Results → tick runs → Make plots (Recipe 6) |
| Take results home from EKS | `ui_cloud.sh fetch` or the zip link |
