# Pulse demo UI

A browser UI on top of Pulse. You can:
- watch an LLM agent call the 14 airline tools, live or replayed
- see the predictor rank the next action before every LLM step
- watch prewarms start pods before the calls arrive
- launch the experiment runs (C0 / C1 / C3) and compare their results

```bash
agent/.venv/Scripts/python ui/server.py        # from the repo root
```

Then open **http://127.0.0.1:8765**. The server listens on localhost only.
It starts the LightGBM predictor in the background at launch; loading takes about a minute on this laptop.

## Two places the tools can run (top bar)

| | What it is | Needs |
|---|---|---|
| **Simulated cluster** (default) | Pods, cold starts, the 60 s idle window and prewarming, simulated on the laptop (`ui/sim_cluster.py`) | Nothing but the venv |
| **Knative** | The real cluster in your kubectl context: kind (`kind-pulse-kn`) or EKS. The UI starts the predictor with `CRD_MODE=k8s` and the prewarm controller | A running cluster with the 14 tools (RUN_EXPERIMENTS.md, B1 or A2) |

How the simulated cluster works:
- **Cold-start times** are the per-tool times measured on EKS (`deploy/knative/results/*_tools-cold-warm.csv`), about 1.6–2.5 s.
- **Tool results** are real: the tools run on the airline database.
- **Pod behaviour** matches the real tools' settings: containerConcurrency 1, up to 10 pods, scale to zero after 60 s idle.
- **Prewarming** with k starts k pods, like the controller's k pings.

## Live demo tab

| Setting | Options |
|---|---|
| Episode | **Live agent**: the ReAct agent (Groq). **Replay recorded**: the tool calls of a recorded tau-bench conversation with seeded think / customer-reply gaps and fair hints. Needs no LLM |
| Customer (live) | **AI customer**: `gpt-oss-20b` plays the task's customer, with a reply delay you choose. **I'll type**: you are the customer, with an optional task goal shown as your brief |
| Task | Any of the 50 tau-bench airline tasks. Tags: *fan-out* (parallel lookups), *held-out* (never seen in training) |
| Agent model | `qwen3.8-27b` batches parallel tool calls; gpt-oss calls one tool at a time |
| Prewarming | **Predictor (C3)** or **None (C0)**. Choose the model and θ on the Predictor card |

The timeline shows the episode step by step:
- **Every LLM step:** the predictor's top guesses with probabilities, what it prewarmed (×k for fan-out), and its response time.
- **Every tool call:** latency, **warm/cold** (cold = slower than 1 s, the runners' rule), **prewarmed ✓**, and "waited … ms (prewarm landed late)" when the prewarm came too close to the call.
- **At the end:** a summary row is added to **Episodes so far**, for side-by-side comparison.

**Side panel:**
- **Tool pods:** live pods per tool (green ready, amber starting, blue serving). The pods come from `kubectl get pods` on Knative, or from the simulation.
- **Prewarm activity:** every prewarm and scale-down, from the controller log on Knative or from the simulation.
- **Simulated cluster controls:** *Scale to zero* / *Always warm (C1)*, and *Reset* (every tool back to zero pods).

### A 5-minute demo script

1. Keep the **Simulated cluster**. Pick task 30 (*fan-out*), **Replay recorded**, real-time gaps.
2. Prewarming **None (C0)** → Start. Every first call of a tool is cold (~2 s).
3. **Reset** pods, switch to **Predictor (C3)** → Start the same replay. The predictor prewarms during the customer's reply and think time, so most calls are warm, and the `get_reservation_details` fan-out gets k pods at once.
4. Compare the two rows in **Episodes so far**: cold calls, cold wait, pod-seconds.
5. Switch to **Live agent** + **I'll type**, and chat as the customer. Watch the predictor react to what you write.

## Experiments tab

This tab is a form for the same runners as [RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md). It needs **Knative** mode.

- **Settings:** replay or live; C0 / C1 / C3; predictor and θ; tasks; episodes; seed; customer-reply median; fan-out; for live runs, the agent model, repeats and customer delay.
- **Runner = Laptop:** the UI sets the tools' min-scale if needed (`deploy/knative/deploy.sh`), starts the predictor and controller for C3, then runs `w1_replay.py` / `w1_live.py`.
- **Runner = In-cluster pod:** `EKS_deployment/cloud/run_cloud.sh`. Needs `cloud_up.sh` done; the predictor and θ are the in-cluster ones.
- **While it runs:** the log streams live. Results go to `results/local_runs/` (kind) or `results/cloud_runs/` (EKS), and the summary appears when the run finishes.

## Results tab

- **Runs:** every run folder (`results/cloud_runs`, `results/local_runs`, and the runs before 25 Oct), with cold calls, latency percentiles and pod-seconds.
- **Compare selected:** bar charts of cold calls, mean, p95, p99 and pod-seconds.
- **Make plots:** runs `test/test_knative/plot_compare.py`, using a Python with matplotlib: `PLOT_PYTHON`, else `C:\Python311\python.exe`. Saves a `compare_…_ui_…` folder next to the runs and shows the PNGs, together with every earlier comparison.

## Files

| File | What |
|---|---|
| `server.py` | FastAPI app: the API and the page |
| `backend.py` | Cluster detection, predictor / controller processes, batch runs, result folders |
| `session.py` | One interactive episode (live agent or replay), with hints and prewarms, as a stream of events |
| `sim_cluster.py` | The simulated Knative cluster |
| `static/` | The page (plain HTML / CSS / JS, no build step) |
| `.state/` | Logs and pid files of the processes the UI starts (git-ignored) |

Settings (environment):

| Variable | Default | Meaning |
|---|---|---|
| `UI_PORT` | 8765 | Port the UI listens on |
| `UI_PREDICTOR_PORT` | 8000 | Port of the predictor the UI starts |
| `UI_AUTOSTART_PREDICTOR` | 1 | Set to 0 to skip loading gbt at launch |
| `PLOT_PYTHON` | | Python used for the plots |

## Notes

- **One thing at a time:** an interactive episode and an experiment run share the tools, so the UI runs only one at a time.
- **Groq keys:** the live agent, the AI customer and the LLM speculator use the keys in `agent/.env`. The UI never shows or sends them anywhere else.
- **Episode scores:** live episodes on the simulated cluster use a fresh copy of the airline database. On Knative they use the shared Redis copy, which batch runs reset at their start.
- **If the predictor shows *exited*:** its last log lines are on the Predictor card. The full log is `ui/.state/logs/predictor.log`.
