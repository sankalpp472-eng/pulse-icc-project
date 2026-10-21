# Running the C0 baseline (Knative, local cluster)

End-to-end guide: install the tools, create a local Kubernetes cluster with Knative, deploy the
14 airline MCP tools (plus their shared database) as serverless functions, and run the C0 baselines.

**C0 = scale-to-zero, no prewarming.** A tool that gets no calls for the **idle window** (60 s,
Knative's default) is scaled to zero; its next call is a cold start. **C1** = one warm replica per tool (same steps, one flag changed).

All commands run from the **project root** unless a step says otherwise.

| Step | What | Time |
|---|---|---|
| 1–2 | Install tools, Python environment | one time |
| 3 | Cluster + Knative | ~5 min |
| 4 | Single-function C0 check | ~8 min |
| 5 | Deploy shared DB + 14 MCP tools | ~3 min |
| 6 | Per-tool cold vs warm check | ~3 min |
| **7** | **W1 workload C0, live agent** | hours (until the LLM quota runs out) |
| **7b** | **W1 workload C0, replay of recorded traces** (the clean C0 numbers) | ~30 min |
| 8 | Clean up | 1 min |

> **Status (19 Oct 2026):** steps 1–7b are built and tested. Step 7b gives the C0 baseline
> numbers; step 7 (live agent) is limited by the LLM provider's daily quota.

---

## 0. What you need

| | Minimum |
|---|---|
| Laptop | 8 GB RAM, 4 CPU. Close heavy apps (many Chrome tabs) before starting |
| Docker Desktop | Running, **not paused**, with ≥ 4 GB memory (Settings → Resources) |
| Shell | **Windows: Git Bash** (not PowerShell). macOS/Linux: any terminal |
| Python | 3.11 |
| Groq API key | Only for step 7 (free tier is enough, but it limits how many tasks run per day) |

Commands below are for **Windows + Git Bash**. On macOS/Linux, replace
`agent/.venv/Scripts/python` with `agent/.venv/bin/python`.

## 1. Install the command-line tools (one time)

`kubectl` comes with Docker Desktop. Install `kind` (Kubernetes in Docker) with arkade:

```bash
curl -sLS https://get.arkade.dev | sh
```
```bash
arkade get kind --os mingw
```
```bash
export PATH="$HOME/.arkade/bin:$PATH"
```

macOS: `brew install kind`. Linux: `arkade get kind`. Check that everything is found:

```bash
docker info --format '{{.ServerVersion}}' && kubectl version --client && kind version && python --version
```

## 2. Set up the Python environment (one time)

```bash
python -m venv agent/.venv
```
```bash
agent/.venv/Scripts/python -m pip install -r agent/requirements.txt
```

If pip fails with an SSL or "invalid wheel" error, the network dropped a download. Rerun the
same command (add `--no-cache-dir` if it repeats).

For step 7 you also need a Groq API key:

```bash
cp agent/.env.example agent/.env
```

Then edit `agent/.env` and set `GROQ_API_KEY`. **Never commit `.env`** (it is in `.gitignore`).

## 3. Create the cluster and install Knative (about 5 minutes)

```bash
bash test/test_knative/setup.sh
```

Creates a 2-node kind cluster `pulse-kn`, installs **Knative Serving v1.23 + Kourier** (ingress on
`http://127.0.0.1:8081`), and deploys a small test function `cold-tool`. Safe to rerun if a
download fails halfway. Quick check (Knative routes by the `Host` header):

```bash
curl -s -X POST -H "Host: cold-tool.default.example.com" http://127.0.0.1:8081/ -d '{}'
```

## 4. Single-function C0 check (about 8 minutes)

```bash
cd test/test_knative && python c0_baseline.py --trials 10 && cd ../..
```

Each trial waits until Knative scales the function to zero by itself, then makes one cold call and
five warm calls. Results: `test/test_knative/results/c0_<time>.csv` and `_summary.json`.

**Our result:** 10/10 cold calls succeeded; cold p50 **1.69 s**, warm p50 **0.11 s**.

## 5. Deploy the shared database and the 14 MCP tools (about 3 minutes)

Each τ-bench airline tool is its own MCP server and its own Knative service
(`get-user-details`, `cancel-reservation`, …), all from the public image
`sankalps2003/pulse-mcp-tools:0.2`. The tools share one **Redis** database, so a write by one tool
(e.g. a cancellation) is seen by the others.

```bash
SKIP_BUILD=1 bash deploy/knative/deploy.sh
```

What it does: deploys Redis → loads the airline data into it (a one-off pod) → deploys the 14
tool services with `min-scale: 0` and a **60 s idle window** → waits until all are Ready.

- `SKIP_BUILD=1` uses the existing public image. Only the image owner runs it **without**
  `SKIP_BUILD` (rebuilds and pushes; needs `docker login`). To use your own Docker Hub account:
  `IMAGE=docker.io/<you>/pulse-mcp-tools:0.2 bash deploy/knative/deploy.sh`.
- **C1** (always warm): `SKIP_BUILD=1 bash deploy/knative/deploy.sh 1`. Back to C0: the same
  command without `1`.
- `WINDOW=6s` (or any value ≥ 6s) changes the idle window, e.g. for a sensitivity sweep. A tool
  stays warm for roughly the window plus a few seconds after its last call (with `6s` we measured
  warm up to a 12 s gap, cold from 13 s).
- `NO_REDIS=1` deploys without the shared database (each tool pod keeps its own copy). Not
  recommended: multi-step tasks then see stale data.

Check, and watch the tools scale to zero about a minute after their last call:

```bash
kubectl get ksvc -l pulse/role=mcp-tool
```
```bash
kubectl get pods -l pulse/role=mcp-tool -w
```

## 6. Per-tool cold vs warm check (about 3 minutes)

```bash
agent/.venv/Scripts/python deploy/knative/check_tools.py
```

For each tool: wait until it is at zero, one cold MCP call, one warm MCP call. Results:
`deploy/knative/results/tools_cold_warm_<time>.csv`.

**Our results (16 Oct 2026, before the shared database):** 14/14 tools OK; cold p50
**3.9–5.8 s** (range 2.7–12.2 s), warm p50 **51–64 ms**. Real MCP tools start much slower than
the small test function in step 4 (MCP SDK + airline data load at start-up).

## 7. W1 workload C0 baseline, live agent

The real agent and a simulated customer (both Groq `gpt-oss-20b` by default) solve **W1 = the 23
τ-bench airline "change / cancel a booking" tasks** while all tool calls go to the 14 tools on
Knative.

```bash
agent/.venv/Scripts/python test/test_knative/w1_live.py
```

| Option | Runs |
|---|---|
| *(none)* | All 23 tasks: the 12 one-write tasks first, then the 11 multi-tool-write tasks |
| `--set safe` / `--set multi` | Only one group |
| `--tasks 1 6 7` | Specific task ids |
| `--set best5` | The 5 most-branching tasks (4, 9, 10, 23, 27), together covering 13 of the 14 tools. **Recommended** |
| `--repeats N` | Run every task N times |
| `--backend groq` | LLM provider (default: `LLM_BACKEND` in `agent/.env`) |
| `--agent-model M` / `--user-model M` | Groq models (default `openai/gpt-oss-20b` for both) |

**Groq free-tier limits** (per organization, per model): 8K tokens per minute, 200K per day.
- **Several keys:** put `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, … in `agent/.env`. When one hits its daily
  limit for a model, the next takes over; `summary.json` → `key_switches` logs each switch and the key's
  organization. Keys from the **same** Groq organization share one quota, so switching doesn't help there.
- **8K per request:** any single request above 8K tokens is rejected (HTTP 413). The agent therefore
  trims its context before each request: the oldest tool results are replaced by a short note once the
  estimate passes 6,500 tokens (`LLM_CONTEXT_BUDGET` in `.env`), keeping the 2 latest in full. Saved traces
  are always complete.

Recommended runs (two agent models, same simulated customer):

```bash
agent/.venv/Scripts/python test/test_knative/w1_live.py --backend groq --set best5 --repeats 2 --agent-model openai/gpt-oss-120b --user-model openai/gpt-oss-20b
```
```bash
agent/.venv/Scripts/python test/test_knative/w1_live.py --backend groq --set best5 --repeats 2 --agent-model openai/gpt-oss-20b --user-model openai/gpt-oss-20b
```
| `-v` | Also print the conversation |

**How it runs:** waits until all 14 tools are at zero → for each task, resets the shared database
to the original data → runs the agent → records every tool call. When Groq's quota runs out
(a rate-limit wait over 2 minutes, or a per-day limit), it **stops cleanly and keeps all data**.
Rerun later with `--tasks <the remaining ids>`.

**Cold or warm?** Every call is labelled with two signals, both kept:
- *pod signal*: a new pod for that tool appeared during the call (a background watcher polls the
  pods every second);
- *latency signal*: the call took more than 1 s (warm calls take ~50–100 ms).

A call is **cold** if either says so. Nothing is forced cold: tools go cold only through normal
idleness, which is what C0 means. Each call also records the **idle gap** since that tool was last
used.

**Output:** `test/test_knative/results/w1_live_<time>/`

| File | Contents |
|---|---|
| `run_config.json` | **What this run was:** models, C0/C1, idle window, Knative version and autoscaler settings (read from the cluster), every tool's image and revision, database mode, cluster, git commit. Written at the start |
| `calls.csv` | One row per tool call: task, tool, start time, latency, cold/warm (+ both signals), pod, idle gap, LLM think time and rate-limit wait before the call |
| `episodes.jsonl` | One episode per line: the full agent trace (steps, tool sequence, timings, status) |
| `pods.csv` | Every tool pod seen: tool, created, first/last seen |
| `summary.json` | Cold share, cold and warm p50, per-tool cold share, why the run stopped, models, window |

**Read the results with care:** Groq's free tier makes the agent wait 10–50 s between calls. Those
waits stretch idle gaps toward (or past) the idle window, so the live run **overstates** how often
tools are cold.
The waits are recorded (`rate_wait_s`) so this stays visible; the replay version gives the clean
numbers.

## 7b. W1 workload C0 baseline, replay (about 30 minutes, no LLM needed)

Replays **real recorded agent runs** from τ-bench (GPT-4o and Claude 3.5 Sonnet on the same 23 W1
tasks, 276 traces in `test/test_knative/data/w1_tau_traces.jsonl`) against the 14 tools. Every
call uses its **recorded arguments**, so the tools do real work. No LLM is called, so there are no
rate limits or quota.

```bash
agent/.venv/Scripts/python test/test_knative/w1_replay.py
```

| Option | Default | Meaning |
|---|---|---|
| `--episodes N` | 46 | Episodes to replay (46 = every task twice) |
| `--rate R` | 2.0 | Average new episodes per minute; they start at random times and overlap |
| `--think-median S` | 1.5 | Median LLM "think" time before each call (seconds) |
| `--user-turn-median S` | 8.0 | Median time per customer turn, added when the recorded agent talked to the customer between two calls |
| `--seed N` | 252 | Fixes trace choice, start times and gaps: **same seed = same request sequence** for C0, C1 and C3 |
| `--traces DIR ...` | τ-bench runs | Replay **our agent's live runs** instead: pass one or more `results/w1_live_<time>` folders. Each recorded episode is replayed once, with its **real** gaps (agent think time + the customer's reply time) |
| `--tasks 4 9 10 23 27` | all | Only these tasks |
| `--gap-scale X` | 1.0 | Multiply the recorded gaps of our traces (e.g. 2 = a slower LLM) |

Example, replaying the two best-5 live runs (120b and 20b agents):

```bash
agent/.venv/Scripts/python test/test_knative/w1_replay.py --traces test/test_knative/results/w1_live_<120b run> test/test_knative/results/w1_live_<20b run>
```

It waits until all tools are at zero, resets the database once, then runs the episodes
concurrently. Calls are labelled cold/warm after the run (latency > 1 s, or a new pod for that
tool appeared during the call).

**Output:** `test/test_knative/results/w1_replay_<time>/` with the same files as step 7
(`run_config.json` includes all replay parameters). `summary.json` adds p50/p95/p99 latency,
total time added by cold starts, pods started and pod-seconds.

**For C1** (always warm): `SKIP_BUILD=1 bash deploy/knative/deploy.sh 1`, then run the replay with
the **same seed**, then switch back with `SKIP_BUILD=1 bash deploy/knative/deploy.sh`.

## 8. Clean up

```bash
bash test/test_knative/teardown.sh
```

Deletes the cluster (Redis and its data included) and frees Docker's memory. Next time: steps 3
and 5 again.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Docker Desktop is manually paused` | Docker paused | Whale icon → Resume. Keep it running during step 7 |
| `connection refused` / `TLS handshake timeout` on 127.0.0.1 after a reboot or sleep | kind lost its port mappings | `bash test/test_knative/teardown.sh && bash test/test_knative/setup.sh`, then step 5 |
| `The paging file is too small` | Laptop out of memory | Close apps, rerun the step |
| `error parsing https://github.com/...` or `connection forcibly closed` | Network dropped a download | Rerun the same script |
| `curl` returns 404 | Missing or wrong `Host:` header | `kubectl get ksvc` shows the right host |
| `Python was not found` (Windows) | `python3` is a Microsoft Store placeholder | Use `python`; `deploy.sh` already does |
| Tool pods crash with "Redis ... has no airline data" | Redis restarted (it keeps data in memory only) | Rerun step 5 (it reseeds) |
| Step 7 prints `quota_exhausted` | Groq daily limit reached | Normal: data so far is saved. Rerun the remaining task ids tomorrow |
| A tool never scales to zero | Something keeps calling it, or a broken old revision | `kubectl get revisions`; delete the broken one |

## Where things are

| Path | What it is |
|---|---|
| `test/test_knative/` | This guide, cluster setup, single-function C0 (`c0_baseline.py`), W1 live C0 (`w1_live.py`), results |
| `deploy/knative/` | `deploy.sh`, `gen_services.py` (tool manifests), `redis.yaml`, `check_tools.py` |
| `agent/mcp_server.py` | Each τ-bench tool as its own MCP server; optional shared Redis database |
| `agent/Dockerfile.mcp` | The tool image |
| `agent/tools.py` | Agent-side tool backends (local or MCP) and the call log |
| `test/test_cold_start/` | The earlier OpenFaaS CE test (why we moved to Knative) |
