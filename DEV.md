# Pulse: developer documentation

What each part of the code does and how the parts fit together. For installing and using the project, see
[README.md](README.md). Every module and script also explains itself (settings, commands) at the top of its file.

---

## 1. Architecture

<img width="2960" height="1900" alt="architecture" src="https://github.com/user-attachments/assets/874a4899-c8d0-4e5d-80ba-52056bbb768a" />

**One step of an episode:**
1. **Hint.** Before every LLM step the agent POSTs a hint to the predictor: the tools called so far, plus whatever
   that predictor reads (numeric signals, or the recent messages). Sent on a background thread; the agent never
   waits for it.
2. **Prediction.** The predictor ranks the next actions (each tool, `RESPOND` = answer the customer, `END`). The
   prewarm policy picks tools (adaptive top-k: up to 3 candidates, each warmed only if worth a pod slot now; or a
   fixed top k) and a fan-out width k per tool (the number of parallel calls expected), and writes one
   `PrewarmRequest` object per tool.
3. **Prewarm.** The controller watches `PrewarmRequest`s and sends the tool k concurrent MCP `ping`s through the
   ingress. With one request per pod (containerConcurrency 1), Knative starts k pods.
4. **Call.** The agent's real call finds a warm pod, or one that is already starting.
5. **Scale down.** Knative's idle window (60 s) is the TTL; unused pods scale back to zero.

**Configurations compared:** C0 scale-to-zero without prewarming (the cold baseline), C1 always warm
(min-scale 1: the latency floor and resource ceiling), C3 scale-to-zero with predictor and controller (Pulse).

---

## 2. Repository layout

| Path | Contents |
|---|---|
| `pulse/` | The Python package (installable: `pyproject.toml`, `pip install -e .`). Everything runs as `python -m pulse.<module>` |
| `data/` | Inputs. In git: `predictor/` (training rows, task split), `traces/` (replay traces), `coldstart/` (cold/warm times measured on EKS, for the simulated cluster). Gitignored: `tau-bench/` (downloaded), `pulse_v3/` (its embedding cache), `speculator/`, `predictor/retail_*` (the retail rows and split) |
| `models/` | Trained models. In git: `action_model.pkl` (lightgbm + markov). Gitignored: `action_model_retail.pkl`, `pulse_v3/` (trained here), `registry/` and `action-model/` (versions fetched from or published to S3) |
| `infra/` | Everything that is not Python: container images, Kubernetes manifests, cluster scripts, Terraform, GitOps, monitoring, training Jobs |
| `scripts/` | `setup.sh` (fresh clone → ready), `embed_server.sh` (pulse_v3's embedding server in Docker) |
| `results/` | Output of experiments, load tests and training (gitignored) |
| `.github/workflows/` | CI (`ci.yml`), the image pipeline (`image.yml`), Terraform plan (`terraform.yml`) |

`pulse/paths.py` is the single place that knows where files live (`ROOT`, `DATA`, `MODELS`, `RESULTS`,
`ENV_FILE`, `INFRA`, …). Pods and Jobs, whose image folders are read-only, redirect them with `PULSE_DATA`,
`PULSE_MODELS`, `PULSE_REGISTRY_CACHE` and `PULSE_RESULTS_DIR` (pulse-v3's own folders: `PULSE_DATA_DIR`,
`PULSE_MODELS_DIR`, in `pulse/pulse_v3/paths.py`).

---

## 3. The `pulse` package

### 3.1 `pulse.agent`: the agent, the customer and the tools

| Module | Role |
|---|---|
| `agent.py` | The ReAct agent: one LLM step at a time, calls one or more tools or answers the customer; sends a hint before every step (`hints.py`) |
| `langgraph_agent.py` | The same agent on LangGraph (an alternative implementation) |
| `llm.py` | The LLM client: Groq (several keys round-robin, rate-limit aware), OpenRouter or Ollama; context fitting for long conversations. Reads `.env` |
| `user.py`, `prompts.py` | The LLM-simulated customer (or a human), and the system prompts (the airline policy from `tau_airline/wiki.md`) |
| `tools.py` | Tool backends: `LocalToolBackend` (in-process tau-bench tools) and `MCPToolBackend` (MCP over HTTP to Knative, with the Host-header routing Kourier needs) |
| `hints.py` | Builds and sends hints: reads the predictor's `GET /info` once and sends only the inputs it reads |
| `mcp_server.py` | One MCP server per tool (stateless JSON over streamable HTTP), the shared Redis database, `--seed` to load it |
| `gen_services.py` | Generates the Knative Services for the tools (`infra/knative/tools.yaml`): min-scale, concurrency, idle window |
| `check_tools.py` | Cold vs warm call to every tool on a cluster; writes `data/coldstart/*_tools-cold-warm.csv` |
| `run_tasks.py` | Runs the agent on tau-bench tasks from the command line |
| `tau_airline/`, `tau_retail/` | tau-bench's tools, tasks, data and policy text (MIT, © Sierra) |

### 3.2 `pulse.predictor`: the prediction service and its five predictors

| Module | Role |
|---|---|
| `contract.py` | The one contract: `PredictRequest` (trajectory; optional `signals`, `history`, `domain`, `workflow_id`) → `PredictionOut` (`ranked` actions with probabilities, `prewarm` tools with k, `lookahead`, latency, inputs used) |
| `service.py` | FastAPI app: `POST /predict`, `GET /info` (the predictor's inputs and capabilities), `/healthz`, `/metrics`. Loads the predictor named by `PREDICTOR` and applies the shared policy, audit log and PrewarmRequest writer |
| `predictors/` | One file per predictor behind `base.Predictor` (`inputs`, `predict()`, `fanout()`, `lookahead()`, `health()`); `action.py` loads the pickle lightgbm and markov share (local file or the registry); `__init__.py` is the registry (`PREDICTOR=lightgbm|lightgbm_adapt|markov|speculator|pulse_v3`) |
| `policy.py` | Which ranked actions become prewarms: `adaptive` (default: the top 3 as candidates with their rank and hit rate; the controller warms those worth a slot, `pulse/controller/adaptive.py`) or a fixed `topk`; a floor `P_MIN` (0.05); never `RESPOND`/`END` |
| `calibration.py` | Adaptive top-k: how often a candidate of probability p really was the next call, learned online from each episode's next hint (per p bin, starting from p itself); `/healthz` shows it |
| `crd_writer.py` | Writes `PrewarmRequest` objects (`CRD_MODE=k8s`), or to a file (`dryrun`), on a background thread |
| `client.py` | What hint senders share: `inputs(url)` from `/info`, `make_hint()`, `compact()` messages |
| `signals.py` | Numeric conversation signals (step, pending IDs, last error, customer keywords, …): no raw text leaves the agent for lightgbm |
| `action_model.py` | The lightgbm (`GBTModel`) and markov (`NGramModel`) models, the pending-ID fan-out rule and the lookahead model, in one `ActionPredictor` (pickled as `models/action_model.pkl`) |
| `build_action_data.py` | Training rows from the tau-bench conversations (one row per LLM step) → `data/predictor/action_rows.csv` |
| `train_action.py` | Trains lightgbm + markov (+ lookahead), scores them on 10 held-out tasks; with `--uri` the MLOps path (§5) |
| `eval_policies.py` | Offline comparison of prewarm policies (adaptive at its two ends / top-k / lookahead) on the held-out steps |

**The five predictors**

| `PREDICTOR` | Inputs | Model |
|---|---|---|
| `lightgbm` | trajectory + signals | LightGBM over (workflow, last two tools, signals); without signals it answers with markov |
| `lightgbm_adapt` | trajectory + signals | lightgbm + pulse-v3's online adapter: `softmax(log P_lightgbm + W·φ)`, φ = signals, last tool, bias; W starts at 0 and learns from each episode once it is idle for `SESSION_TIMEOUT_S` (`ADAPT_LR` 0.05). In memory: a restart starts again from lightgbm |
| `markov` | trajectory | Order-2 Markov chain over actions with back-off to the last tool, then to the action frequencies |
| `speculator` | trajectory + history | An LLM reads the last 10 messages and lists likely next actions with probabilities (`pulse.speculator`) |
| `pulse_v3` | trajectory + history | `pulse.pulse_v3`: TF-IDF + sentence embedding of the conversation, a tool graph from the schemas (which IDs are known), state flags, an online adapter per agent model |

On 11 live episodes (gpt-oss-120b, tasks no model trained on; each episode scored before the adapter learned
from it) the adapter lifted lightgbm's top-1 from 64% to 71% (airline) and 39% to 57% (retail), but most of the
retail gain came from one episode that repeated the same calls; ADAPT_LR 0.2 overfitted (retail 36%).

**Adding a predictor:** subclass `predictors.base.Predictor` in `predictors/<name>.py` (set `name`, `inputs`,
`domains`; implement `predict()` returning `{action: probability}`), and add it to `REGISTRY` in
`predictors/__init__.py`. The service, the policy, the hint senders (via `/info`), the UI and the cluster
deployment pick it up by name.

### 3.3 `pulse.pulse_v3`: the pulse-v3 model

`predictors/pulse_v3/` holds the model (`models.py`), features (`features.py`), the tool graph from tool schemas
(`graph.py`), environment-state flags (`state.py`), sentence embeddings from the llama.cpp server with a disk
cache (`embed.py`), the online adapter (`online.py`) and `predictor.py`, which turns a session's hints into
predictions. `contract/` defines its session events (`Hint`, `End`) and `Prediction`; `traces/tau.py` turns
tau-bench conversations into episodes; `tools.py` is the tool catalog (domain-prefixed names). `train.py`
retrains it (§5). The service-side adapter is `pulse/predictor/predictors/pulse_v3.py`: it replays each
episode's hints into a pulse-v3 session.

### 3.4 `pulse.speculator`: the LLM speculator

`prompt.py` (the prompt: tools, last N messages, the answer format) and `speculate.py` (the LLM key pool and
calls) serve the `speculator` predictor. `steps.py`, `speculate.py`'s command line and `evaluate.py` are its
offline evaluation (Speculative Actions baseline) against lightgbm/markov on the same held-out steps.

### 3.5 `pulse.controller`

`prewarm_controller.py`: watches `PrewarmRequest`s (`infra/prewarm/prewarmrequest-crd.yaml`), pings each tool k
times concurrently through the ingress, records `Warming → Ready → Evicted/Failed` in `.status`, logs one JSON
line per action and serves Prometheus metrics (port 9102). Runs on the laptop (the UI starts it) or as the
`pulse-controller` Deployment.

Two modes (`CONTROLLER_MODE`):

| | `ping` (default, "Option A") | `lease` (admission control) |
|---|---|---|
| Per request | k warm-up pings: the tool gets k pods | only the NEW pods (k minus the tool's pods) that fit in the tool nodes' free pod slots next to a reserve (`LEASE_RESERVE_PODS`, 4): all, fewer, or none |
| Nodes full | prewarm pods take the slots real calls then wait for | a warm tool just gets its window refreshed; a cold one is `Rejected` and its first real call starts it |
| TTL | the tools' 60 s Knative window, from the last call or ping | the same window; the lease (`spec.ttlSeconds`, from the predictor: 20 s next step, 75 s lookahead, clamped 5–120 s) adds one extending ping when it outlives the window |

**How many pods (k), both modes** (`leases.rule_k`): the predictor's fan-out k (lightgbm's pending-ID rule) is cut to
the smallest number of pods whose queueing fits a budget, `min(k, ceil(k × warm call time / K_BUDGET_S))`. A tool pod
serves one call at a time and Knative's activator queues the rest on the first pod to free, so for the ~45 ms tau-bench
tools one pod serves a burst of up to 6 calls within 0.3 s: k = 1. Slow tools (`TOOL_WARM_S_JSON`, e.g. 0.5 s) keep k.
In the calibrated simulator (`results/sim_runs/*_ksweep-*`) this matched the best fixed policy both ways: with fast
calls 1 pod had the same latency as the predicted or even the true fan-out with room and ~20 % fewer pod-seconds, and
removed the ping controller's tail at 15 pods (p99 72 s → 0.50 s); with 0.5 s calls k > 1 lowered p95 (1.87 → 1.61 s)
and the true fan-out did best (1.42 s). `K_RULE=0` turns it off. With k = 1 the lease controller's reserve of 4 turned
away prewarms that would have fit at 15 pods (p99 1.68 s vs the ping controller's 0.50 s): a smaller reserve may suit.

**How many tools a step warms: adaptive top-k** (replaces the θ = 0.3 threshold; `PREWARM_POLICY=adaptive`, the
default; `topk` = a fixed `TOP_K` for comparisons). The predictor offers its top 3 tools with p ≥ 0.05 as candidates,
each with its rank and a hit rate: how often a candidate of that probability really was the next call, learned online
from each episode's next hint (`pulse/predictor/calibration.py`; it starts at p and corrects an over- or
under-confident predictor). The controller warms a candidate only if `hit rate × the tool's cold start ≥ (1 − hit
rate) × the price of a pod slot` (only a wrong prewarm wastes a slot; a right one starts the pod the call would have
started anyway): 0.075 s with room (a 1.5 s tool from a 5 % chance) up to 10× that, 33 %, when the nodes are full or pods wait
(`TOPK_PRICE_S`, `TOPK_FULL_X`; the cold start is learned from the controller's pings). So a step warms up to 3 tools
with room and 1 or none when full; refused ones get `Skipped` and a `skipped` log line (`why: topk`). Sim:
`--prewarm-policy adaptive|topk`. Not yet compared with θ in the simulator or on EKS: with room it warms more tools
than θ did (more pod-seconds).

**The k rule's numbers at runtime** (`K_ADAPT=1`, `adaptive.py`; off = the fixed 0.05 s and 0.3 s). The warm call
time per tool becomes the median of its warm calls in Prometheus (`pulse_tool_call_seconds{temperature="warm"}`,
which the runners export; `PROM_URL`, polled every 60 s; the settings until a tool has 5 warm calls, and throughout
without monitoring). The budget becomes `K_BUDGET_S × cold factor × (1 + fullness)`: the cold factor is the tool's
cold start, learned from the controller's own pings to tools at zero pods, over 1.5 s (clamped 0.5–3), and fullness is
1 − free slots / 6 (0 with room). So with room and normal cold starts nothing changes; a full node or a slow-starting
tool raises the budget and gets fewer pods. For today's ~45 ms tools k stays 1; it matters for slow tools. Sim:
`--k-policy adaptive`. The `warmed` log line records the `k_warm_s` and `k_budget_s` it used.

**How long prewarmed pods live, ping mode** (`refresh.py`; every lever off by default = the behaviour above). A pod
lives until its tool's Knative window passes with no request, and any request restarts it, a prewarm ping included: a
wrong prediction for a warm tool keeps an unused pod alive another window, holding a slot real calls may then wait for.

| Lever | Setting | What it does |
|---|---|---|
| refresh policy | `REFRESH_POLICY` always \| never \| value | always: every prediction pings. never: a tool with its k pods is not pinged, only real calls extend them. value: pinged only if p ≥ a threshold, 0 with `REFRESH_FREE_HIGH` (6) or more free slots, rising to `REFRESH_P_MAX` (0.8) with none |
| pressure shedding | `SHED_P` (0 = off) | while tool pods wait for a slot, predictions below it neither start nor refresh pods, so low-value pods age out |
| per-tool windows | `TOOL_WINDOWS=per-tool` (run_cloud.sh; UI: Idle window) | each tool's window = the 80th percentile of its re-use gaps in C0 runs, clamped to 20–60 s (over 60 s → 20 s: keeping it rarely pays). `python -m pulse.experiments.tool_windows <runs> --domain …` writes `data/tool_windows.json`; the shipped one is from the 19 Nov EKS C0 runs at 8 episodes/min (tools with under 5 re-uses are rough) |

Skipped predictions get `status.phase: Skipped` and a `skipped` log line. There is no real eviction: Knative's
autoscaler sets a tool's pod count from the traffic in its window, so a deleted idle pod is recreated at once; a pod
is freed only by sending it no traffic, which is what shedding and shorter windows do. The simulator models all three
(`sim_capacity --refresh always never value --shed-p 0 0.6 --windows 60 per-tool`). Not measured on EKS yet; the
in-cluster controller needs a rebuilt image for them.

EKS with the k rule (19 Nov 2026, third cluster, same runs; `*_eks3-*`, `*_eks3p-*`): predictions of 2–5 pods all
became 1. With room: ping p99 1.51 s, cold 2.4 %; lease 1.47 s, 3.2 %. Under pressure (one worker, 16/min) C0 / ping /
lease: p99 2.15 / **0.58** / 1.86 s, max 85 / **5.9** / 87 s, cold 5.0 / **1.8** / 3.4 %, peak Pending 3 / 1 / 3. The
ping controller's p99 under pressure went from 24–46 s (first two clusters, predicted k) to 0.58 s: the extra fan-out
pods were its whole problem. The lease controller saw 0 free slots most of the run (13 tool pods filled the worker),
so its reserve of 4 left it refreshing warm tools only and rejecting 9 cold ones: about C0's behavior. With the k rule
the ping controller is the better default; the lease controller needs a smaller reserve to add anything here.

Both run over the same 60 s window, and neither ever removes a warm pod: on EKS a warm pod that already exists was
the cheapest one to use, and shortening the window to make leases evictable (the first lease design: 6 s window,
keep-alive pings, value-based eviction) made the nodes restart pods all the time. The controller counts free pod
slots on the schedulable tool nodes (`TOOL_NODES`, default `pulse/role=worker`; cordoned nodes count as full):
allocatable pods and CPU minus what runs there and minus the tool pods waiting for a slot, converting CPU at
`TOOL_CPU_M` (75m: the tool's 50m plus Knative's 25m queue-proxy sidecar). The PrewarmRequest's status records each
decision (`grantedReplicas`, `leaseExpiresAt`, `decision`, phase `Rejected` / `Expired`). The policy is pure logic in
`leases.py`, shared by the controller and the simulator. The Experiments tab picks the controller (ping / lease,
plus the reserve); `run_cloud.sh` takes `CONTROLLER=lease`. Every run sets the tools' window to 60 s.

EKS (18 Nov 2026, 60 tau-bench episodes replayed in the cluster, lightgbm θ 0.3; `results/cloud_runs/*_eks-*`,
`*_eksp-*`): with room (both workers, 8 episodes/min) C0 / ping / the first lease design had p99 1.74 / 1.46 / 1.43 s
and cold share 7.5 / 3.0 / 3.2 %. Under pressure (one worker cordoned, ~15 tool pods of room, 16 episodes/min):
p99 1.85 / 24 / 156 s, cold share 4.6 / 3.4 / 47 %, tool pods started 21 / 27 / 214. Prewarm pods crowding out real
calls is what hurt the ping controller; constant pod restarts under the 6 s window what hurt the first lease design.
The admission-control design on EKS (19 Nov 2026, a new cluster, same runs; `*_eks2-*`, `*_eks2p-*`): with room
(8 episodes/min) p99 1.52 s, cold share 4.2 %, every lease admitted. Under pressure (one worker, 16 episodes/min)
C0 / ping / lease: p99 33.8 / 45.9 / **2.09 s**, max 87 / 123 / 60 s, cold share 5.2 / 3.8 / 3.2 %, tool pods started
26 / 30 / 23, pod-seconds 4,726 / 5,015 / 4,166, peak Pending pods 4 / 9 / 2. The controller saw 0–1 free slots
most of the run, lowered k for 130 predictions, rejected 9 and started only 21 new pods. C0 varied between the two
clusters (p99 1.85 s on 18 Nov, 33.8 s here): one run per configuration, so the gaps are indicative, not tight.

`pulse/experiments/sim_capacity.py` compares C0, C1, C3-ping and C3-lease on the simulated cluster with a capacity
of tool pod slots, on a virtual clock (no cluster, Docker or LLM): tool-call latency p50/p95/p99, cold share (calls
over 1 s, the runners' rule), calls that waited for a slot, pod-seconds. The simulator (`autoscaler="kpa"`) models
Knative's KPA: tools sized every 2 s by average concurrency over 60 s at 0.7 per pod, panic mode for 6 s bursts,
scale-down at most halving, a call queued on the first pod to free (the activator), a scaled-down pod holding its
slot for the 30 s grace, cold starts scaled to the in-cluster median (1.56 s; `data/coldstart/` measures from the
laptop). Against the six EKS runs it matches C0 (room: p99 1.80 vs 1.74 s, cold 7.7 vs 7.5 %; 15 pods: 1.75 vs 1.85 s,
4.6 vs 4.6 %) and the ping controller with room (1.59 vs 1.46 s); it overstates the ping controller's tail under
pressure about 3x (p99 72 vs 24 s) but gets its direction. C1's always-warm pods ignore the capacity (a shortcut).

Admission-control lease controller, simulated (`results/sim_runs/*_v2-*`, lightgbm θ 0.3, reserve 4): with room it
matches ping (16/min: p99 1.08 vs 1.01 s, cold 1.6 vs 1.2 %, 6 % fewer pod-seconds); at 15 pods p99 1.69 s vs 72 s
(ping) and 1.75 s (C0), cold 3.2 % (lowest), 24 % fewer pod-seconds than ping; at 24 episodes/min and 15 pods
1.82 vs 59.6 s (ping). At 10 pods, less room than the traffic needs, every configuration queues for minutes (lease
241 s, C0 274 s, ping 273 s). Reserve 2 lowers cold share (2.6 %) but lets a few calls wait (max 75 s); 8 removes the
waits at 3.6 % cold. EKS (above) agreed on the ranking under pressure: lease far below ping and C0. The adaptive
part of k (capping it at each tool's observed fan-out width)
needs to see calls, so only the simulator has it; the real controller lowers k only for room.

### 3.6 `pulse.experiments`: experiment runners

| Module | Role |
|---|---|
| `replay.py` | Replays recorded episodes (`data/traces/`, or tau-bench's) against the tools with seeded think / customer-reply gaps, fan-out groups fired in parallel, and one hint per LLM step (C3). Reproducible with a seed |
| `live.py` | The live agent with an LLM customer on the real tools |
| `common.py` | Shared by both: the pod watcher (cold/warm by pod start), the Redis reset, `run_config.json`, the result folder |
| `metrics.py` | Prometheus metrics of a run (port 9105) |
| `sim_capacity.py` | Capacity simulation: the same episodes on the simulated cluster with a pod limit, C0 / C1 / C3-ping / C3-lease (§3.5) |
| `plot_compare.py`, `plot_c0.py` | Comparison plots across runs |
| `score_hints.py`, `fill_trace_args.py` | Scoring recorded hints; adding arguments to traces |

A run folder holds `calls.csv`, `pods.csv`, `episodes.jsonl`, `hints.jsonl` (C3), `run_config.json` and
`summary.json` (latency percentiles, cold share, pod-seconds).

**Domains.** `DOMAIN=airline` (default, workload W1) or `DOMAIN=retail` (W3). One variable switches every piece:
the tools' backend and read-only set (`agent/tools.py`), the agent's prompt (the domain's tau-bench policy,
`agent/prompts.py`) and tasks, the tool pods the runners watch (label
`pulse/role=mcp-tool` / `mcp-tool-retail`), the Redis database reset between runs (DB 0: reservations, users; DB 1:
orders, users), the replay's traces (retail: built from tau-bench's recorded retail conversations,
`replay.tau_traces()`), the hints' `domain` and `workflow_id`, the action predictor's model
(`models/action_model_retail.pkl`) and the PrewarmRequests' service names (`retail-<tool>`). The retail tools are
Knative services `retail-<tool>` reached with `MCP_HOST_TEMPLATE=retail-{svc}.default.example.com`. The UI sets all
of it from the top bar's domain switch. Run pods (`infra/cloud/run_cloud.sh DOMAIN=retail`) get the same: the domain,
`MCP_URL_TEMPLATE=http://retail-{svc}.default.svc.cluster.local/mcp`, and the cluster's predictor is switched to the
domain too. The UI's Live demo serves both domains in one process, so it passes the domain explicitly
(`tools.package(domain)`, `read_only(domain)`, `load_tasks(domain)`, `prompts.agent_system_prompt(domain)`,
`LocalToolBackend(domain)` / `MCPToolBackend(domain)`). The live runner on retail defaults to the held-out retail
tasks. The speculator and the tau-bench "Replay recorded" demo stay airline; the simulated cluster is airline.

### 3.7 `pulse.loadtest`

Locust (closed loop: `locust_predictor.py`, `locust_tools.py`, `locust_episodes.py`, with load stages in
`locust_common.py`) and asyncio (open loop: `async_load.py`, exact request rates and cold bursts). `common.py`
records every request into `results/load_runs/`; `plot_load.py` plots latency, throughput and cold share against
load.

### 3.8 `pulse.ui`: the demo UI

| Module | Role |
|---|---|
| `server.py` | FastAPI: the page (`static/`), `/api/state` (polled twice a second), and endpoints for each tab; proxies Grafana under `/grafana/` |
| `backend.py` | The cluster (simulated or the current kubectl context), the predictor / controller processes it starts, the job runner (`BatchRun`: steps that are commands or Python callables, output streamed), experiment and load-test commands, results and plots, the model registry, monitoring and GitOps status |
| `setup.py` | The Setup tab: `status()` (tools on PATH, data, models, keys, clusters, AWS account), `ACTIONS` and `build()` (the steps of each button: scripts in `scripts/` and `infra/`), `save_env()` |
| `session.py` | One interactive episode (live agent or replay) with its events |
| `sim_cluster.py` | The simulated cluster: pods, cold starts from `data/coldstart/`, the idle window, prewarms |

The UI only starts processes and scripts that also work on their own; it adds no logic of its own to them.

### 3.9 The rest

`pulse/registry.py` is the S3 model registry (§5). `pulse/compat.py` loads models pickled under older module
names. `pulse/setup/` gets a fresh clone ready: `download_data.py` (tau-bench data and tool schemas) and
`train_models.py` (local training).

---

## 4. Infrastructure (`infra/`)

| Folder | Contents |
|---|---|
| `knative/` | The tools' image (`Dockerfile.mcp`), `redis.yaml` (the shared database), `deploy.sh` (Redis + the tools as Knative Services, min-scale 0 or 1; `DOMAIN=retail` for the retail tools), `tools.yaml` / `tools-retail.yaml` (what `gen_services.py` last generated) |
| `prewarm/` | The `PrewarmRequest` CRD: the contract between predictor (writes `.spec`) and controller (writes `.status`) |
| `kind/` | Local cluster: kind + Knative + Kourier (NodePort on localhost:31080), `teardown.sh` |
| `eks/` | `cluster.yaml` (eksctl: 2 worker nodes for the tools and Knative, 1 system node for the predictor, controller and `pulse-embed`, 1 client node for run pods, monitoring and training; system and client are tainted so no tool lands there), `eks_up.sh` (cluster, Knative + Kourier with an AWS load balancer open to your IP only, the tools, the CRD, a cold/warm check), `eks_down.sh`, `setup_knative.sh`, `registry_access.sh` (on an eksctl cluster: the cluster's OIDC provider and the IRSA roles `pulse-eks-pulse` / `pulse-eks-pulse-trainer`, the same names `infra/terraform/eks/irsa.tf` gives the Terraform cluster), `use_eks.sh` / `use_local.sh` (point a terminal's kubectl and tool URLs at EKS or back at kind) |
| `cloud/` | The runner image (`Dockerfile`, `cloud_entry.sh`, `build_image.sh` → ECR or kind), `pulse-cloud.yaml` (RBAC, `pulse-predictor` with `PREDICTOR=…`, `pulse-embed`, `pulse-controller`), `cloud_up.sh` (deploy + the LLM-key Secret), `run_cloud.sh` (a run as a pod; results copied back), `kustomization.yaml` + `gitops/` (the same manifests for Argo CD, and the model pins in `config.yaml`) |
| `monitoring/` | kube-prometheus-stack values (`values-kind.yaml` also scrapes the predictor, controller and runner on the laptop), PodMonitors for Pulse's components and Knative, the Grafana dashboards (`dashboards/build.py` generates them), `install.sh` (also turns on Knative's own metrics) |
| `gitops/` | Argo CD: `apps.yaml` (pulse-system, pulse-crds, monitoring, monitoring-config), `install.sh`, `configure.sh` (writes the AWS account into the manifests) |
| `training/` | Retraining as Kubernetes Jobs: pulse-v3 (with an embedding sidecar) and lightgbm + markov; `run_training.sh` |
| `terraform/` | Three roots: `bootstrap/` (the state bucket), `.` (S3 model registry with versioning, encryption and TLS-only access, reader/publisher IAM policies, ECR, GitHub OIDC roles for CI), `eks/` (the same cluster as eksctl, as code, with IRSA roles) |

**Images.** `pulse-runner` (`infra/cloud/Dockerfile`) runs the predictor, controller, run pods and training Jobs:
the package, `models/action_model.pkl` (and `action_model_retail.pkl` if trained), a locally trained pulse-v3 model
if present, the airline and retail trajectories and the replay traces. The tools' image is built from
`infra/knative/Dockerfile.mcp`; `deploy.sh` defaults to a published copy (airline only). *Deploy the retail tools*
(`infra/knative/deploy_retail.sh`) builds it here: on kind it loads it into the nodes as
`kind.local/pulse-mcp-tools:retail` (a registry Knative does not resolve), on EKS it pushes it to the account's ECR
repository `pulse-tools`; then `DOMAIN=retail deploy.sh`.

**Cloud concepts used:** serverless scale-to-zero (Knative), a custom resource with a controller (CRD +
reconcile loop), containers and registries (Docker, ECR), managed Kubernetes (EKS) and its networking (load
balancer restricted by IP, VPC), infrastructure as code (Terraform with remote state, eksctl), object storage with
versioning and lifecycle (S3), identity without stored keys (IRSA, GitHub OIDC), CI/CD and GitOps (GitHub Actions,
Argo CD), observability (Prometheus, Grafana), load testing (Locust, asyncio), and MLOps (§5).

---

## 5. Models and MLOps

| Family | Predictors | File | Trained by | Release gate |
|---|---|---|---|---|
| `action` | lightgbm, markov | `action_model.pkl` | `pulse.predictor.train_action` | lightgbm's held-out top-1 |
| `pulse_v3` | pulse_v3 | `predictor.pkl` | `pulse.pulse_v3.train` | held-out warm@3 (the share of next calls that would find a warm pod with the top 3 prewarmed) |

- **Locally** (no AWS): `python -m pulse.setup.train_models action|action_retail|pulse_v3` writes `models/`.
  `action_retail` trains the retail lightgbm + markov on the LLM speculator's split (92 tasks, 23 held out:
  `data/predictor/retail_split.json`); it is not in the registry.
- **The S3 registry** (`pulse/registry.py`, bucket from `infra/terraform/model_store.tf`): one prefix per family
  (`pulse-v3/`, `action-model/`), one folder per version (model + `manifest.json` with sha256, scikit-learn
  version, metrics), `latest.json`, and the training data under `_data/` (keyed like `data/`).
- **Retraining** (`--uri`, the Models tab, or a Job via `infra/training/run_training.sh`): pull the data, train,
  score the new model and the current *latest* on the same held-out tasks, publish a new version; it becomes
  *latest* only if it beats the current one (`--promote always|never` overrides).
- **Serving** loads `latest` (or a pinned version: `PULSE_MODEL_VERSION` / `ACTION_MODEL_VERSION`, which GitOps
  sets in `infra/cloud/gitops/config.yaml`). If the registry cannot be read, lightgbm/markov fall back to the
  model in the image and say so on `/healthz`.

---

## 6. CI/CD and GitOps

```
pull request ─▶ ci.yml: install, compile, load the main predictor · Terraform fmt/validate · kustomize build
               terraform.yml: terraform plan (read-only, OIDC role)
merge to main ─▶ image.yml: build pulse-runner → ECR (OIDC role) → write its tag into infra/cloud/kustomization.yaml
                                                                     │
Argo CD (in the cluster) ◀────────────── watches main ───────────────┘ → applies infra/cloud, the CRD, monitoring
```

Argo CD does not manage the tools (experiments switch their min-scale at runtime) or the LLM-key Secret
(`cloud_up.sh` creates it). A model version or the cluster's predictor changes in production by a pull request to
`infra/cloud/gitops/`.

**Setting up the AWS side once (per account; all optional: EKS runs work without it).**
1. **Terraform state bucket:** Setup → Cloud extras → *1. Terraform state bucket* (`infra/terraform/bootstrap`:
   creates `pulse-tfstate-<account>`; skipped if it exists). The other roots (`infra/terraform/`, and
   `infra/terraform/eks/` when the cluster is made with Terraform) keep their state there.
2. **Registry, ECR, CI roles:** *2. Terraform plan*, then *3. Terraform apply* (`infra/terraform`, state in that
   bucket). With your fork in the form (`-var github_repo=<owner>/<repo>`) it also creates the two GitHub OIDC roles.
3. **Use the registry:** *Terraform apply* ends by saving its outputs `model_uri` / `action_model_uri` as
   `PULSE_MODEL_URI` / `ACTION_MODEL_URI` in `.env` (also *Fill in from Terraform*, or type them in *Model
   registry*). The UI uses them at once and passes them to everything it starts; set in the environment that
   starts the UI, they work too. Then *Upload training data* and *Publish local model* (or *Retrain* in the
   Models tab). On EKS, the cluster's pods reach the registry through IRSA roles: *Create cluster* sets them up when
   the registry exists (`infra/eks/registry_access.sh`); for a cluster made earlier, *AWS EKS → Model registry
   access*. Without them the predictors use the models in the image and retraining Jobs cannot publish.
4. **CI on your fork:** add the repository variables `AWS_IMAGE_ROLE_ARN` and `AWS_PLAN_ROLE_ARN` (output
   `ci_role_arns`) and `TF_STATE_BUCKET`. Until they exist, `image.yml` and `terraform.yml` are skipped, not failed.
5. **GitOps:** Setup → *GitOps: set my AWS account* (`infra/gitops/configure.sh <account>`), commit and push, then
   *Install Argo CD* on the EKS cluster (it follows `main`, or the branch in the form).

---

## 7. Settings and ports

| Setting | Used by |
|---|---|
| `.env`: `LLM_BACKEND`, `GROQ_API_KEY[_N]`, `GROQ_MODEL`, `OPEN_ROUTER_API_KEY`, `OLLAMA_*`, `USER_MODEL`, `SPEC_*` | the agent, the customer, the speculator |
| `PREDICTOR`, `PREWARM_POLICY`, `TOP_K`, `TOP_K_MAX`, `P_MIN`, `LOOKAHEAD`, `THETA_NEXT`, `TTL_S`, `CRD_MODE`, `AUDIT_LOG` | the prediction service |
| `PULSE_MODEL_URI` / `PULSE_MODEL_VERSION` / `PULSE_MODEL`, `PULSE_EMBED_BASE` | pulse_v3 |
| `ACTION_MODEL_URI` / `ACTION_MODEL_VERSION` / `ACTION_MODEL_PATH` | lightgbm, markov |
| `PREDICTOR_URL`, `TOOL_BACKEND`, `MCP_URL_TEMPLATE`, `MCP_HOST_TEMPLATE`, `KUBE_CONTEXT` | the agent and the runners |
| `DOMAIN` (`airline` / `retail`) | the tools, the runners, the predictor (§3.6) |
| `REGION`, `TARGET`, `USE_TERRAFORM`, `USE_GITOPS`, `TF_STATE_BUCKET` | the cluster and image scripts |
| `UI_PORT`, `UI_PREDICTOR_PORT`, `UI_PREDICTOR` / `UI_AUTOSTART_PREDICTOR` (the predictor it starts with), `UI_GRAFANA_PORT`, `UI_ARGOCD_PORT`, `UI_STATE_DIR` | the UI |

Each module lists its other, rarer settings at the top of its file.

| Port | Service |
|---|---|
| 8765 | the UI |
| 8000 | the predictor |
| 31080 | Kourier on kind (the tools) |
| 8091 | the embedding server |
| 9102 / 9105 | Prometheus metrics of the controller / of a run |
| 13000 / 18080 / 8089 | local forwards to Grafana / Argo CD, Locust's web UI |
