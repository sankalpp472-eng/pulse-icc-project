# Pulse: developer documentation

What each part of the code does and how the parts fit together. For installing and using the project, see
[README.md](README.md). Every module and script also explains itself (settings, commands) at the top of its file.

---

## 1. Architecture

```
                    ┌─────────────────────────── Kubernetes: kind (laptop) or AWS EKS ─────────────────────────┐
 agent / replay     │                                                                                          │
 runner / UI ──hint─┼─▶ predictor ──P ≥ θ──▶ PrewarmRequest (CRD) ◀──watch── prewarm controller                │
   (pulse.agent,    │   (pulse.predictor)                                            │ MCP ping (k times)       │
    .experiments)   │                                                                ▼                          │
            ──tool call (MCP over HTTP)──▶ Kourier ingress ──▶ Knative Services: 14 (16 retail) tools, to zero  │
                    │                                            (one pod per call in flight; shared Redis DB)  │
                    │   embedding server (pulse_v3) · Prometheus + Grafana · Argo CD · training Jobs           │
                    └──────────────────────────────────────────────────────────────────────────────────────────┘
 AWS: S3 (model registry, Terraform state) · ECR (images) · IAM (IRSA, GitHub OIDC) · EKS · VPC     GitHub Actions: CI
```

**One step of an episode:**
1. **Hint.** Before every LLM step the agent POSTs a hint to the predictor: the tools called so far, plus whatever
   that predictor reads (numeric signals, or the recent messages). Sent on a background thread; the agent never
   waits for it.
2. **Prediction.** The predictor ranks the next actions (each tool, `RESPOND` = answer the customer, `END`). The
   prewarm policy picks tools (P ≥ θ, or the top k) and a fan-out width k per tool (the number of parallel calls
   expected), and writes one `PrewarmRequest` object per tool.
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

### 3.2 `pulse.predictor`: the prediction service and its four predictors

| Module | Role |
|---|---|
| `contract.py` | The one contract: `PredictRequest` (trajectory; optional `signals`, `history`, `domain`, `workflow_id`) → `PredictionOut` (`ranked` actions with probabilities, `prewarm` tools with k, `lookahead`, latency, inputs used) |
| `service.py` | FastAPI app: `POST /predict`, `GET /info` (the predictor's inputs and capabilities), `/healthz`, `/metrics`. Loads the predictor named by `PREDICTOR` and applies the shared policy, audit log and PrewarmRequest writer |
| `predictors/` | One file per predictor behind `base.Predictor` (`inputs`, `predict()`, `fanout()`, `lookahead()`, `health()`); `action.py` loads the pickle lightgbm and markov share (local file or the registry); `__init__.py` is the registry (`PREDICTOR=lightgbm|markov|speculator|pulse_v3`) |
| `policy.py` | Which ranked actions become prewarms: θ threshold or top-k with a floor; never `RESPOND`/`END` |
| `crd_writer.py` | Writes `PrewarmRequest` objects (`CRD_MODE=k8s`), or to a file (`dryrun`), on a background thread |
| `client.py` | What hint senders share: `inputs(url)` from `/info`, `make_hint()`, `compact()` messages |
| `signals.py` | Numeric conversation signals (step, pending IDs, last error, customer keywords, …): no raw text leaves the agent for lightgbm |
| `action_model.py` | The lightgbm (`GBTModel`) and markov (`NGramModel`) models, the pending-ID fan-out rule and the lookahead model, in one `ActionPredictor` (pickled as `models/action_model.pkl`) |
| `build_action_data.py` | Training rows from the tau-bench conversations (one row per LLM step) → `data/predictor/action_rows.csv` |
| `train_action.py` | Trains lightgbm + markov (+ lookahead), scores them on 10 held-out tasks; with `--uri` the MLOps path (§5) |
| `eval_policies.py` | Offline comparison of prewarm policies (θ / top-k / lookahead) on the held-out steps |

**The four predictors**

| `PREDICTOR` | Inputs | Model |
|---|---|---|
| `lightgbm` | trajectory + signals | LightGBM over (workflow, last two tools, signals); without signals it answers with markov |
| `markov` | trajectory | Order-2 Markov chain over actions with back-off to the last tool, then to the action frequencies |
| `speculator` | trajectory + history | An LLM reads the last 10 messages and lists likely next actions with probabilities (`pulse.speculator`) |
| `pulse_v3` | trajectory + history | `pulse.pulse_v3`: TF-IDF + sentence embedding of the conversation, a tool graph from the schemas (which IDs are known), state flags, an online adapter per agent model |

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

### 3.6 `pulse.experiments`: experiment runners

| Module | Role |
|---|---|
| `replay.py` | Replays recorded episodes (`data/traces/`, or tau-bench's) against the tools with seeded think / customer-reply gaps, fan-out groups fired in parallel, and one hint per LLM step (C3). Reproducible with a seed |
| `live.py` | The live agent with an LLM customer on the real tools |
| `common.py` | Shared by both: the pod watcher (cold/warm by pod start), the Redis reset, `run_config.json`, the result folder |
| `metrics.py` | Prometheus metrics of a run (port 9105) |
| `plot_compare.py`, `plot_c0.py` | Comparison plots across runs |
| `score_hints.py`, `fill_trace_args.py` | Scoring recorded hints; adding arguments to traces |

A run folder holds `calls.csv`, `pods.csv`, `episodes.jsonl`, `hints.jsonl` (C3), `run_config.json` and
`summary.json` (latency percentiles, cold share, pod-seconds).

**Domains.** `DOMAIN=airline` (default, workload W1) or `DOMAIN=retail` (W3). One variable switches every piece:
the tools' backend and read-only set (`agent/tools.py`), the tool pods the runners watch (label
`pulse/role=mcp-tool` / `mcp-tool-retail`), the Redis database reset between runs (DB 0: reservations, users; DB 1:
orders, users), the replay's traces (retail: built from tau-bench's recorded retail conversations,
`replay.tau_traces()`), the hints' `domain` and `workflow_id`, the action predictor's model
(`models/action_model_retail.pkl`) and the PrewarmRequests' service names (`retail-<tool>`). The retail tools are
Knative services `retail-<tool>` reached with `MCP_HOST_TEMPLATE=retail-{svc}.default.example.com`. The UI sets all
of it from the top bar's domain switch; the live agent, the speculator and the in-cluster runner stay airline.

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
| `eks/` | `cluster.yaml` (eksctl: 2 worker nodes + 1 client node), `eks_up.sh` (cluster, Knative + Kourier with an AWS load balancer open to your IP only, the tools, the CRD, a cold/warm check), `eks_down.sh`, `setup_knative.sh`, `use_eks.sh` / `use_local.sh` (point a terminal's kubectl and tool URLs at EKS or back at kind) |
| `cloud/` | The runner image (`Dockerfile`, `cloud_entry.sh`, `build_image.sh` → ECR or kind), `pulse-cloud.yaml` (RBAC, `pulse-predictor` with `PREDICTOR=…`, `pulse-embed`, `pulse-controller`), `cloud_up.sh` (deploy + the LLM-key Secret), `run_cloud.sh` (a run as a pod; results copied back), `kustomization.yaml` + `gitops/` (the same manifests for Argo CD, and the model pins in `config.yaml`) |
| `monitoring/` | kube-prometheus-stack values (`values-kind.yaml` also scrapes the predictor, controller and runner on the laptop), PodMonitors for Pulse's components and Knative, the Grafana dashboards (`dashboards/build.py` generates them), `install.sh` (also turns on Knative's own metrics) |
| `gitops/` | Argo CD: `apps.yaml` (pulse-system, pulse-crds, monitoring, monitoring-config), `install.sh`, `configure.sh` (writes the AWS account into the manifests) |
| `training/` | Retraining as Kubernetes Jobs: pulse-v3 (with an embedding sidecar) and lightgbm + markov; `run_training.sh` |
| `terraform/` | Three roots: `bootstrap/` (the state bucket), `.` (S3 model registry with versioning, encryption and TLS-only access, reader/publisher IAM policies, ECR, GitHub OIDC roles for CI), `eks/` (the same cluster as eksctl, as code, with IRSA roles) |

**Images.** `pulse-runner` (`infra/cloud/Dockerfile`) runs the predictor, controller, run pods and training Jobs:
the package, `models/action_model.pkl`, a locally trained pulse-v3 model if present, the airline trajectories and
replay traces. The tools' image is built from `infra/knative/Dockerfile.mcp`; `deploy.sh` defaults to a published
copy (airline only). The UI's *Deploy the retail tools* builds it locally, loads it into kind as
`kind.local/pulse-mcp-tools:retail` (a registry Knative does not resolve) and runs `DOMAIN=retail deploy.sh`.

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
1. **Terraform state bucket:** `terraform -chdir=infra/terraform/bootstrap init && terraform -chdir=infra/terraform/bootstrap apply`
   creates `pulse-tfstate-<account>`. The other roots (`infra/terraform/`, and `infra/terraform/eks/` when the
   cluster is made with Terraform) keep their state there.
2. **Registry, ECR, CI roles:** Setup → Cloud extras → *Terraform plan*, then *Terraform apply* (or
   `terraform -chdir=infra/terraform init -backend-config="bucket=pulse-tfstate-<account>"` and `apply`). With
   your fork in the form (`-var github_repo=<owner>/<repo>`) it also creates the two GitHub OIDC roles.
3. **Use the registry:** restart the UI with `PULSE_MODEL_URI` and `ACTION_MODEL_URI` set to
   `terraform -chdir=infra/terraform output -raw model_uri` / `action_model_uri`; then *Upload training data* and
   *Publish local model* (or *Retrain* in the Models tab).
4. **CI on your fork:** add the repository variables `AWS_IMAGE_ROLE_ARN` and `AWS_PLAN_ROLE_ARN` (output
   `ci_role_arns`) and `TF_STATE_BUCKET`. Until they exist, `image.yml` and `terraform.yml` are skipped, not failed.
5. **GitOps:** Setup → *GitOps: set my AWS account* (`infra/gitops/configure.sh <account>`), commit and push, then
   *Install Argo CD* on the EKS cluster (it follows `main`, or the branch in the form).

---

## 7. Settings and ports

| Setting | Used by |
|---|---|
| `.env`: `LLM_BACKEND`, `GROQ_API_KEY[_N]`, `GROQ_MODEL`, `OPEN_ROUTER_API_KEY`, `OLLAMA_*`, `USER_MODEL`, `SPEC_*` | the agent, the customer, the speculator |
| `PREDICTOR`, `THETA`, `PREWARM_POLICY`, `TOP_K`, `P_MIN`, `LOOKAHEAD`, `THETA_NEXT`, `TTL_S`, `CRD_MODE`, `AUDIT_LOG` | the prediction service |
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
