# Pulse: project documentation

DS252 Introduction to Cloud Computing, IISc (Aug 2026).

This document explains the whole system: what problem it solves, how the parts fit together, which
cloud-computing concepts each part shows, and where everything is in the repository. To run the
project, see **[RUNNING.md](RUNNING.md)**.

## 1. The problem

An LLM agent works in steps: it reasons, calls a tool, reads the result, and repeats. When each tool
is a serverless function that scales to zero, the first call to an idle tool waits for a **cold
start**: Knative has to start a pod, about 1.5–2.5 s on EKS and 3–6 s on a laptop. An episode makes
many calls, so these waits add up.

**Pulse** predicts the agent's next tool call and wakes that tool's pod before the call arrives. The
agent's think time and the customer's reply time become lead time for the prewarm.

The workload is tau-bench's airline domain: 14 MCP tools such as `get_user_details`,
`get_reservation_details` and `update_reservation_flights`, used by a ReAct agent and an
LLM-simulated customer.

## 2. Architecture

```
                         ┌──────────────────────────── Kubernetes (kind or AWS EKS) ───────────────────────────┐
 agent / replay / UI     │                                                                                     │
 ──hint (trajectory,──▶  predictor  ──P ≥ θ──▶  PrewarmRequest (CRD)  ◀──watch──  prewarm controller            │
    signals, history)    (gbt | lr | n-gram |                                          │ MCP ping                 │
                         speculator | pulse_v3)                                         ▼                          │
 ──tool call (MCP)──────────────────────────────────────────────────▶  Kourier ──▶  Knative tool pods (14, scale-to-zero)
                         │                                                          shared Redis (airline DB)     │
                         │  Prometheus + Grafana · Argo CD · training Job · UI pod (EKS)                         │
                         └─────────────────────────────────────────────────────────────────────────────────────┘
 AWS: S3 (model registry, Terraform state) · ECR (runner image) · IAM (IRSA, GitHub OIDC) · EKS · VPC
 GitHub: Actions (CI, image build, terraform plan)
```

**Request path:** each step runs through these stages in order.
1. **Hint:** before every LLM step the agent sends a **hint** to the predictor. The hint carries the
   tools called so far, numeric signals about the conversation, and optionally recent messages.
2. **Prediction:** the predictor ranks the next action (a tool, RESPOND or END). Each tool with
   probability ≥ θ, or the top-k under the top-k policy, becomes a `PrewarmRequest` object. The
   request also carries a fan-out width k, the number of parallel calls expected.
3. **Prewarm:** the **prewarm controller** watches `PrewarmRequest`s and sends the tool one MCP ping
   through the ingress, or k pings to get k pods. Knative's activator starts the pod.
4. **Call:** when the agent's call arrives, the pod is warm, or at least already starting.
5. **Scale down:** Knative's 60 s idle window is the TTL; after it the tool scales back to zero. The
   controller marks the request `Evicted`.

**Configurations compared:**

| Config | Meaning |
|---|---|
| C0 | Scale-to-zero, no prewarming: the cold-start baseline |
| C1 | Always warm (min-scale 1): the latency floor and resource ceiling |
| C3 | Scale-to-zero, with predictor and controller: Pulse |

**Targets (PRD):** relative to C0, C3 should have a p50 at least 20% lower and a p99 at least 15%
lower. Relative to C1, it should use at least 30% fewer warm-pod-seconds. The predictor's p95 should
be under 50 ms at 10 hints/s.

## 3. Components

### 3.1 Agent and tools (`agent/`)
- **Agent:** a custom ReAct agent (`agent.py`, LangGraph variant `langgraph_agent.py`). LLMs come
  from Groq, OpenRouter or Ollama.
- **Simulated customer:** `user.py`.
- **Tools:** the 14 airline tools (`tau_airline/`). Each is its own MCP server (`mcp_server.py`,
  streamable HTTP, stateless, JSON responses) and its own Knative Service.
- **Database:** one shared Redis copy of the airline data.
- **Hints:** sent by `hints.py`, never blocking the agent. If the predictor is slow or down, the
  call simply happens cold (the reactive fallback).

### 3.2 Predictors (all serve the same `POST /predict` API)

| Predictor | Where | Idea |
|---|---|---|
| `gbt` (main) | `predictor/` | LightGBM next-action model over the trajectory plus agent-side signals (numbers and flags only, no raw text), with a fan-out width k. Siblings: `lr_action`, `ngram_action`, `ensemble` |
| LLM speculator | `baselines/llm_speculator/` | An LLM (Groq gpt-oss-20b) reads the recent conversation and guesses the next action: the speculative-actions baseline |
| `pulse_v3` | `pulse_pred/` | Reads the conversation text (TF-IDF plus a sentence embedding), a tool graph derived from the tool schemas (which IDs are known, so which tools are ready), and environment-state flags. An online adapter learns per agent model. `pulse_pred/service.py` replays the hints into a per-episode session |

Because they share one API, the replay runner, the UI and the controller work with any of them.
Prewarm policies are θ-threshold or top-k, optionally with lookahead.

### 3.3 Prewarm controller (`controller/prewarm_controller.py`, `deploy/prewarm/`)
- **Contract:** the `PrewarmRequest` CRD (`pulse.iisc.ac.in/v1alpha1`). Predictors write `.spec`;
  the controller writes `.status`.
- **Status phases:** Warming → Ready → Evicted (or Failed).
- **Action log:** each action is a JSON line in `controller.jsonl`.
- **Prometheus metrics** on port 9102: warm-ups (cold or warm), pings, ping latency, failures,
  evictions and warm duration.

### 3.4 Experiment runners (`test/test_knative/`)
- **`w1_replay.py`:** replays recorded tau-bench traces with seeded think and customer-reply gaps.
  The same seed gives the same requests on C0, C1 and C3. `--fanout` fires independent lookups
  together, and `--metrics-port` publishes live metrics.
- **`w1_live.py`:** the real LLM agent and simulated customer.
- **Output:** each run writes `results/<local|cloud>_runs/<date>_<config>_W1-<mode>_…/` with
  `summary.json`, `calls.csv`, `pods.csv`, `episodes.jsonl`, `run_config.json`, and for C3 also
  `hints.jsonl` and `controller.jsonl`.
- **Plots:** `plot_compare.py` draws latency CDFs, percentiles, pod-seconds and cold starts.

### 3.5 Demo UI (`ui/`)
A browser UI at http://127.0.0.1:8765. It runs on the laptop, or inside EKS as the `pulse-ui` pod.

| Tab / area | What |
|---|---|
| Live demo | Live or replayed episodes on a **simulated** or a real **Knative** cluster. Shows the predictor's guesses, prewarms and cold/warm calls, plus a C0-vs-C3 race; optional background load |
| Experiments | Replay, live and **load test** runs, with the log streamed and a summary |
| Results | Every run folder, Compare, Make plots, zip download, and a Grafana link per run |
| Monitoring | Embedded Grafana dashboards that follow the running run or episode |
| Models | pulse-v3 registry versions with held-out metrics: Use, Promote, Pin in Git, Retrain |
| Side panel / top bar | Predictor card (model, θ, p95), tool pods and pod-seconds, prewarm activity, monitoring and GitOps status |

### 3.6 Monitoring (`deploy/monitoring/`)
- **Stack:** kube-prometheus-stack (Prometheus, Grafana, kube-state-metrics, node-exporter). On EKS
  it runs on the client node, away from the tools.
- **Scraped:** the predictors, the controller, the replay runner's tool-call metrics, the tool pods,
  the nodes and Knative.
- **Dashboards as code** (`dashboards/build.py`):
  - **Pulse overview:** pods per tool, warm pod-seconds, cold share, tool latency, prewarms.
  - **Predictor:** p95 against 50 ms, hints/s, prewarm candidates.
  - **Cluster:** node CPU and memory, container CPU.

### 3.7 Load testing (`loadtest/`)
- **Locust (closed loop):** the users ramp up in stages, and each waits for its answer before sending
  the next request.
  - **predictor API:** NFR-4;
  - **tool services:** read-only MCP calls;
  - **whole agent episodes:** C0 vs C3 under concurrency.
- **asyncio (open loop):** requests go out on a fixed schedule whether or not earlier ones were
  answered: exact hints/s, or bursts of cold starts.
- **Shared data:** all tests use the replay's recorded episodes.
- **Output:** results in `results/load_runs/`, plotted by `plot_load.py`.

### 3.8 Model registry and retraining (`pulse_pred/`, `deploy/training/`)
- **Registry:** a versioned S3 bucket with one folder per model version (`predictor.pkl` +
  `manifest.json` with sha256, scikit-learn version and held-out metrics). `latest.json` points at
  the current version, and the training data lives under `_data/`.
- **Retraining (`train.py`):** pulls the data, splits by task (train / val / held-out test), trains,
  then evaluates the new model and the current `latest` on the same held-out hints.
- **Release gate:** the new version becomes `latest` only if its **warm@3** is higher. warm@3 is the
  share of next-step calls that would find a warm pod if the top 3 tools were prewarmed.
- **Where it runs:** locally, or as a **Kubernetes Job** with an embedding-server sidecar and an IRSA
  role.

### 3.9 Infrastructure as code (`terraform/`)

| Root | Manages |
|---|---|
| `bootstrap/` | The S3 bucket for Terraform's remote state |
| `.` (registry) | S3 model registry and its IAM policies, the ECR `pulse-runner` repository, the GitHub OIDC CI roles |
| `eks/` | The EKS cluster (2 workers + 1 tainted client node, m7i-flex.large), its VPC, and IRSA roles. Opt-in: `USE_TERRAFORM=1 eks_up.sh`; eksctl is the default |

### 3.10 CI/CD and GitOps (`.github/workflows/`, `deploy/gitops/`)
- **`ci.yml`:** on PRs and pushes to main: tests, `terraform fmt` and `validate`, kustomize builds.
- **`image.yml`:** on merge to main: builds the runner image, pushes it to ECR with an OIDC role (no
  stored keys), and commits the new tag into `EKS_deployment/cloud/kustomization.yaml`.
- **`terraform.yml`:** a read-only `terraform plan` on PRs that touch `terraform/`.
- **Argo CD** makes the cluster follow Git: Pulse's services (`EKS_deployment/cloud`, kustomize), the
  CRD and monitoring. The Knative tools stay script-managed, because experiments switch their
  min-scale at runtime.

## 4. Cloud computing concepts

| Concept | Where in Pulse |
|---|---|
| Serverless / FaaS, scale-to-zero, cold starts | Knative Serving for the 14 MCP tools; the activator holds requests during a cold start |
| Container orchestration | Kubernetes (kind locally, AWS EKS), node roles, taints and tolerations, Deployments, Jobs |
| Extending Kubernetes (CRD + controller / operator pattern) | `PrewarmRequest` CRD and the prewarm controller reconciling it |
| Autoscaling | Knative's per-request autoscaling (containerConcurrency 1, idle window as TTL); predictive prewarming on top |
| Managed cloud services | EKS, S3, ECR, IAM, VPC, Elastic Load Balancing (Kourier) |
| Infrastructure as code | Terraform: three roots, remote state with locking, modules per lifecycle |
| Identity and least privilege | IRSA roles per service account; GitHub OIDC federation with no stored keys; reader / publisher policies |
| Object storage, versioning, data lifecycle | S3 model registry with versioning, encryption, TLS-only access, and expiry of old versions |
| CI/CD | GitHub Actions: tests, image build to ECR, `terraform plan` on PRs |
| GitOps | Argo CD applications reconciling the cluster to Git; image tags committed by CI |
| Observability | Prometheus metrics (FR-11), Grafana dashboards as code, audit logs (FR-12) |
| Performance engineering / load testing | Locust (closed loop) and asyncio (open loop), SLO checks (NFR-4) |
| MLOps | Model registry, reproducible training, held-out evaluation gate, promotion and rollback, online adaptation |
| Cost awareness | Scale-to-zero vs always-warm pod-seconds (NFR-7); no NAT gateway; teardown checklists; Free Tier instance type |
| Reproducible experiments | Seeded replay: identical request sequences across C0, C1 and C3; three trials per config |

## 5. Repository map

| Path | What |
|---|---|
| `agent/` | ReAct agent, LLM client, simulated customer, MCP tool servers, hints, airline tools and data |
| `predictor/` | Main predictor (gbt and others): data, training, evaluation, API, CRD writer, load test |
| `baselines/llm_speculator/` | LLM speculator baseline |
| `pulse_pred/` | pulse-v3 predictor, S3 model store, retraining (`train.py`) |
| `controller/` | Prewarm controller |
| `deploy/knative/` | Knative services for the tools, Redis, `deploy.sh`, cold/warm check |
| `deploy/prewarm/` | `PrewarmRequest` CRD and the predictor ↔ controller contract |
| `deploy/monitoring/` | Prometheus + Grafana values, PodMonitors, dashboards, `install.sh` |
| `deploy/training/` | Retraining Job and `run_training.sh` |
| `deploy/gitops/` | Argo CD install, applications, account configuration |
| `EKS_deployment/` | EKS bring-up and teardown (eksctl or Terraform), in-cluster runner image and services (`cloud/`) |
| `terraform/` | Infrastructure as code (bootstrap, registry, EKS) |
| `.github/workflows/` | CI, image build, Terraform plan |
| `loadtest/` | Locust and asyncio load tests, `plot_load.py` |
| `test/test_knative/` | kind setup, C0/C1/C3 runners, plots, metrics |
| `test/test_cold_start/` | The OpenFaaS tests that led to Knative |
| `ui/` | Demo UI (`server.py`, `backend.py`, `session.py`, `sim_cluster.py`, `static/`) |
| `results/` | Run results: `cloud_runs/`, `local_runs/`, `load_runs/` |
| `docs/` | PRD, Knative decision, architecture diagram, this documentation |

## 6. Key design decisions

- **Knative over OpenFaaS CE:** Knative really scales to zero, and its activator holds a request
  during a cold start, so a missed prediction still succeeds. See `docs/Knative_decision.docx`.
- **Prewarm by MCP ping:** changing min-scale creates a new Revision. A ping wakes the tool through
  the same path the agent's calls take, and Knative's idle window then acts as the TTL.
- **CRD + controller:** the predictor only writes the desired state; the controller reconciles it.
  The two are decoupled and each can be replaced.
- **One `/predict` API for every predictor:** baselines and new models plug in without touching the
  controller, the runners or the UI.
- **Lightweight models first:** scikit-learn and LightGBM before deep models. This keeps the
  prewarming effect separate from model complexity, and keeps the predictor's p95 well under 50 ms.
- **Seeded replay for comparisons:** identical requests across configurations, without spending LLM
  quota.
- **Separation by lifecycle:** each piece's lifetime decides where it lives.
  - **Long-lived** (registry, ECR, CI roles): their own Terraform state.
  - **Per-session** (the EKS cluster): its own state, created and destroyed each session.
  - **Out of git:** models live in S3, so they can be retrained without code changes.
- **Monitoring and load generators on the client node:** they never compete with a cold start for
  CPU. On EKS, cold starts are CPU-bound.

## 7. Status and limitations (Nov 2026)

- **Results:** the C0/C1/C3 experiments with gbt and the speculator have run on EKS (in
  `results/cloud_runs/`). Only workload W1 (change or cancel a booking) has been run so far.
- **Not yet run end to end:** the additions on branch `iac-gitops`, each written and statically
  checked:
  - pulse-v3 integration, the S3 registry (applied, model published), monitoring, load tests,
    retraining, the Terraform EKS cluster, CI and Argo CD;
  - see `PLAN.md` for each phase's status and deviations.
- **pulse-v3 and the embedding server:** pulse-v3 needs an embedding server for full accuracy, and it
  isn't deployed in-cluster as its own service yet. It runs as the UI's predictor.
- **Retraining data:** retraining uses tau-bench trajectories. Turning the team's own run traces into
  training data is a follow-up.
- **The UI pod's S3 access** is read-only: promote and retrain from the laptop UI or as the Job.

## 8. Further reading

| Topic | Document |
|---|---|
| Running everything | [RUNNING.md](RUNNING.md) |
| Experiments step by step | [../RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md) |
| UI guide (recipes 1–10) | [../ui/GUIDE.md](../ui/GUIDE.md) |
| Predictor ↔ controller contract | [../deploy/prewarm/README.md](../deploy/prewarm/README.md) |
| pulse-v3, model registry, retraining | [../pulse_pred/README.md](../pulse_pred/README.md) |
| Monitoring | [../deploy/monitoring/README.md](../deploy/monitoring/README.md) |
| Load testing | [../loadtest/README.md](../loadtest/README.md) |
| Terraform | [../terraform/README.md](../terraform/README.md) |
| CI/CD and GitOps | [../deploy/gitops/README.md](../deploy/gitops/README.md) |
| Roadmap and phase status | [../PLAN.md](../PLAN.md) |
| Original requirements | `PRD.docx` (this folder) |
