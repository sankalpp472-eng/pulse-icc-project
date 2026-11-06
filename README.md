# Pulse: predictive prewarming for serverless MCP tools

DS252 Introduction to Cloud Computing, IISc (Aug 2026).

> **Documentation:** **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)** covers the architecture,
> components, cloud concepts and design decisions. **[docs/RUNNING.md](docs/RUNNING.md)** is the
> step-by-step guide to running everything: the demo UI, kind, AWS EKS, pulse-v3 and retraining,
> and IaC / CI / GitOps.

An LLM agent calls tools one step after another. When those tools are serverless and scaled to zero,
every first call waits for a cold start of 1–5 s, and the waits add up over an episode. Pulse
predicts the agent's next tool call and wakes that tool's pod before the call arrives.

```
agent ──hint──▶ predictor (POST /predict) ──P ≥ θ──▶ PrewarmRequest (CRD) ◀──watches── prewarm controller ──MCP ping──▶ Knative tool pod
```

- **Agent** (`agent/`): a ReAct agent on tau-bench's airline domain with an LLM-simulated customer.
  It sends a hint to the predictor before every LLM step.
- **Predictor:** ranks the next action. The tools above the threshold θ become `PrewarmRequest` objects.
- **Prewarm controller** (`controller/`): watches the `PrewarmRequest`s and sends each tool one MCP
  ping, so Knative starts its pod. Knative's 60 s idle window is the TTL.
- **Tools:** the 14 airline tools, each its own MCP server and Knative Service, scaled to zero, on
  kind (laptop) or AWS EKS.

Configurations compared:

| Config | Meaning |
|---|---|
| C0 | Scale-to-zero, no prewarming (the cold-start baseline) |
| C1 | Always warm, min-scale 1 per tool (the latency floor and resource ceiling) |
| C3 | Scale-to-zero, with predictor and prewarm controller (Pulse) |

The design and its decisions are described in [CLAUDE.md](CLAUDE.md) and `docs/` (PRD, Knative decision).

## Prerequisites

Commands are for **Git Bash** (Windows) or a Linux / macOS shell, run from the repository root.
After cloning, one command sets everything up:

```bash
bash scripts/setup.sh        # Python 3.11+ venv with all requirements, tau-bench data, agent/.env from the example
export PY="$PWD/agent/.venv/Scripts/python"            # in each new terminal (Linux/macOS: agent/.venv/bin/python)
```

| What | Needed for |
|---|---|
| `scripts/setup.sh` (above) | everything |
| LLM keys in `agent/.env` (`GROQ_API_KEY=...`, or a local server via `SPEC_BASE_URL`). Never commit it | live runs, the AI customer, the speculator predictor |
| The pulse-v3 model: `bash scripts/embed_server.sh` (Docker), then `"$PY" scripts/train_models.py pulse_v3` | the pulse_v3 predictor (lightgbm and markov ship trained) |
| Docker Desktop, kubectl, kind | the local cluster |
| AWS CLI (logged in), eksctl | EKS |
| Terraform ≥ 1.6, AWS credentials | `terraform/`, the S3 model registry |

All data and models can be rebuilt from the clone: `scripts/download_data.py` (public tau-bench data)
and `scripts/train_models.py action|pulse_v3|all`. Details: [docs/RUNNING.md](docs/RUNNING.md), section 1.

## 1. Quick start: the demo UI, no cluster (free)

```bash
"$PY" ui/server.py                  # then open http://127.0.0.1:8765
```

- The UI starts on a **simulated cluster**: it simulates pods, cold starts measured on EKS and
  prewarming, with real tool results.
- Pick a task, **Replay recorded**, and run it once with **None (C0)** and once with
  **Predictor (C3)**, then compare the cold calls.
- The UI loads the LightGBM predictor at launch, which takes about a minute.
- **Full guide:** [ui/GUIDE.md](ui/GUIDE.md).

## 2. Local Knative cluster (kind)

```bash
bash test/test_knative/setup.sh                       # kind cluster + Knative + Kourier
SKIP_BUILD=1 bash deploy/knative/deploy.sh 0          # Redis + 14 tools, min-scale 0
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
"$PY" deploy/knative/check_tools.py                   # expect 14/14 tools OK
```

**C0 replay:**
```bash
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --label kind
```

**C3 replay**, using three terminals:
```bash
KUBE_CONTEXT=kind-pulse-kn "$PY" controller/prewarm_controller.py                                   # 1: controller
CRD_MODE=k8s KUBE_CONTEXT=kind-pulse-kn PREDICTOR=lightgbm "$PY" -m uvicorn service:app --app-dir predictor --port 8000   # 2: predictor
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --predictor-url http://127.0.0.1:8000 --label kind-lightgbm   # 3: replay
```

Results go to `results/local_runs/`. For C1, live runs and troubleshooting, see
[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md), Path B.

## 3. AWS EKS (the reported results)

Creating the cluster costs money until it's torn down, so always finish with `eks_down.sh`.

```bash
bash EKS_deployment/cloud/build_image.sh               # runner image -> ECR (~5 min)
PYTHON="$PY" bash EKS_deployment/eks_up.sh             # cluster, Knative, tools (~25 min)
bash EKS_deployment/cloud/cloud_up.sh                  # predictor and controller pods
CONFIG=C0 bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=lightgbm bash EKS_deployment/cloud/run_cloud.sh --fanout   # or markov / speculator / pulse_v3
bash EKS_deployment/eks_down.sh                        # tear down (~10 min)
```

Results go to `results/cloud_runs/`.
- **Running the UI inside the cluster:** `bash EKS_deployment/cloud/ui_cloud.sh up`.
- **Every setting, trials, plots and teardown checks:** [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md),
  Path A.

## Predictors

One service, four predictors: `PREDICTOR=<name> uvicorn service:app --app-dir predictor`.
Every predictor takes a `PredictRequest` and returns ranked next actions plus the tools to prewarm
([predictor/contract.py](predictor/contract.py)). `GET /info` says which inputs it reads. The agent, the replay
and the UI read `/info` and send only those inputs, so the conversation text leaves the agent only for
predictors that need it.

| `PREDICTOR` | Reads | What it is |
|---|---|---|
| `lightgbm` (main, default) | trajectory + signals | LightGBM next action from the last two tools and agent-side signals (numbers only, no raw text), plus fan-out k and lookahead. Without signals it answers with `markov` |
| `markov` | trajectory | Order-2 Markov chain over actions with back-off, P(next \| last two tools); lookahead |
| `speculator` | trajectory + history | An LLM reads the last 10 messages (Groq keys, or any OpenAI-compatible server via `SPEC_BASE_URL`) |
| `pulse_v3` | trajectory + history | Conversation text, a tool graph from the schemas, state flags, and an online adapter. See [pulse_pred/README.md](pulse_pred/README.md) |

The code lives in [predictor/predictors/](predictor/predictors/), one file per predictor. `lightgbm` and
`markov` share `predictor/data/action_model.pkl` (`train_action.py`).
Common settings: `THETA` (0.3), `PREWARM_POLICY` theta | topk, `TOP_K`, `P_MIN`, `LOOKAHEAD`, `TTL_S`, and
`CRD_MODE` off | dryrun | k8s. In the UI, pick the predictor on the Predictor card. On EKS, one
`pulse-predictor` Deployment runs whichever predictor `PREDICTOR` names (`run_cloud.sh` switches it).

### pulse-v3 and its model on S3

The trained pulse-v3 model is not in git. It lives in a versioned S3 bucket created by Terraform,
and the service downloads it at start:

```bash
"$PY" -m pip install -r pulse_pred/requirements.txt              # incl. boto3, scikit-learn 1.9.1
export PULSE_MODEL_URI=s3://pulse-models-<account id>/pulse-v3   # = terraform -chdir=terraform output -raw model_uri
"$PY" ui/server.py                                               # the UI passes it on to the predictor it starts
```

- **Retraining (MLOps):** `python pulse_pred/train.py --uri "$PULSE_MODEL_URI"` (or the Kubernetes Job
  `deploy/training/run_training.sh`). It trains, evaluates on held-out tasks, and publishes a new
  version, which becomes `latest` only if its warm@3 beats the current model's. In the UI, the
  **Models** tab lists versions and their metrics, and has **Use**, **Promote** (rollback) and
  **Retrain**. See [pulse_pred/README.md](pulse_pred/README.md#retraining-mlops).
- **lightgbm + markov** have the same pipeline for their shared `action_model.pkl`
  (`ACTION_MODEL_URI=s3://.../action-model`, `predictor/train_action.py --uri`, gate on lightgbm's
  held-out top-1, `MODEL=action deploy/training/run_training.sh`, Models tab → lightgbm + markov). See
  [docs/RUNNING.md](docs/RUNNING.md) section 5.4.
- **Embedding server:** pulse-v3 is more accurate with its local embedding server
  (llama.cpp + nomic-embed-text on port 8091). Without it, it runs with a zero embedding.

## Monitoring (Prometheus + Grafana)

```bash
bash deploy/monitoring/install.sh      # into the current kubectl context (kind or EKS), needs helm
```

- **What it is:** kube-prometheus-stack with three dashboards kept as code: Pulse overview, Predictor
  and Cluster.
- **What is scraped:** the predictors (`/metrics`), the prewarm controller, the replay runner's live
  tool-call metrics, the tool pods (kube-state-metrics), the nodes and Knative.
- **In the UI** (Knative mode):
  - a monitoring status, **Install** button and **Grafana ↗** link in the top bar;
  - a **Monitoring** tab whose time window follows the running run or episode;
  - Prometheus figures on the Predictor and Tool pods cards;
  - a **Grafana** link per run in Results.
- **On EKS** it runs on the client node, away from the tools.
- **Details:** [deploy/monitoring/README.md](deploy/monitoring/README.md).

## Load testing (Locust + asyncio)

```bash
"$PY" -m pip install -r loadtest/requirements.txt
"$PY" -m locust -f loadtest/locust_predictor.py --headless --host http://127.0.0.1:8000   # NFR-4 under rising load
"$PY" loadtest/async_load.py tools --burst 1 5 10 --pause 90                               # cold-start storms
```

There are three Locust tests and an asyncio open-loop generator, all built on the replay's recorded
episodes:
- **predictor API:** hints with history, so any predictor can be tested;
- **tool services:** read-only MCP calls;
- **whole agent episodes:** with hints, so C0 can be compared with C3 under concurrency.

Results go to `results/load_runs/`, and `loadtest/plot_load.py` plots them.
- **In the UI:** Experiments → **Load test**, load runs in Results (Compare, Make plots, Grafana), and
  **Background load** in the Live demo.
- **Details:** [loadtest/README.md](loadtest/README.md).

## Infrastructure as code

`terraform/` has three roots, each with its own state:
- `bootstrap/`: the remote-state bucket;
- the long-lived **registry** (`terraform/`): the S3 model bucket, ECR, and the CI roles for GitHub
  Actions through OIDC;
- the **EKS cluster** (`terraform/eks/`): the same nodes as `cluster.yaml`, with IRSA roles so pods
  get AWS access without keys.

```bash
USE_TERRAFORM=1 bash EKS_deployment/eks_up.sh      # the cluster from Terraform (eksctl remains the default)
```

The setup order is in [terraform/README.md](terraform/README.md).

## CI/CD and GitOps

- **On pull requests:** GitHub Actions (`.github/workflows/`) run the tests, `terraform
  fmt/validate`, the kustomize builds, and a read-only `terraform plan`.
- **On merge to main:** the workflow builds the runner image, pushes it to ECR, and writes its tag
  into `EKS_deployment/cloud/kustomization.yaml`.
- **In the cluster:** Argo CD makes it follow Git: Pulse's services, the CRD and monitoring.

```bash
USE_GITOPS=1 bash EKS_deployment/eks_up.sh         # or: bash deploy/gitops/install.sh
```

In the UI, the top bar shows the running commit or image, the **GitOps n/m synced** status and an
**Argo CD ↗** link, and the Models tab can **Pin in Git**. See
[deploy/gitops/README.md](deploy/gitops/README.md).

## Results

Each run writes one folder:
`results/<cloud|local>_runs/<date>_<time>_<config>_W1-<replay|live>_<details>_<label>/`.
- **Contents:** `summary.json`, `calls.csv`, `pods.csv`, `episodes.jsonl`, `run_config.json`, and
  for C3 also `hints.jsonl` and `controller.jsonl`.
- **Plots:** `test/test_knative/plot_compare.py <run folders> --out <folder>` draws latency CDFs,
  percentiles, pod-seconds and cold starts. The UI's Results tab does the same.

## Repository map

| Path | What |
|---|---|
| `agent/` | ReAct agent, LLM client, simulated customer, MCP tool servers, hints, airline tools and data |
| `predictor/` | The prediction service and its four predictors (`predictors/`), contract, prewarm policy, training data + training for lightgbm/markov, CRD writer |
| `baselines/llm_speculator/` | The speculator's prompt and LLM calls, plus its offline evaluation |
| `pulse_pred/` | pulse-v3 model code, its S3 model store and retraining |
| `scripts/` | Setup from a fresh clone: `setup.sh`, `download_data.py`, `train_models.py`, `embed_server.sh` |
| `controller/` | Prewarm controller |
| `deploy/knative/`, `deploy/prewarm/` | Knative services for the tools; the `PrewarmRequest` CRD and contract |
| `deploy/monitoring/` | Prometheus + Grafana: Helm values, scrape configs, dashboards as code, `install.sh` |
| `loadtest/` | Load tests: Locust (predictor, tools, episodes), asyncio open-loop generator, `plot_load.py` |
| `deploy/training/` | Retraining Jobs: pulse-v3 (with an embedding sidecar) and lightgbm + markov, `run_training.sh` |
| `EKS_deployment/` | EKS cluster scripts, in-cluster runner image and services |
| `terraform/` | Infrastructure as code: state bucket (`bootstrap/`), model registry + ECR + CI roles, EKS cluster (`eks/`) |
| `deploy/gitops/`, `.github/workflows/` | Argo CD applications and install script; CI (tests, image build, terraform plan) |
| `test/test_knative/` | kind setup, C0/C1/C3 runners, plots |
| `ui/` | Demo UI |
| `results/` | Run results (`cloud_runs/`, `local_runs/`) |
| `docs/` | PRD, Knative decision, architecture diagram |

## More documentation

- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md): the project documentation.
- [docs/RUNNING.md](docs/RUNNING.md): how to run everything, step by step.
- [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md): every experiment step by step, on EKS or kind.
- [ui/GUIDE.md](ui/GUIDE.md): the demo UI.
- [deploy/prewarm/README.md](deploy/prewarm/README.md): the predictor ↔ controller contract.
- [deploy/monitoring/README.md](deploy/monitoring/README.md): Prometheus and Grafana.
- [loadtest/README.md](loadtest/README.md): load testing.
- [deploy/gitops/README.md](deploy/gitops/README.md): CI/CD and GitOps.
- [EKS_deployment/README.md](EKS_deployment/README.md): the older laptop-driven EKS scripts.
- [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md): planned experiments.
- [pulse_pred/README.md](pulse_pred/README.md): the pulse-v3 predictor.
- [terraform/README.md](terraform/README.md): Terraform.
