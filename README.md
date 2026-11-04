# Pulse: predictive prewarming for serverless MCP tools

DS252 Introduction to Cloud Computing, IISc (Aug 2026).

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

Commands are for **Git Bash** (Windows), run from the repository root.

| What | Needed for |
|---|---|
| Python venv: `python -m venv agent/.venv`, then `agent/.venv/Scripts/python -m pip install -r agent/requirements.txt -r predictor/requirements.txt` | everything |
| `agent/.env` with `GROQ_API_KEY_1=...` (up to `_4`). Never commit it | live runs, the AI customer, the LLM speculator |
| tau-bench conversations in `predictor/data/tau-bench/historical_trajectories/` (`gpt-4o-airline.json`, `sonnet-35-new-airline.json`) | replays |
| Docker Desktop, kubectl, kind | the local cluster |
| AWS CLI (logged in), eksctl | EKS |
| Terraform ≥ 1.6, AWS credentials | `terraform/`, and the pulse-v3 model from S3 |

In each new terminal:

```bash
export PY="$PWD/agent/.venv/Scripts/python"
```

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
CRD_MODE=k8s KUBE_CONTEXT=kind-pulse-kn PREDICTOR_MODEL=gbt "$PY" -m uvicorn service:app --app-dir predictor --port 8000   # 2: predictor
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --predictor-url http://127.0.0.1:8000 --label kind-gbt   # 3: replay
```

Results go to `results/local_runs/`. For C1, live runs and troubleshooting, see
[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md), Path B.

## 3. AWS EKS (the reported results)

Creating the cluster costs money until it's torn down, so always finish with `eks_down.sh`.

```bash
bash EKS_deployment/cloud/build_image.sh               # runner image -> ECR (~5 min)
PYTHON="$PY" bash EKS_deployment/eks_up.sh             # cluster, Knative, tools (~25 min)
bash EKS_deployment/cloud/cloud_up.sh                  # predictor, speculator, controller pods
CONFIG=C0 bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=gbt bash EKS_deployment/cloud/run_cloud.sh --fanout
bash EKS_deployment/eks_down.sh                        # tear down (~10 min)
```

Results go to `results/cloud_runs/`.
- **Running the UI inside the cluster:** `bash EKS_deployment/cloud/ui_cloud.sh up`.
- **Every setting, trials, plots and teardown checks:** [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md),
  Path A.

## Predictors

All predictors serve the same `POST /predict` API, so the controller, replay and UI work with any of them.

| Predictor | Start (for C3) | What it uses |
|---|---|---|
| `gbt` (main) | `PREDICTOR_MODEL=gbt uvicorn service:app --app-dir predictor` | LightGBM next action from the trajectory and agent-side signals (no raw text), plus fan-out k |
| `lr_action`, `ngram_action`, `ensemble` | `PREDICTOR_MODEL=<name> ...` (same command) | Baselines; see `predictor/` |
| LLM speculator | `uvicorn service:app --app-dir baselines/llm_speculator` | An LLM reads the conversation (Groq); the replay needs `--hint-context` |
| `pulse_v3` | `uvicorn service:app --app-dir pulse_pred` | Conversation text, a tool graph from the schemas, state flags, and an online adapter; the replay needs `--hint-context`. See [pulse_pred/README.md](pulse_pred/README.md) |

Common settings: `THETA` (0.3), `PREWARM_POLICY` theta | topk, `TOP_K`, `P_MIN`, `TTL_S`, and
`CRD_MODE` off | dryrun | k8s. In the UI, pick the predictor on the Predictor card.

### pulse-v3 and its model on S3

The trained pulse-v3 model is not in git. It lives in a versioned S3 bucket created by Terraform,
and the service downloads it at start:

```bash
"$PY" -m pip install -r pulse_pred/requirements.txt              # incl. boto3, scikit-learn 1.9.1
export PULSE_MODEL_URI=s3://pulse-models-<account id>/pulse-v3   # = terraform -chdir=terraform output -raw model_uri
"$PY" ui/server.py                                               # the UI passes it on to the predictor it starts
```

- **Publishing a retrained model:** `python pulse_pred/model_store.py publish <file> "$PULSE_MODEL_URI" --version <v>`.
- **Embedding server:** pulse-v3 is more accurate with its local embedding server
  (llama.cpp + nomic-embed-text on port 8091). Without it, it runs with a zero embedding.

## Infrastructure as code

`terraform/` holds the AWS resources managed with Terraform. So far that's the model registry:
- an S3 bucket that is private, versioned and encrypted, and accepts TLS only;
- IAM policies for reading models and for publishing them.

```bash
cd terraform && terraform init && terraform plan && terraform apply
```

The cluster itself is still created by `eksctl` (`EKS_deployment/`). Planned next: EKS and ECR in
Terraform, remote state, CI with GitHub Actions, and GitOps with Argo CD. See
[terraform/README.md](terraform/README.md).

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
| `predictor/` | Main predictor: trace export, graphs, models, training, evaluation, API, CRD writer |
| `baselines/llm_speculator/` | LLM speculator baseline |
| `pulse_pred/` | pulse-v3 predictor and its S3 model store |
| `controller/` | Prewarm controller |
| `deploy/knative/`, `deploy/prewarm/` | Knative services for the tools; the `PrewarmRequest` CRD and contract |
| `EKS_deployment/` | EKS cluster scripts, in-cluster runner image and services |
| `terraform/` | Infrastructure as code (S3 model registry) |
| `test/test_knative/` | kind setup, C0/C1/C3 runners, plots |
| `ui/` | Demo UI |
| `results/` | Run results (`cloud_runs/`, `local_runs/`) |
| `docs/` | PRD, Knative decision, architecture diagram |

## More documentation

- [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md): every experiment step by step, on EKS or kind.
- [ui/GUIDE.md](ui/GUIDE.md): the demo UI.
- [deploy/prewarm/README.md](deploy/prewarm/README.md): the predictor ↔ controller contract.
- [EKS_deployment/README.md](EKS_deployment/README.md): the older laptop-driven EKS scripts.
- [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md): planned experiments.
- [pulse_pred/README.md](pulse_pred/README.md): the pulse-v3 predictor.
- [terraform/README.md](terraform/README.md): Terraform.
