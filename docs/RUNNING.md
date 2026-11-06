# Running Pulse: step by step

This guide takes you from a fresh clone to each way of running the project. What the parts are and
why they exist is in [DOCUMENTATION.md](DOCUMENTATION.md).

Run every command in **Git Bash** on Windows (or a Linux/macOS shell), **from the repository root**,
unless a step says otherwise.

## 0. Pick a path

| Path | What you get | Needs | Cost | Time |
|---|---|---|---|---|
| **A. Demo UI, simulated cluster** | The whole idea in the browser: live or replayed episodes, predictor, prewarms, C0 vs C3 | Python only | free | 5 min |
| **B. Local Knative cluster (kind)** | Real Knative tools, the controller, experiment runs, monitoring, load tests | Docker, kind, kubectl (+ helm) | free | 30 min |
| **C. AWS EKS** | The reported results, everything in the cloud, GitOps | AWS account, eksctl or Terraform | **EKS and EC2 bill by the hour** | 45 min + runs |
| **D. pulse-v3, model registry, retraining** | The pulse_v3 predictor from S3, the Models tab, retraining | AWS account (S3) | a few cents a month | 15 min |

Paths B–D build on the one-time setup in section 1.

## 1. One-time setup

### 1.1 Tools

| Tool | For | Check |
|---|---|---|
| Git + Git Bash | everything | `git --version` |
| Python 3.11+ | everything | `python --version` |
| Docker Desktop (give it ≥ 4 GB) | B, C | `docker info` |
| kubectl, kind | B, C | `kubectl version --client`, `kind version` |
| helm | monitoring, GitOps | `helm version` |
| AWS CLI v2, logged in | C, D | `aws sts get-caller-identity` |
| eksctl **or** Terraform ≥ 1.10 | C | `eksctl version` / `terraform version` |

Use an IAM user or IAM Identity Center login for the AWS CLI, **not the account's root user**.

### 1.2 Python environment and data (one command)

```bash
bash scripts/setup.sh                                 # venv + requirements + tau-bench data + agent/.env
export PY="$PWD/agent/.venv/Scripts/python"          # in every new terminal (Linux/macOS: agent/.venv/bin/python)
```

`scripts/setup.sh` does three things:
- It creates `agent/.venv` and installs every requirement.
- It runs `scripts/download_data.py`. That downloads tau-bench's public, MIT-licensed recorded conversations
  (airline and retail) into `predictor/data/tau-bench/` and `pulse_pred/data/tau_bench/`, and writes pulse-v3's
  tool schemas from `agent/tau_*/tools/`.
- It copies `agent/.env.example` to `agent/.env`.

The plots need a Python with matplotlib. Set `PLOT_PYTHON` if it isn't `C:\Python311\python.exe`.

### 1.3 Keys

- **LLM keys (live runs, the AI customer, the speculator predictor):** put them in `agent/.env`,
  either `GROQ_API_KEY=...` or `GROQ_API_KEY_1=...` up to `_4`. The speculator can also use a local
  OpenAI-compatible server instead (`SPEC_BASE_URL`, see the file). **Never commit `agent/.env`.**

### 1.4 Models

| Predictor | Model | From a fresh clone |
|---|---|---|
| `lightgbm`, `markov` | `predictor/data/action_model.pkl` | In git, ready. To reproduce it: `"$PY" scripts/train_models.py action` (~1 min) |
| `pulse_v3` | `pulse_pred/data/predictor.pkl` | Train it: `bash scripts/embed_server.sh` (Docker), then `"$PY" scripts/train_models.py pulse_v3` (~15–30 min). Or set `PULSE_MODEL_URI` to an S3 registry (section 5) |
| `speculator` | none (an LLM) | Needs keys or `SPEC_BASE_URL` in `agent/.env` |

`scripts/embed_server.sh` runs llama.cpp with nomic-embed-text on port 8091, the same image as the
cluster. pulse_v3 uses it for training and, for full accuracy, while serving too.

## 2. Path A: the demo UI on the simulated cluster

```bash
"$PY" ui/server.py                 # open http://127.0.0.1:8765
```

1. Wait until the **Predictor** card says *ready*. The LightGBM model loads in about a minute.
2. **Live demo** tab: Episode **Replay recorded**, a task (e.g. 30, *fan-out*), then **⏱ Run C0
   then C3**.
3. Compare the two lanes (end-to-end time, tool wait, cold calls) and the rows in **Episodes so far**.
4. Optionally, **Live agent** with **AI customer** or **I'll type** (needs the Groq keys).

The full UI guide is [ui/GUIDE.md](../ui/GUIDE.md).

## 3. Path B: local Knative cluster (kind)

### 3.1 Cluster and tools (once)

```bash
bash test/test_knative/setup.sh                   # kind cluster pulse-kn + Knative + Kourier (localhost:8081)
SKIP_BUILD=1 bash deploy/knative/deploy.sh 0      # Redis + 14 tools, min-scale 0
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
"$PY" deploy/knative/check_tools.py               # expect 14/14 tools OK
```

### 3.2 Monitoring (optional)

```bash
bash deploy/monitoring/install.sh                 # Prometheus + Grafana in namespace monitoring (~3 min)
```

This uses `host.docker.internal`, which Docker Desktop provides, to scrape the processes on your
laptop.

### 3.3 Run experiments from the UI

```bash
"$PY" ui/server.py
```

1. **Top bar:** switch to **Knative (kind-pulse-kn)**. With monitoring installed the pill says
   *monitoring on*. If not, **Install** runs 3.2 for you.
2. **Experiments** tab: Mode **Replay**, Configuration **C0**, Tasks *Held-out*, **▶ Start run**.
   Then the same with **C3**; the UI starts the predictor and the controller itself.
3. **Monitoring** tab: Grafana follows the running run.
4. **Results** tab: tick both runs, then **Compare selected** and **Make plots**.

### 3.4 Or from the command line

```bash
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --label kind                 # C0
# C3, three terminals:
KUBE_CONTEXT=kind-pulse-kn "$PY" controller/prewarm_controller.py
CRD_MODE=k8s KUBE_CONTEXT=kind-pulse-kn PREDICTOR=lightgbm "$PY" -m uvicorn service:app --app-dir predictor --port 8000
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --predictor-url http://127.0.0.1:8000 --label kind-lightgbm
```

`PREDICTOR` is one of `lightgbm` (main), `markov`, `speculator` or `pulse_v3`. The replay reads the
predictor's `/info` and sends the inputs it needs. For `speculator` and `pulse_v3` add `--hint-timeout 10`.

C1 (always warm): `SKIP_BUILD=1 bash deploy/knative/deploy.sh 1`, run the replay, then go back with
`... deploy.sh 0`. Results go to `results/local_runs/`. Every option is in
[RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md), Path B.

### 3.5 Load tests

**From the UI:** Experiments → Mode **Load test**. Pick a target (Predictor API / Tool services /
Whole agent episodes) and an engine (Locust / asyncio), then **▶ Start load test**.

**From the command line:**

```bash
"$PY" -m locust -f loadtest/locust_predictor.py --headless --host http://127.0.0.1:8000     # needs a predictor running
"$PY" -m locust -f loadtest/locust_tools.py --headless --stages "2:60,5:60,10:60"
"$PY" loadtest/async_load.py tools --burst 1 5 10 --pause 90
"$PY" loadtest/plot_load.py results/load_runs/<run> results/load_runs/<run> --out results/load_runs/compare_mine
```

The tests are described in [loadtest/README.md](../loadtest/README.md).

### 3.6 Clean up

```bash
bash test/test_knative/teardown.sh                # deletes the kind cluster
```

## 4. Path C: AWS EKS

Everything here **costs money until step 4.7**. Plan a session, and always tear down at the end.

### 4.1 Build the runner image (after every code change)

```bash
bash EKS_deployment/cloud/build_image.sh          # -> ECR repository pulse-runner (tag saved in EKS_deployment/cloud/.image-ecr)
```

### 4.2 Create the cluster (~25 min)

```bash
mkdir -p EKS_deployment/logs
PYTHON="$PY" bash EKS_deployment/eks_up.sh                        # with eksctl (default)
# or, as infrastructure as code (one-time setup in section 6 first):
USE_TERRAFORM=1 PYTHON="$PY" bash EKS_deployment/eks_up.sh
```

Either way you get 2 worker nodes, 1 client node, Knative + Kourier (open to your IP only), Redis,
the 14 tools at scale-to-zero, and the CRD. It ends with a cold/warm table: expect **14/14 tools
OK**.

### 4.3 Pulse's services

```bash
bash EKS_deployment/cloud/cloud_up.sh             # Secret from agent/.env, pulse-predictor (lightgbm), pulse-embed, pulse-controller
bash deploy/monitoring/install.sh                 # optional: Prometheus + Grafana on the client node
```

**GitOps instead of `cloud_up.sh` for the services:** run `bash deploy/gitops/install.sh` once the
GitOps setup in section 6 is done. `cloud_up.sh` is still needed once, for the Secret.

### 4.4 Experiment runs

```bash
CONFIG=C0                      bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C1                      bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=lightgbm   bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=markov     bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=speculator bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=pulse_v3   bash EKS_deployment/cloud/run_cloud.sh --fanout   # needs PULSE_MODEL_URI + IRSA
```

There is one `pulse-predictor` Deployment. `run_cloud.sh` switches it to the run's predictor
(`kubectl set env deploy/pulse-predictor PREDICTOR=...`). `pulse_v3` gets its sentence embeddings from
the `pulse-embed` Deployment (llama.cpp + nomic-embed-text, on the client node). Check it with
`kubectl logs deploy/pulse-embed`; the predictor's `/healthz` shows `"embedding": "ok"` once it is used.

Results go to `results/cloud_runs/`. For trials (three per configuration), live runs and plots, see
[RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md), Path A.

### 4.5 The UI inside the cluster

```bash
bash EKS_deployment/cloud/ui_cloud.sh up          # deploy + port-forward -> http://127.0.0.1:8765
```

It does everything the laptop UI does, next to the tools: experiments, load tests and monitoring
(Grafana comes through the same tunnel). Before teardown, copy its runs home with
`bash EKS_deployment/cloud/ui_cloud.sh fetch` (`eks_down.sh` also does this).

### 4.6 Retraining as a Job (optional)

See section 5.3.

### 4.7 Tear down (always, ~10 min)

```bash
bash EKS_deployment/eks_down.sh                   # eksctl cluster
USE_TERRAFORM=1 bash EKS_deployment/eks_down.sh   # Terraform cluster
```

Then check that nothing is left. Each command should print nothing:

```bash
R=ap-south-1
aws eks list-clusters --region $R --query clusters --output text
aws ec2 describe-instances --region $R --filters Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text
aws elb describe-load-balancers --region $R --query 'LoadBalancerDescriptions[].LoadBalancerName' --output text
aws elbv2 describe-load-balancers --region $R --query 'LoadBalancers[].LoadBalancerName' --output text
```

ECR, the S3 registry and the state bucket remain on purpose. They cost cents a month.

## 5. Path D: pulse-v3, the model registry and retraining

### 5.1 Registry (once per AWS account)

The registry root keeps its state in S3, so first create the state bucket and initialise the root:
section 6, steps 1–2 (leave out `-var github_repo` if you don't want the CI roles yet). Then:

```bash
terraform -chdir=terraform plan       # review: S3 bucket + 2 IAM policies, ECR (+ CI roles with -var github_repo)
terraform -chdir=terraform apply
export PULSE_MODEL_URI="$(terraform -chdir=terraform output -raw model_uri)"     # s3://pulse-models-<account>/pulse-v3
```

In the team's own AWS account this is already done. In your account, train pulse-v3 locally first
(section 1.4), then publish that file in 5.2.

### 5.2 Publish a model and use pulse_v3

```bash
"$PY" pulse_pred/model_store.py publish <predictor.pkl> "$PULSE_MODEL_URI" --version 2026-11-03 --notes "first model"
"$PY" pulse_pred/model_store.py list "$PULSE_MODEL_URI"
"$PY" ui/server.py                # with PULSE_MODEL_URI exported: Predictor card -> pulse-v3, or the Models tab -> Use
```

Or serve it directly:
`PULSE_MODEL_URI=... CRD_MODE=off "$PY" -m uvicorn service:app --app-dir pulse_pred --port 8000`.

**Embedding server:** pulse-v3 is more accurate with its embedding server. It's llama.cpp with
nomic-embed-text-v1.5 on port 8091; the command is in
[pulse_pred/README.md](../pulse_pred/README.md). Without it, `/healthz` says
`"embedding": "unavailable"`.

### 5.3 Retrain

The training data is tau-bench's historical trajectories for airline **and** retail, plus the tools'
`schemas/`. `scripts/download_data.py` (run by `setup.sh`) puts them in `pulse_pred/data/tau_bench/`.
`embed_cache.npz` is built by the first training run, with the embedding server running.

```bash
"$PY" pulse_pred/model_store.py data push "$PULSE_MODEL_URI"        # once: pulse_pred/data/{tau_bench,embed_cache.npz} -> S3
"$PY" pulse_pred/train.py --uri "$PULSE_MODEL_URI"                   # here (needs the embedding server for new texts)
PULSE_MODEL_URI=... bash deploy/training/run_training.sh             # or as a Job on EKS (needs the pulse-trainer service account)
```

- **Order:** train → evaluate the new model and the current one on held-out tasks → publish.
- **Gate:** the new model becomes `latest` only if its warm@3 is higher. The last line shows the
  verdict (`GATE: promoted …` or `GATE: candidate …`).
- **From the UI:** Models tab → **Retrain**. **Promote** (or roll back) any version from the same
  tab.

### 5.4 lightgbm + markov in the registry

lightgbm and markov share one model file, `action_model.pkl`, published under its own prefix with
the same layout, gate and UI. After `terraform apply` (which gives both prefixes to the reader and
publisher policies):

```bash
export ACTION_MODEL_URI="$(terraform -chdir=terraform output -raw action_model_uri)"   # s3://pulse-models-<account>/action-model
"$PY" pulse_pred/model_store.py data push "$ACTION_MODEL_URI" --local predictor/data --files action_rows.csv   # once
"$PY" pulse_pred/model_store.py publish predictor/data/action_model.pkl "$ACTION_MODEL_URI" --version 2026-11-06 --notes "first model"   # once: the current model
"$PY" predictor/train_action.py --uri "$ACTION_MODEL_URI"                 # retrain here (~1 min)
MODEL=action ACTION_MODEL_URI=... bash deploy/training/run_training.sh     # or as a Job on EKS
```

- **Gate:** the new version becomes `latest` only if lightgbm's held-out top-1 is higher than the
  current latest's on the same 10 held-out tasks.
- **Serving:** with `ACTION_MODEL_URI` set, the `lightgbm` and `markov` predictors load the registry's
  latest version (or `ACTION_MODEL_VERSION`). Without it, or if S3 can't be read, they use
  `predictor/data/action_model.pkl` from git / the image, and `/healthz` shows `model_source`.
- **New data:** rebuild `action_rows.csv` (`predictor/build_action_data.py`), `data push` it, then
  retrain.
- **UI:** Models tab → **lightgbm + markov**: the same Use, Promote, Pin in Git and Retrain buttons.

## 6. One-time setup for infrastructure as code, CI and GitOps

You only need this to use the Terraform cluster (`USE_TERRAFORM=1`), CI on GitHub, or Argo CD.
**Each step creates or changes real AWS or GitHub resources: run them deliberately.**

```bash
# 1. Remote state bucket
terraform -chdir=terraform/bootstrap init && terraform -chdir=terraform/bootstrap apply
B=pulse-tfstate-$(aws sts get-caller-identity --query Account --output text)

# 2. Registry root onto the remote state (moves an existing local state), then apply with the CI roles
terraform -chdir=terraform init -migrate-state -backend-config="bucket=$B"
terraform -chdir=terraform apply -var github_repo=<owner>/<repo>
#    (if ECR pulse-runner already exists: terraform -chdir=terraform import aws_ecr_repository.runner pulse-runner)

# 3. GitHub: Settings -> Secrets and variables -> Actions -> Variables
#    AWS_IMAGE_ROLE_ARN, AWS_PLAN_ROLE_ARN  (terraform -chdir=terraform output ci_role_arns),  TF_STATE_BUCKET=$B

# 4. GitOps manifests for your account (then commit + push)
bash deploy/gitops/configure.sh

# 5. With a cluster up: Argo CD
bash deploy/gitops/install.sh                    # or USE_GITOPS=1 with eks_up.sh
```

**After this setup:**
- **On every pull request:** CI runs the checks, plus a read-only `terraform plan` when
  `terraform/` changes.
- **On every merge to main:** CI builds the image, pushes it to ECR and commits its tag; Argo CD
  then deploys it.
- **Deployment status:** the UI's top bar shows the commit or image and the GitOps sync status.

Details: [terraform/README.md](../terraform/README.md),
[deploy/gitops/README.md](../deploy/gitops/README.md).

## 7. Where things end up

| What | Where |
|---|---|
| Experiment runs | `results/local_runs/` (kind), `results/cloud_runs/` (EKS), or the UI pod's folder (download with zip or `ui_cloud.sh fetch`) |
| Load test runs | `results/load_runs/` |
| Comparison plots | `compare_*` folders next to the runs |
| Controller log | `controller/results/controller.jsonl` (laptop), `kubectl logs deploy/pulse-controller` (EKS) |
| UI process logs | `ui/.state/logs/` |
| Models | S3 `pulse-models-<account>/pulse-v3/<version>/`; local cache in `pulse_pred/data/models/` |
| Metrics | Prometheus in the cluster (2 days on EKS, 1 day on kind) |

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Predictor card says *exited* | Read its last log lines on the card, or `ui/.state/logs/predictor.log`. For pulse_v3, check `PULSE_MODEL_URI` and the AWS credentials |
| `14/14 tools OK` not reached | `kubectl get pods -A`. On kind, recreate the cluster: `teardown.sh`, then `setup.sh`, then 3.1 |
| Every call fails on EKS | Your IP changed: rerun `eks_up.sh` (safe to rerun) |
| Monitoring tab: "not reachable" | Wait about a minute after install. On the laptop the UI keeps the port-forward itself |
| Models tab: an S3 error | No AWS credentials (laptop), or you're in the UI pod (read-only role): use the laptop UI |
| Terraform: "Backend initialization required" | Step 2 of section 6 (`init -migrate-state ...`) |
| Run never finishes in Git Bash | Put `MSYS_NO_PATHCONV=1` in front of your own `kubectl exec/cp` commands |

More fixes: [RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md#troubleshooting) and
[ui/GUIDE.md](../ui/GUIDE.md).
