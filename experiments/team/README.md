# Team experiment scripts

Four scripts, one per teammate. Each one runs the whole pipeline with one command, on your own laptop and your
**own AWS account**:

1. Retrain the predictors.
2. Set up EKS.
3. Run its experiments.
4. Export the telemetry.
5. Push the results to GitHub as a pull request.
6. Optionally, delete the cluster.

| Script | Who | Experiments | Cluster time |
|---|---|---|---|
| `script1_latency_resources.sh` | teammate 1 | **Main latency** (C0, C3 markov, C3 lightgbm_adapt, C1) and **resource usage** (pod-seconds, CPU, memory), airline and retail | ~10 h |
| `script2_topk_ttl.sh` | teammate 2 | **Top-k sweep + adaptive top-k / adaptive k**, and the **TTL sweep** (30/60/120 s), airline | ~10 h |
| `script3_kfanout_concurrency.sh` | teammate 3 | **k fan-out** (k rule vs the predictor's static k), and **concurrency** (~4/10/20 agents at once, C0 vs C3), airline | ~10 h |
| `script4_coldstart_api.sh` | teammate 4 | **Per-tool cold start** (with the start-up breakdown) and **predictor API latency** (1/5/10/20 hints/s), airline and retail. This teammate also gathers the results at the end. | ~2.5 h |

Each script also needs about 1 h of set-up: about 25 min to create the cluster, plus image builds. With
`TRIALS=3` (the default) every configuration runs 3 times, and results are reported as the median, min and max.

## What every script uses

- **Predictors:** `lightgbm_adapt` (lightgbm plus an online adapter) and `markov`. The θ threshold, the speculator
  and pulse_v3 are not used.
  - The prewarm policy is the branch's default: adaptive top-k, plus the k rule for how many pods to start.
  - Each run restarts the predictor and the controller, so every run starts from the same state, including
    lightgbm_adapt's adapter.
- **Held-out tasks** (no model is trained on them). The script checks these after training and stops if they differ:
  - **Airline:** 6 11 20 21 22 26 30 38 45 46. The replays use 6 20 22 26 30: 46 conversations, 293 tool calls,
    2 new conversations per minute, seed 252.
  - **Retail:** 3 14 19 21 23 26 28 29 38 42 44 48 50 51 57 67 68 72 73 82 94 98 106. The replays use all 23:
    46 conversations, 333 tool calls.
- **Identical models on every laptop.** Training is seeded. The script compares a fingerprint of what the trained
  models predict (`pulse/experiments/model_fingerprint.py`) with the values in `lib.sh`, and stops if they differ.
  Set `ALLOW_MODEL_MISMATCH=1` to continue anyway.
- **Cluster:** 4 × m7i-flex.large (2 vCPU, 8 GiB each) in `ap-south-1`.
  - 2 nodes run the tools.
  - 1 node runs the predictor and controller.
  - 1 node runs the replay pod and the monitoring.
  - Every run happens inside the cluster. Your laptop only starts the runs and copies the results back.

## Prerequisites (once)

- **An AWS account of your own.**
  - Run `aws configure` with a user that can create EKS, EC2, IAM, ECR and CloudFormation resources.
  - The account needs at least **8 vCPU** of "Running On-Demand Standard instances" quota (Service Quotas, EC2).
  - The cluster is billed while it runs: the EKS control plane plus the 4 nodes.
- **Tools on your PATH:**
  - `git`, `aws` (v2), `eksctl`, `kubectl`, `helm`, `docker` (Docker Desktop **running**), `curl`
  - Python **3.11**
  - `gh` (GitHub CLI, logged in with `gh auth login`) is optional. Without it, the script pushes your branch and you
    open the pull request yourself.
- **Windows:** run the script from **Git Bash** (not PowerShell).
- **The repo, on this branch:**
  ```bash
  git clone https://github.com/sankalpp472-eng/pulse-icc-project.git
  cd pulse-icc-project
  git switch exp/team-scripts
  ```
- **Keep the laptop awake and online for the whole run.** On Windows, set Power & battery → Screen and sleep →
  "Never" while plugged in.
  - If the laptop drops off, the run continues in the cluster. The script waits up to 30 min for the cluster to
    answer again, then retries collecting the results; the run pod keeps them for 4 h.
  - If it's off for longer, that run is retried the next time you start the script.

## Run it

Get the latest version first (fixes may have landed since you cloned):

```bash
git switch exp/team-scripts && git pull
```


```bash
TEAM_MEMBER=<your-name> bash experiments/team/script<N>_<...>.sh
```

The script asks two questions:
- whether to use your AWS account (at the start)
- whether to delete the cluster (at the end)

`YES=1` answers both without asking; `AUTO_DOWN=1` deletes the cluster without asking.

It's **resumable**. If it stops for any reason (laptop sleep, network, a failed run), run the same command again.
Finished runs are skipped, and failed ones are retried.

Progress is in `results/team/<script>/progress.log`; each run's log is in `results/team/<script>/logs/`.

Useful settings:

| Setting | Default | Meaning |
|---|---|---|
| `TRIALS` | 3 | runs per configuration |
| `ONLY` | all | run only some experiments, e.g. `ONLY="ttl"` |
| `DOMAINS` | airline | scripts 2 and 3: `"airline retail"` adds retail (doubles the time) |
| `SKIP_TRAIN=1` / `SKIP_CLUSTER=1` | | reuse the models / cluster already there |
| `PUBLISH=0` | | do not commit and push |
| `REBUILD=1` | | with an existing cluster: rebuild the runner image and restart the Pulse services from the current code (after a `git pull`) |
| `KEEP_CLUSTER=1` | | never delete the cluster at the end |
| `REGION` | ap-south-1 | AWS region |

## What you get

`results/team/<script>/`, and the same content in `experiments/results/<script>/` on your branch
`exp/<script>-<your-name>`, with a pull request into `exp/team-scripts`:

- **`manifest.tsv`:** every run, with columns experiment, domain, configuration, trial, status, run folder, start and end.
- **`<experiment>/<experiment>.md`, `.json`, `_<domain>.png`:** the summary table and plots (median over trials, with
  min and max).
  - Replays: p50/p95/p99, cold calls, pod-seconds, new pods and their start-up time, concurrent conversations,
    prewarms, wasted prewarms, calls that got no answer (`transport_failures`), plus CPU and memory.
  - Load tests: latency per hint rate.
  - Cold start: per-tool cold and warm latency, and the start-up breakdown.
- **`<experiment>/prometheus/*.csv`:** the raw telemetry for each run's time window. It includes CPU and memory per
  pod, tool pods per tool, pending pods, node CPU and memory, the predictor's p95 and the controller's actions. The
  queries are in `queries.json`. Each run is exported right after it finishes, for its own time window and its own
  domain's tool pods only. Prometheus loses this data when the cluster is deleted, so it's exported first; exporting
  again later never removes runs that were already exported.
- **`runs/<run>/`:** each run's raw files: `calls.csv` (every tool call), `pods.csv`, `summary.json`,
  `controller.jsonl`, `run_config.json`.
- **`environment.txt`, `models.json`:** the git commit, runner image, nodes, held-out tasks and model fingerprints.

## After the runs

Script 4's owner:

1. Merges the four pull requests into `exp/team-scripts`.
2. Checks that each `models.json` shows the same fingerprints and each `environment.txt` the same git commit.
3. Collects the `.md` tables into the report.

Any teammate who kept their cluster deletes it:
```bash
REGION=ap-south-1 bash infra/eks/eks_down.sh
```

## Trying a script locally first (no AWS)

To check that a script works before using AWS, run it in quick mode on a local kind cluster:

```bash
CLUSTER=kind QUICK=1 TRIALS=1 MONITORING=0 TEAM_MEMBER=<name> bash experiments/team/script4_coldstart_api.sh
```

- **What changes:** each run lasts minutes (6 conversations, 10 s per hint rate, 1 cold-start round). Results go to
  `results/team/<script>-kind/`. Nothing is pushed, and the kind cluster is kept.
- **Selecting runs:** `ARMS="top2 ttl30"` runs only those arms. `SKIP_TRAIN=1 SKIP_CLUSTER=1` reuses the models and
  the kind setup from an earlier script.
- **Every step is the same as on EKS:** training and checks, image, Pulse services, retail tools, runs, summaries.
- **The numbers mean nothing.** A laptop's single kind node is far slower and more crowded than EKS: cold starts take
  4–10 s, and the predictor can take over 1 s to answer, so hints time out and prewarms are skipped.
- **Use `MONITORING=0` on a laptop.** Prometheus and Grafana don't fit beside the tools in about 5 GB of Docker
  memory.
- **Windows: if kubectl cannot reach kind after Docker restarts,** the API port may be in a range Windows reserves.
  Check with `netsh interface ipv4 show excludedportrange protocol=tcp`.

## If something goes wrong

| Problem | What to do |
|---|---|
| `held-out tasks or models differ` | You're not on `exp/team-scripts`, or the tau-bench data differs. Run `git status` and `python -m pulse.setup.download_data --force`. |
| A run `FAILED` | Read its log in `results/team/<script>/logs/`, then rerun the script; only the missing runs are repeated. |
| `eks_up failed` | Usually quota or permissions. Read `logs/eks_up.log`. |
| `Docker is not running` | Start Docker Desktop. It's needed to build the runner and retail tool images. |
| `PUSH FAILED` | The results are committed on your branch locally: `git push -u origin exp/<script>-<name>`, then open the PR. |
| Your IP changed | Rerun the script. The load balancer only accepts the IP that set it up, but the experiments don't need it: runs happen inside the cluster. |
