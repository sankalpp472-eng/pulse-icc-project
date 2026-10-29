# Running the Pulse experiments, step by step

This page shows how to run every experiment, replay or live, C0 / C1 / C3, on AWS EKS or on the laptop's kind cluster.
Each step says what to type, roughly how long it takes, and what you should see.

- **Path A: EKS, everything in the cluster** (recommended). Used for the 28 Oct 2026 results.
  The runner, predictor, speculator and controller all run as pods. The laptop only starts runs and copies results back.
- **Path B: laptop kind cluster.** Free, but cold starts vary a lot on the laptop. Use it for quick checks.

Run every command in **Git Bash**, from the repository root (`Pulse-project1/`), unless a step says otherwise.

---

## 0. One-time setup

| What | Check / how |
|---|---|
| Docker Desktop running | `docker info` |
| kubectl, kind | `kubectl version --client`, `kind version` (`export PATH="$HOME/.arkade/bin:$PATH"`) |
| aws (logged in), eksctl | `aws sts get-caller-identity`, `eksctl version` (EKS only) |
| Project Python | `agent/.venv`: `python -m venv agent/.venv` then `agent/.venv/Scripts/python -m pip install -r agent/requirements.txt -r predictor/requirements.txt` |
| Plotting Python | `/c/Python311/python.exe -c "import matplotlib"` (agent/.venv has no matplotlib) |
| Groq keys | `agent/.env` with `GROQ_API_KEY_1=...` up to `_4`. Only needed for live runs and the LLM speculator. Never commit this file |
| tau-bench conversations | `predictor/data/tau-bench/historical_trajectories/` must contain `gpt-4o-airline.json` and `sonnet-35-new-airline.json`. Replays need them for the predictor's signals |

For convenience, in every new Git Bash window:

```bash
cd ~/OneDrive/Documents/iisc_ICC/Pulse-project1
export PY="$PWD/agent/.venv/Scripts/python"
```

---

## Path A: EKS, everything in the cluster

### A1. Build the runner image and push it to ECR (~5 min)

One image holds the replay/live runner, the predictor, the speculator and the controller.
**Rebuild it whenever you change code or retrain a model.** Pods only see what is in the image.

```bash
bash EKS_deployment/cloud/build_image.sh
```

You should see `Image: 294056656243.dkr.ecr.ap-south-1.amazonaws.com/pulse-runner:<date>-<commit>`.
The tag is saved in `EKS_deployment/cloud/.image-ecr`, and the other scripts read it from there.

### A2. Create the cluster (~25 min; costs money until A8)

```bash
mkdir -p EKS_deployment/logs
PYTHON="$PY" bash EKS_deployment/eks_up.sh 2>&1 | tee EKS_deployment/logs/eks_up_$(date +%Y%m%d).log
```

This creates 2 worker nodes (tools, Knative, Pulse services) and 1 client node (run pods only), plus Knative and Kourier, Redis and the 14 tools at scale-to-zero.
It ends with a cold/warm table. You want `14/14 tools OK`, with cold around 1.5–2.5 s and warm around 300 ms (the warm figure is measured from the laptop).

### A3. Deploy Pulse's own services (~1 min)

```bash
bash EKS_deployment/cloud/cloud_up.sh            # THETA=0.5 bash ... for another threshold
```

This creates:
- the Secret `pulse-llm-keys`, filled from `agent/.env`
- `pulse-predictor` (LightGBM `gbt`)
- `pulse-speculator`
- `pulse-controller`

All three pods should be `Running`.

### A4. Sanity check (30 s)

```bash
MSYS_NO_PATHCONV=1 kubectl exec deploy/pulse-predictor -- python -c "
import urllib.request,json,time
for i in range(2):
  t=time.time(); r=urllib.request.urlopen(urllib.request.Request('http://get-user-details.default.svc.cluster.local/mcp',
    data=json.dumps({'jsonrpc':'2.0','id':1,'method':'ping'}).encode(),
    headers={'Content-Type':'application/json','Accept':'application/json, text/event-stream'}),timeout=60)
  print(r.status, round(time.time()-t,2),'s')"
```

You should get `200 ~1.4 s` (cold), then `200 ~0.01 s` (warm). If it times out, the tools are not healthy: check `kubectl get pods -A`.

### A5. Replay runs (~25–35 min each, no LLM needed)

Run them **one at a time**: they share the tools.
`run_cloud.sh` sets the tools' min-scale for the config, starts a run pod on the client node, streams its log to `EKS_deployment/logs/`, and copies the result folder into `results/cloud_runs/`.

```bash
CONFIG=C0                      bash EKS_deployment/cloud/run_cloud.sh --fanout   # scale-to-zero, no prewarming
CONFIG=C1                      bash EKS_deployment/cloud/run_cloud.sh --fanout   # always warm (min-scale 1)
CONFIG=C3 PREDICTOR=gbt        bash EKS_deployment/cloud/run_cloud.sh --fanout   # LightGBM predictor
CONFIG=C3 PREDICTOR=speculator bash EKS_deployment/cloud/run_cloud.sh --fanout   # LLM speculator baseline
```

The script ends with `Done (runner exit code 0). Result: results/cloud_runs/<folder>`.

Settings (environment variables in front of the command):

| Setting | Values | Meaning |
|---|---|---|
| `CONFIG` | C0, C1, C3 | Configuration |
| `PREDICTOR` | gbt, speculator | C3 only. The run's label gets `-gbt` / `-speculator` |
| `WORKLOAD` | **heldout** (default), taubench, none | heldout = tasks 6 20 22 26 30 (never seen in training), 46 episodes. taubench = all W1 tasks. none = pass `--tasks` yourself |
| `LABEL` | text, default `cloud` | End of the result folder name. Use it for trials: `LABEL=cloud-t2` |

Extra arguments go to `w1_replay.py`:

| Argument | Default | Meaning |
|---|---|---|
| `--fanout` | off | Fire independent read calls of one step concurrently. **Use it for all new runs** |
| `--seed` | 252 | Same seed = identical request sequence across configs. Keep it fixed |
| `--episodes N` | 46 | Number of episodes |
| `--rate R` | 2.0 | New episodes per minute |
| `--user-turn-median S` | 8.0 | Customer reply time (s) |
| `--think-median S` | 1.5 | Agent think time before a call (s) |
| `--hint-timing fair\|legacy` | fair | fair = one hint per LLM step, with only what was known then. legacy = the runs before 28 Oct |
| `--tasks …` | | Only these tasks (with `WORKLOAD=none`) |

Quick smoke test (~3 min): `CONFIG=C3 PREDICTOR=gbt WORKLOAD=none LABEL=smoke bash EKS_deployment/cloud/run_cloud.sh --tasks 30 --episodes 2 --fanout`

**Trials (protocol: 3 per config, report the median with min/max):** repeat each command with `LABEL=cloud-t1`, `cloud-t2`, `cloud-t3`.
Keep the same seed, so all trials replay the same requests and only the cluster's noise changes.

### A6. Live runs (real LLM agent + simulated customer; uses Groq quota)

```bash
MODE=live CONFIG=C0 bash EKS_deployment/cloud/run_cloud.sh --tasks 30 --backend groq \
  --agent-model qwen/qwen3.8-27b --user-model openai/gpt-oss-20b
MODE=live CONFIG=C3 PREDICTOR=gbt bash EKS_deployment/cloud/run_cloud.sh --tasks 30 --backend groq \
  --agent-model qwen/qwen3.8-27b --user-model openai/gpt-oss-20b
```

Live arguments (`w1_live.py`):

| Argument | Meaning |
|---|---|
| `--tasks …` / `--set safe\|multi\|all\|best5` | Which tasks |
| `--repeats N` | Run every task N times |
| `--user-delay-median S` (8.0) / `--user-delay-sigma` (0.5) / `--user-delay-seed` (252) | Customer reply delay (lognormal, 1–60 s) |
| `-v` | Print the conversation |

- **Only `qwen/qwen3.8-27b` on Groq** batches parallel tool calls. The gpt-oss models call one tool at a time.
- **Live runs are not identical across configs:** the LLM's choices differ. Compare live runs by pattern, not call by call.
- **Scoring live C3 hints:** `"$PY" test/test_knative/score_hints.py <run>/hints.jsonl <run>/episodes.jsonl`
  reports top-1/top-3 hits, prewarm hits, wasted prewarms and the real lead time.

### A7. Plots

```bash
R=results/cloud_runs
/c/Python311/python.exe test/test_knative/plot_compare.py \
  $R/<C0 folder> $R/<C3 gbt folder> $R/<C3 speculator folder> \
  --out $R/compare_$(date +%Y-%m-%d)_heldout-fanout-fair_C0-C3gbt-C3speculator_cloud
```

This writes the latency CDF, percentiles, pod-seconds, cold starts, and `comparison.json` with the numbers.
Only compare runs with the same workload and seed.

### A8. Tear down (~10 min). Do this every time.

```bash
bash EKS_deployment/eks_down.sh 2>&1 | tee EKS_deployment/logs/eks_down_$(date +%Y%m%d).log
```

Check that nothing is left. Every command should print nothing:

```bash
R=ap-south-1
aws eks list-clusters --region $R --query clusters --output text
aws ec2 describe-instances --region $R --filters Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text
aws elb describe-load-balancers --region $R --query 'LoadBalancerDescriptions[].LoadBalancerName' --output text
aws cloudformation list-stacks --region $R --stack-status-filter CREATE_COMPLETE DELETE_FAILED --query 'StackSummaries[].StackName' --output text
aws ec2 describe-volumes --region $R --filters Name=status,Values=available --query 'Volumes[].VolumeId' --output text
```

The ECR repository `pulse-runner` is kept on purpose (small storage cost). Delete it when the project is over:

```bash
aws ecr delete-repository --repository-name pulse-runner --region ap-south-1 --force
```

### A typical cloud evening (~2.5 h, all copy-paste)

```bash
bash EKS_deployment/cloud/build_image.sh
PYTHON="$PY" bash EKS_deployment/eks_up.sh
bash EKS_deployment/cloud/cloud_up.sh
CONFIG=C0 bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=gbt bash EKS_deployment/cloud/run_cloud.sh --fanout
CONFIG=C3 PREDICTOR=speculator bash EKS_deployment/cloud/run_cloud.sh --fanout
/c/Python311/python.exe test/test_knative/plot_compare.py results/cloud_runs/<3 folders> --out results/cloud_runs/compare_<...>
bash EKS_deployment/eks_down.sh
```

---

## Path B: laptop kind cluster

### B1. Cluster + tools (~10 min, one time)

```bash
bash test/test_knative/setup.sh                 # kind cluster pulse-kn + Knative + Kourier (localhost:8081)
SKIP_BUILD=1 bash deploy/knative/deploy.sh 0    # Redis + 14 tools, min-scale 0
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
"$PY" deploy/knative/check_tools.py             # expect 14/14 tools OK
```

If `kubectl get pods -A` shows `kube-scheduler` or `kube-controller-manager` in `CrashLoopBackOff`, the cluster is unhealthy (this happened on 28 Oct).
Recreate it with `bash test/test_knative/teardown.sh && bash test/test_knative/setup.sh`, then repeat B1.
Give Docker 4 GB and close Chrome first.

### B2. C0 / C1 replay

```bash
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --label kind                 # C0
SKIP_BUILD=1 bash deploy/knative/deploy.sh 1                                                      # C1: tools always warm
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --label kind                 # C1
SKIP_BUILD=1 bash deploy/knative/deploy.sh 0                                                      # back to scale-to-zero
```

Results go to `results/local_runs/`.

### B3. C3 replay (three terminals)

```bash
# Terminal 1: controller
KUBE_CONTEXT=kind-pulse-kn "$PY" controller/prewarm_controller.py
# Terminal 2: predictor (LightGBM)
CRD_MODE=k8s KUBE_CONTEXT=kind-pulse-kn PREDICTOR_MODEL=gbt THETA=0.3 "$PY" -m uvicorn service:app --app-dir predictor --port 8000
#   or the LLM speculator instead:
#   PREDICTOR_DIR=predictor CRD_MODE=k8s KUBE_CONTEXT=kind-pulse-kn "$PY" -m uvicorn service:app --app-dir baselines/llm_speculator --port 8000
# Terminal 3: the replay
"$PY" test/test_knative/w1_replay.py --tasks 6 20 22 26 30 --fanout --predictor-url http://127.0.0.1:8000 --label kind-gbt
#   speculator: add --hint-context --hint-timeout 10 and use --label kind-speculator
```

Stop the controller and predictor afterwards (Ctrl+C), and copy `controller/results/controller.jsonl` into the run folder if you need it.

### B4. Live run

```bash
"$PY" test/test_knative/w1_live.py --backend groq --tasks 30 --agent-model qwen/qwen3.8-27b --user-model openai/gpt-oss-20b
#   C3: start terminals 1 and 2 from B3 first, then add  --predictor-url http://127.0.0.1:8000
```

### B5. Alternative: the cloud scripts on kind

The in-cluster setup also works on kind, with the same commands as Path A:

```bash
TARGET=kind bash EKS_deployment/cloud/build_image.sh
TARGET=kind bash EKS_deployment/cloud/cloud_up.sh
CONFIG=C3 PREDICTOR=gbt bash EKS_deployment/cloud/run_cloud.sh --fanout   # picks kind from the kubectl context
```

Results go to `results/local_runs/`.

---

## What a result folder contains

`results/<cloud|local>_runs/<date>_<time>_<config>_W1-<replay|live>_<details>_<label>/`

| File | What |
|---|---|
| `summary.json` | Totals: calls, cold calls, latency percentiles, pods started, pod-seconds, fan-out, hints sent/answered |
| `calls.csv` | One row per tool call: latency, cold (by pod / by latency), gap, group, concurrency |
| `pods.csv` | Every pod seen: start, ready, end (for pod-seconds) |
| `episodes.jsonl` | One line per episode (trace, timing; live runs: the whole step list) |
| `run_config.json` | Every setting, git commit, image, Knative settings |
| `hints.jsonl` | C3: every hint sent, the predictor's answer, round-trip time |
| `controller.jsonl` | C3: every warm-up the controller did (`warmed`, `was_cold`, `pings`, `pods_before/after`) and every `evicted` |
| `predictor_health.json` | C3: predictor model, θ, CRDs written; speculator: LLM call counts and key failures |

Cold call = latency > 1 s, or a new pod appeared for a tool that had no pod before.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `run_cloud.sh` never finishes, pod log ends with `PULSE_RUN_DONE` | Git Bash path conversion. The script sets `MSYS_NO_PATHCONV=1`; for your own `kubectl exec/cp` commands, put `MSYS_NO_PATHCONV=1` in front |
| A run was interrupted | `kubectl get pod -l app=pulse-run`. While it is `Running`, copy with `MSYS_NO_PATHCONV=1 kubectl cp default/<pod>:/home/app/results <local dir>`, then `kubectl delete pod <pod>` |
| `No image` | Run `build_image.sh` (A1) first; for kind use `TARGET=kind` |
| Pods `ImagePullBackOff` on EKS | Image tag in `.image-ecr` not pushed: rerun A1, then A3 |
| Hints `answered` much lower than `sent` | Predictor not reachable or too slow: `kubectl logs deploy/pulse-predictor`; speculator: Groq rate limits (`predictor_health.json` → `no_key_free`) |
| Every call is a transport failure | Tools not healthy (A4 check); on kind see B1 |
| `eks_up.sh` fails on the IP restriction | Your IP changed: rerun `eks_up.sh` (safe to rerun) |
| Windows: "paging file is too small" | Close apps; Docker 4 GB |

Related docs:
- `EKS_deployment/README.md`: the older laptop-driven EKS scripts (`run_c0/c1/c3.sh`)
- `test/test_knative/C0_BASELINE_README.md`: kind setup in detail
- `deploy/prewarm/README.md`: the CRD and controller contract
- `NEXT_EXPERIMENTS.md`: top-k and decision-model experiments
