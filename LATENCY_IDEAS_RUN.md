# Lower p95: top-k prewarming + lookahead + scale-down-delay, then a C3 run on EKS

These three changes target the 37 cold calls left in the 28 Oct C3 run:

| Change | Cold calls it targets | Where |
|---|---|---|
| **Lookahead:** on a reply step, also prewarm the tool the agent calls *after* the customer answers | 18 ("first call after a customer turn") | `train_action.py` (second LightGBM, saved in the same pickle), `service.py` (`LOOKAHEAD=1`) |
| **Top-k:** prewarm the 2 most likely tools (each with P ≥ 0.1) instead of every tool with P ≥ 0.3 | 13 ("call right after another call") and more | `models.py` (`prewarm_select`), `service.py` (`PREWARM_POLICY=topk`) |
| **scale-down-delay:** Knative keeps a pod 30 s longer after deciding to scale down | 5 ("prewarmed earlier, pod gone or busy") | `gen_services.py`, `deploy.sh` (`SCALE_DOWN_DELAY=30s`) |

Everything is off by default, so existing runs and the UI behave as before until you set the variables.
The patch is `latency_ideas.patch`. It was tested on a copy of the repo: retrained, scored offline, and the predictor service was started with all options on.

**Offline estimate** (held-out steps; `predictor/eval_policies.py`):

| Policy | Tool calls prewarmed | First call after a customer turn | Prewarms / step |
|---|---|---|---|
| θ 0.3 (28 Oct run) | 0.561 | 0.466 | 0.49 |
| top-2, p_min 0.1 + lookahead 0.3 | 0.719 | 0.639 | 1.18 |
| top-3, p_min 0.05 | 0.774 | 0.736 | 1.63 |

Top-k is the big lever. Lookahead mostly prewarms the same tools *earlier*: during the customer's turn rather than the ~1.5 s of think time.
Expect fewer cold calls and more pod-seconds than the 28 Oct C3. p95 only drops a lot once the cold share falls below 5%.

---

All commands run in **Git Bash**, from the repo root.

## 1. Apply the changes (laptop, 2 min)

```bash
cd ~/OneDrive/Documents/iisc_ICC/Pulse-project1
export PY="$PWD/agent/.venv/Scripts/python"
git apply --ignore-whitespace latency_ideas.patch
git status --short            # models.py, service.py, train_action.py, gen_services.py, deploy.sh changed; eval_policies.py new
```

## 2. Retrain (adds the lookahead model to action_model.pkl, ~1 min)

```bash
PYTHONIOENCODING=utf-8 "$PY" predictor/train_action.py
```

Check the output:
- **Main model unchanged:** the LightGBM row still shows top-1 0.637, and θ=0.3 hit 0.56.
- **A new last line:** `Lookahead (gbt, trained on 2561 reply steps): … top-1 0.357 / top-3 0.595 …`

## 3. Offline check (~30 s)

```bash
PYTHONIOENCODING=utf-8 "$PY" predictor/eval_policies.py
```

## 4. Commit (optional, recommended)

```bash
git add predictor/models.py predictor/service.py predictor/train_action.py predictor/eval_policies.py \
        predictor/data/action_model.pkl deploy/knative/gen_services.py deploy/knative/deploy.sh
git commit -m "predictor: top-k prewarm policy, lookahead model; tools: optional scale-down-delay"
```

## 5. Build the image with the new code and model (Docker running, ~5 min)

```bash
bash EKS_deployment/cloud/build_image.sh
```

## 6. Bring up EKS, with the tools' scale-down-delay set (~25 min; costs money from here on)

```bash
export SCALE_DOWN_DELAY=30s           # keep this exported in this terminal: any tool redeploy keeps the delay
PYTHON="$PY" bash EKS_deployment/eks_up.sh
kubectl get ksvc get-user-details -o jsonpath='{.spec.template.metadata.annotations}'; echo
```

The annotations should include `"autoscaling.knative.dev/scale-down-delay":"30s"`.

## 7. Deploy Pulse's services, then turn the new predictor options on (~2 min)

```bash
bash EKS_deployment/cloud/cloud_up.sh
kubectl set env deploy/pulse-predictor PREWARM_POLICY=topk TOP_K=2 P_MIN=0.1 LOOKAHEAD=1 THETA_NEXT=0.3
kubectl rollout status deploy/pulse-predictor
MSYS_NO_PATHCONV=1 kubectl exec deploy/pulse-predictor -- python -c "import urllib.request,json; h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz')); print(h['policy'], h['lookahead'])"
```

The last command must print `{'name': 'topk', 'top_k': 2, 'p_min': 0.1}` and `{'on': True, 'available': True, ...}`.
- **`'available': False`:** the image has the old model. Repeat steps 2 and 5, then `bash EKS_deployment/cloud/cloud_up.sh`.
- **The settings persist:** `run_cloud.sh` restarts the predictor before each run, and the env settings survive the restart.

## 8. The runs (one at a time, ~30 min each)

```bash
# a) C0 with the same tool settings (scale-down-delay 30s): the fair baseline for this cluster
CONFIG=C0 LABEL=cloud-sdd30 bash EKS_deployment/cloud/run_cloud.sh --fanout

# b) C3: top-2 + lookahead + scale-down-delay
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-top2-look-sdd30 bash EKS_deployment/cloud/run_cloud.sh --fanout
```

Optional extras, if you have time:

```bash
# c) the old policy on the same cluster: separates "new policy" from "scale-down-delay"
kubectl set env deploy/pulse-predictor PREWARM_POLICY=theta LOOKAHEAD=0
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-theta03-sdd30 bash EKS_deployment/cloud/run_cloud.sh --fanout

# d) more aggressive: top-3 (higher hit rate, more pods)
kubectl set env deploy/pulse-predictor PREWARM_POLICY=topk TOP_K=3 P_MIN=0.05 LOOKAHEAD=1 THETA_NEXT=0.3
CONFIG=C3 PREDICTOR=gbt LABEL=cloud-top3-look-sdd30 bash EKS_deployment/cloud/run_cloud.sh --fanout
```

Each run ends with `Done (runner exit code 0). Result: results/cloud_runs/<folder>`.

## 9. Compare (laptop)

```bash
ls -d results/cloud_runs/*sdd30*
/c/Python311/python.exe test/test_knative/plot_compare.py \
  results/cloud_runs/<C0 sdd30 folder> results/cloud_runs/<C3 top2-look folder> [<C3 theta03 folder>] [<C3 top3 folder>] \
  --out results/cloud_runs/compare_$(date +%Y-%m-%d)_heldout-fanout_top2-look-sdd30
```

Or in the UI: `"$PY" ui/server.py` → Results tab → tick the runs → **Compare selected** / **Make plots**.

What to look at in each run's `summary.json`:

| Field | Meaning |
|---|---|
| `cold_calls`, `cold_share` | Target: well below 37 / 12.6%. Under 5% is what moves p95 |
| `latency_ms.all.p50/p95/p99`, `mean` | The latency numbers |
| `pod_seconds_seen` | The cost. It goes up with top-k; compare with C1's ceiling |
| `hints.answered` | Must equal `hints.sent` |

For `fanout.cold_calls_in_groups`, and the `prewarm-lookahead` lines in the predictor's audit log (if enabled), see the run folder.

## 10. Tear down (every time, ~10 min)

```bash
bash EKS_deployment/eks_down.sh
R=ap-south-1
aws eks list-clusters --region $R --query clusters --output text
aws ec2 describe-instances --region $R --filters Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text
aws elb describe-load-balancers --region $R --query 'LoadBalancerDescriptions[].LoadBalancerName' --output text
aws cloudformation list-stacks --region $R --stack-status-filter CREATE_COMPLETE DELETE_FAILED --query 'StackSummaries[].StackName' --output text
```

Each AWS check must print nothing.

---

## New settings, for reference

| Where | Setting | Default | Meaning |
|---|---|---|---|
| Predictor (`service.py`) | `PREWARM_POLICY` | `theta` | `theta` = every tool with P ≥ THETA; `topk` = the TOP_K most likely tools, each with P ≥ P_MIN |
| | `TOP_K`, `P_MIN` | 2, 0.1 | For `topk` |
| | `LOOKAHEAD` | 0 | 1 = on a likely reply step (P(RESPOND) ≥ `LOOKAHEAD_MIN_RESPOND`, 0.5), also prewarm the tools the lookahead model gives P ≥ `THETA_NEXT` (0.4), with k = 1 |
| | `/predict` | | Has a new `lookahead` list |
| | `/healthz` | | Shows `policy` and `lookahead` |
| Tools (`deploy.sh`) | `SCALE_DOWN_DELAY` | `0s` | Knative `autoscaling.knative.dev/scale-down-delay` on every tool. It changes C0 too, so always compare against a C0 run with the same value |

**Undo:**
- **Predictor:** `kubectl set env deploy/pulse-predictor PREWARM_POLICY=theta LOOKAHEAD=0`
- **Tools:** `unset SCALE_DOWN_DELAY`, then redeploy them with `KUBE_CONTEXT=$(kubectl config current-context) SKIP_BUILD=1 PYTHON="$PY" bash deploy/knative/deploy.sh 0`
- **Code:** `git apply -R --ignore-whitespace latency_ideas.patch` (before committing)
