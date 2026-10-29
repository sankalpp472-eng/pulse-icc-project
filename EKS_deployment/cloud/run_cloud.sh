#!/usr/bin/env bash
# Run a replay or a live run INSIDE the cluster (a pod with the runner image) and fetch its result folder.
# The laptop only starts the run, streams its log and copies the results back.
#
#   CONFIG=C0 bash EKS_deployment/cloud/run_cloud.sh --fanout                       # replay, no prewarming
#   CONFIG=C3 PREDICTOR=gbt bash EKS_deployment/cloud/run_cloud.sh --fanout         # LightGBM predictor
#   CONFIG=C3 PREDICTOR=speculator bash EKS_deployment/cloud/run_cloud.sh --fanout  # LLM speculator
#   MODE=live CONFIG=C3 bash EKS_deployment/cloud/run_cloud.sh --tasks 30 --agent-model qwen/qwen3.8-27b
#
# Settings: MODE replay|live (replay) · CONFIG C0|C1|C3 (C3) · PREDICTOR gbt|speculator (gbt)
#           WORKLOAD heldout|taubench|none (heldout; replay only: --tasks 6 20 22 26 30 or all tau-bench runs)
#           LABEL (cloud) · PYTHON (for deploy.sh if the tools must be redeployed) · TARGET kind|ecr (auto)
# Needs cloud_up.sh done. Results: results/cloud_runs/ (EKS) or results/local_runs/ (kind),
# plus controller.jsonl for C3; the pod's log goes to EKS_deployment/logs/.
set -euo pipefail
export MSYS_NO_PATHCONV=1    # Git Bash on Windows would rewrite the pod paths (/home/app/...) into C:/Program Files/Git/...
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=EKS_deployment/cloud
MODE=${MODE:-replay}
CONFIG=${CONFIG:-C3}
PREDICTOR=${PREDICTOR:-gbt}
WORKLOAD=${WORKLOAD:-heldout}
LABEL=${LABEL:-cloud}
CTX=$(kubectl config current-context)
case "$CTX" in kind-*) TARGET=${TARGET:-kind}; RESULTS=results/local_runs ;; *) TARGET=${TARGET:-ecr}; RESULTS=results/cloud_runs ;; esac
IMAGE=${IMAGE:-$(cat "$HERE/.image-$TARGET" 2>/dev/null || true)}
[ -n "$IMAGE" ] || { echo "No image: TARGET=$TARGET bash $HERE/build_image.sh, then cloud_up.sh"; exit 1; }
mkdir -p "$RESULTS" EKS_deployment/logs
STAMP=$(date +%Y-%m-%d_%H%M%S)
POD="pulse-run-$(date +%m%d-%H%M%S)"
RDIR=/home/app/results

# ---- tools: scale-to-zero for C0 / C3, one warm replica for C1 (redeploy only if needed)
want_min=$([ "$CONFIG" = C1 ] && echo 1 || echo 0)
have=$(kubectl get ksvc -l pulse/role=mcp-tool -o jsonpath='{range .items[*]}{.spec.template.metadata.annotations.autoscaling\.knative\.dev/min-scale}{"\n"}{end}' | sort -u)
if [ "$have" != "$want_min" ]; then
  echo "Redeploying the tools with min-scale $want_min ..."
  KUBE_CONTEXT="$CTX" SKIP_BUILD=1 bash deploy/knative/deploy.sh "$want_min"
fi

# ---- arguments for the runner
[ "$CONFIG" = C3 ] && LABEL="$LABEL-$PREDICTOR"     # e.g. ..._cloud-gbt
ARGS=(--label "$LABEL")
if [ "$MODE" = replay ]; then
  case "$WORKLOAD" in
    heldout)  ARGS+=(--tasks 6 20 22 26 30) ;;
    taubench|none) ;;
    *) echo "Unknown WORKLOAD=$WORKLOAD"; exit 1 ;;
  esac
fi
if [ "$CONFIG" = C3 ]; then
  case "$PREDICTOR" in
    gbt)        PRED_DEPLOY=pulse-predictor;  ARGS+=(--predictor-url http://pulse-predictor:8000) ;;
    speculator) PRED_DEPLOY=pulse-speculator; ARGS+=(--predictor-url http://pulse-speculator:8000)
                [ "$MODE" = replay ] && ARGS+=(--hint-context --hint-timeout 10) ;;
    *) echo "Unknown PREDICTOR=$PREDICTOR (gbt or speculator)"; exit 1 ;;
  esac
  # fresh state for this run: no old PrewarmRequests, a new controller and predictor (clean logs and counters)
  kubectl delete pwr --all --ignore-not-found >/dev/null
  kubectl rollout restart deploy/pulse-controller "deploy/$PRED_DEPLOY" >/dev/null
  kubectl rollout status deploy/pulse-controller --timeout=300s >/dev/null
  kubectl rollout status "deploy/$PRED_DEPLOY" --timeout=300s >/dev/null
fi
ARGS+=("$@")
echo "Run $POD | $MODE $CONFIG$([ "$CONFIG" = C3 ] && echo " ($PREDICTOR)") | context $CTX"
echo "  args: ${ARGS[*]}"

# ---- the run pod (on the client node if the cluster has one, so it does not compete with the tools)
PLACE=""
if [ -n "$(kubectl get nodes -l pulse/role=client -o name)" ]; then
  PLACE='  nodeSelector: {pulse/role: client}
  tolerations: [{key: pulse/role, operator: Equal, value: client, effect: NoSchedule}]'
fi
{
  cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  namespace: default
  labels: {app: pulse-run, pulse/role: runner}
spec:
  restartPolicy: Never
  serviceAccountName: pulse
$PLACE
  containers:
    - name: run
      image: $IMAGE
      imagePullPolicy: IfNotPresent
      command: ["bash", "/app/cloud_entry.sh"]
      args:
        - "$MODE"
YAML
  for a in "${ARGS[@]}"; do printf '        - "%s"\n' "${a//\"/\\\"}"; done
  cat <<YAML
      env:
        - {name: PULSE_RESULTS_DIR, value: $RDIR}
        - {name: PULSE_RUNNER, value: "pod $POD"}
        - {name: HOLD_S, value: "3600"}
      envFrom: [{secretRef: {name: pulse-llm-keys, optional: true}}]
      resources: {requests: {cpu: 500m, memory: 1Gi}, limits: {memory: 3Gi}}
YAML
} | kubectl apply -f - >/dev/null
trap 'echo "(pod $POD left running; delete with: kubectl delete pod $POD)"' INT

kubectl wait --for=condition=Ready "pod/$POD" --timeout=900s >/dev/null
LOG="EKS_deployment/logs/${STAMP}_${POD}.log"
kubectl logs -f "$POD" > "$LOG" 2>&1 &
LOGPID=$!
echo "  running; log: $LOG"
until kubectl exec "$POD" -- test -f "$RDIR/.done" 2>/dev/null; do
  phase=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || echo gone)
  case "$phase" in Failed|Succeeded|gone) echo "Pod ended ($phase) before finishing; see $LOG"; exit 1 ;; esac
  sleep 15
done
CODE=$(kubectl exec "$POD" -- cat "$RDIR/.exit_code")

# ---- copy the results out (relative path: kubectl cp reads "C:" as a pod name)
INCOMING="_incoming_$POD"
( cd "$RESULTS" && kubectl cp "default/$POD:$RDIR" "$INCOMING" >/dev/null )
RUNS=()
for d in "$RESULTS/$INCOMING"/*/; do
  name=$(basename "$d")
  mv "$d" "$RESULTS/$name"
  RUNS+=("$RESULTS/$name")
done
rm -rf "$RESULTS/$INCOMING"
if [ "$CONFIG" = C3 ] && [ ${#RUNS[@]} -gt 0 ]; then
  kubectl logs deploy/pulse-controller | grep '^{' > "${RUNS[0]}/controller.jsonl" || true
fi
kubectl exec "$POD" -- touch /tmp/collected >/dev/null 2>&1 || true
kill "$LOGPID" 2>/dev/null || true
kubectl delete pod "$POD" --wait=false >/dev/null
echo "Done (runner exit code $CODE). Result: ${RUNS[*]:-none}"
[ "$CODE" = 0 ]
