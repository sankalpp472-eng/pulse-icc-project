#!/usr/bin/env bash
# Run a replay or a live run INSIDE the cluster (a pod with the runner image) and fetch its result folder.
# The laptop only starts the run, streams its log and copies the results back.
#
#   CONFIG=C0 bash infra/cloud/run_cloud.sh --fanout                       # replay, no prewarming
#   CONFIG=C3 PREDICTOR=lightgbm bash infra/cloud/run_cloud.sh --fanout    # LightGBM predictor
#   CONFIG=C3 PREDICTOR=speculator bash infra/cloud/run_cloud.sh --fanout  # LLM speculator
#   MODE=live CONFIG=C3 bash infra/cloud/run_cloud.sh --tasks 30 --agent-model qwen/qwen3.8-27b
#
# Settings: MODE replay|live (replay) · CONFIG C0|C1|C3 (C3)
#           PREDICTOR lightgbm|markov|speculator|pulse_v3 (lightgbm): switches deploy/pulse-predictor to it
#           WORKLOAD heldout|taubench|none (heldout; replay only: --tasks 6 20 22 26 30 or all tau-bench runs)
#           DOMAIN airline|retail (airline): retail replays the retail tools (Setup -> Deploy the retail tools; pass
#           --tasks, e.g. the held-out retail tasks); live runs and the speculator are airline only
#           LABEL (cloud) · PYTHON (for deploy.sh if the tools must be redeployed) · TARGET kind|ecr (auto)
# Needs cloud_up.sh done. Results: results/cloud_runs/ (EKS) or results/local_runs/ (kind),
# plus controller.jsonl for C3; the pod's log goes to infra/eks/logs/.
set -euo pipefail
export MSYS_NO_PATHCONV=1    # Git Bash on Windows would rewrite the pod paths (/home/app/...) into C:/Program Files/Git/...
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=infra/cloud
MODE=${MODE:-replay}
CONFIG=${CONFIG:-C3}
PREDICTOR=${PREDICTOR:-lightgbm}
case "$PREDICTOR" in gbt|lgbm) PREDICTOR=lightgbm ;; ngram_action) PREDICTOR=markov ;; esac   # old names
WORKLOAD=${WORKLOAD:-heldout}
LABEL=${LABEL:-cloud}
DOMAIN=${DOMAIN:-airline}
if [ "$DOMAIN" = retail ]; then
  ROLE=mcp-tool-retail; PREFIX=retail-
  [ "$MODE" = live ] && { echo "Live runs use the airline agent: DOMAIN=retail works for replays"; exit 1; }
  [ "$PREDICTOR" = speculator ] && [ "${CONFIG}" = C3 ] && { echo "The speculator knows the airline tools only"; exit 1; }
  [ "$WORKLOAD" = heldout ] && WORKLOAD=none          # the airline held-out tasks: give retail ones with --tasks
else
  DOMAIN=airline; ROLE=mcp-tool; PREFIX=
fi
CTX=$(kubectl config current-context)
case "$CTX" in kind-*) TARGET=${TARGET:-kind}; RESULTS=results/local_runs ;; *) TARGET=${TARGET:-ecr}; RESULTS=results/cloud_runs ;; esac
IMAGE=${IMAGE:-$(cat "$HERE/.image-$TARGET" 2>/dev/null || true)}
[ -n "$IMAGE" ] || { echo "No image: TARGET=$TARGET bash $HERE/build_image.sh, then cloud_up.sh"; exit 1; }
mkdir -p "$RESULTS" infra/eks/logs
STAMP=$(date +%Y-%m-%d_%H%M%S)
POD="pulse-run-$(date +%m%d-%H%M%S)"
RDIR=/home/app/results

# ---- tools: scale-to-zero for C0 / C3, one warm replica for C1 (patched only if needed; the rest of each tool's
#      settings stays as infra/knative/deploy.sh made it)
want_min=$([ "$CONFIG" = C1 ] && echo 1 || echo 0)
[ -n "$(kubectl get ksvc -l pulse/role=$ROLE -o name)" ] || \
  { echo "No $DOMAIN tools in this cluster$([ "$DOMAIN" = retail ] && echo ' (Setup -> Deploy the retail tools)')"; exit 1; }
have=$(kubectl get ksvc -l pulse/role=$ROLE -o jsonpath='{range .items[*]}{.spec.template.metadata.annotations.autoscaling\.knative\.dev/min-scale}{"\n"}{end}' | sort -u)
if [ "$have" != "$want_min" ]; then
  echo "Setting the $DOMAIN tools to min-scale $want_min ..."
  for s in $(kubectl get ksvc -l pulse/role=$ROLE -o name); do
    kubectl patch "$s" --type merge -p \
      "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"autoscaling.knative.dev/min-scale\":\"$want_min\"}}}}}" >/dev/null
  done
  kubectl wait ksvc -l pulse/role=$ROLE --for=condition=Ready --timeout=600s >/dev/null
fi

# ---- arguments for the runner
[ "$CONFIG" = C3 ] && LABEL="$LABEL-$PREDICTOR"     # e.g. ..._cloud-lightgbm
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
    lightgbm|markov) ;;
    speculator|pulse_v3) [ "$MODE" = replay ] && ARGS+=(--hint-timeout 10) ;;   # slower answers (LLM / embeddings)
    *) echo "Unknown PREDICTOR=$PREDICTOR (lightgbm, markov, speculator or pulse_v3)"; exit 1 ;;
  esac
  # the hint senders read the predictor's /info and send the inputs it needs (signals and/or history)
  ARGS+=(--predictor-url http://pulse-predictor:8000)
  # fresh state for this run: no old PrewarmRequests, a new controller and predictor (clean logs and counters);
  # set env switches the one pulse-predictor Deployment to this predictor (a rollout either way)
  kubectl delete pwr --all --ignore-not-found >/dev/null
  # DOMAIN: the retail lightgbm / markov model, and PrewarmRequests for the retail-<tool> services
  kubectl set env deploy/pulse-predictor PREDICTOR="$PREDICTOR" DOMAIN="$DOMAIN" >/dev/null
  # pulse_v3 loads its model from S3: PULSE_MODEL_URI from here, else the pulse-config ConfigMap (GitOps)
  [ "$PREDICTOR" = pulse_v3 ] && [ -n "${PULSE_MODEL_URI:-}" ] && \
    kubectl set env deploy/pulse-predictor PULSE_MODEL_URI="$PULSE_MODEL_URI" >/dev/null
  kubectl rollout restart deploy/pulse-controller deploy/pulse-predictor >/dev/null
  kubectl rollout status deploy/pulse-controller --timeout=300s >/dev/null
  kubectl rollout status deploy/pulse-predictor --timeout=300s >/dev/null
fi
ARGS+=("$@")
echo "Run $POD | $MODE $CONFIG$([ "$CONFIG" = C3 ] && echo " ($PREDICTOR)") | $DOMAIN | context $CTX"
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
        - {name: PULSE_METRICS_PORT, value: "9105"}
        - {name: DOMAIN, value: "$DOMAIN"}
        - {name: MCP_URL_TEMPLATE, value: "http://$PREFIX{svc}.default.svc.cluster.local/mcp"}
      ports: [{name: metrics, containerPort: 9105}]   # live run metrics for Prometheus (infra/monitoring)
      envFrom: [{secretRef: {name: pulse-llm-keys, optional: true}}]
      resources: {requests: {cpu: 500m, memory: 1Gi}, limits: {memory: 3Gi}}
YAML
} | kubectl apply -f - >/dev/null
trap 'echo "(pod $POD left running; delete with: kubectl delete pod $POD)"' INT

kubectl wait --for=condition=Ready "pod/$POD" --timeout=900s >/dev/null
LOG="infra/eks/logs/${STAMP}_${POD}.log"
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
