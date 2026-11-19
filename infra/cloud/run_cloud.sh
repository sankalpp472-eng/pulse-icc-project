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
#           PREDICTOR lightgbm|lightgbm_adapt|markov|speculator|pulse_v3 (lightgbm): switches deploy/pulse-predictor to it
#           PREWARM_POLICY adaptive|topk (adaptive) · TOP_K (2, topk) · P_MIN (0.05): which tools a step prewarms
#           (pulse/predictor/policy.py: adaptive = top 3 candidates, the controller warms those worth a slot now)
#           WORKLOAD heldout|taubench|none (heldout; replay only: --tasks 6 20 22 26 30 or all tau-bench runs)
#           DOMAIN airline|retail (airline): replay or live on the retail tools (Setup -> Deploy the retail tools;
#           pass --tasks, e.g. the held-out retail tasks); the speculator is airline only
#           CONTROLLER ping|lease (ping): C3's prewarm controller. ping = Option A, every prediction starts its k
#           pods; lease = admission control, only the new pods that fit next to a reserve of free pods
#           (LEASE_RESERVE_PODS, 4; pulse/controller/leases.py). The tools' window is 60 s for every run
#           how long prewarmed pods live (pulse/controller/refresh.py; ping controller): REFRESH_POLICY
#           always|never|value (always) · SHED_P (0 = off) · TOOL_WINDOWS 60|per-tool (60, every configuration)
#           K_ADAPT 0|1 (0): learn the k rule's warm time (PROM_URL) and budget at runtime (pulse/controller/adaptive.py)
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
CONTROLLER=${CONTROLLER:-ping}
[ "$CONFIG" = C3 ] || CONTROLLER=ping
LEASE_RESERVE_PODS=${LEASE_RESERVE_PODS:-4}
PREWARM_POLICY=${PREWARM_POLICY:-adaptive}
TOP_K=${TOP_K:-2}
P_MIN=${P_MIN:-0.05}
REFRESH_POLICY=${REFRESH_POLICY:-always}
K_ADAPT=${K_ADAPT:-0}
PROM_URL=${PROM_URL:-http://prometheus-operated.monitoring.svc:9090}   # the in-cluster Prometheus (infra/monitoring)
SHED_P=${SHED_P:-0}
TOOL_WINDOWS=${TOOL_WINDOWS:-60}
want_window=60s       # the same for every configuration (deploy.sh's tools.yaml may have been generated otherwise)
if [ "$DOMAIN" = retail ]; then
  ROLE=mcp-tool-retail; PREFIX=retail-
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

# ---- tools: scale-to-zero for C0 / C3, one warm replica for C1, and the idle window: 60 s, or each tool's own
#      (TOOL_WINDOWS=per-tool: data/tool_windows.json, pulse/controller/refresh.py). Patched only where it differs;
#      the rest of each tool's settings stays as infra/knative/deploy.sh made it
want_min=$([ "$CONFIG" = C1 ] && echo 1 || echo 0)
[ -n "$(kubectl get ksvc -l pulse/role=$ROLE -o name)" ] || \
  { echo "No $DOMAIN tools in this cluster$([ "$DOMAIN" = retail ] && echo ' (Setup -> Deploy the retail tools)')"; exit 1; }
window_for() {   # Knative service name -> its idle window
  if [ "$TOOL_WINDOWS" = per-tool ]; then
    "${PYTHON:-python3}" -c 'import json,sys; d=json.load(open("data/tool_windows.json"))[sys.argv[1]]["windows"]
s=sys.argv[2][len(sys.argv[3]):].replace("-","_"); print(f"{d.get(s, 60)}s")' "$DOMAIN" "$1" "$PREFIX"
  else echo "$want_window"; fi
}
[ "$TOOL_WINDOWS" != per-tool ] || [ -f data/tool_windows.json ] || \
  { echo "No data/tool_windows.json: python -m pulse.experiments.tool_windows <C0 runs> --domain $DOMAIN"; exit 1; }
patched=0
while read -r s min win; do
  want=$(window_for "$s")
  if [ "$min/$win" != "$want_min/$want" ]; then
    [ $patched = 0 ] && echo "Setting the $DOMAIN tools to min-scale $want_min, idle window(s) $TOOL_WINDOWS ..."
    kubectl patch ksvc "$s" --type merge -p \
      "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"autoscaling.knative.dev/min-scale\":\"$want_min\",\"autoscaling.knative.dev/window\":\"$want\"}}}}}" >/dev/null
    patched=1
  fi
done < <(kubectl get ksvc -l pulse/role=$ROLE -o jsonpath='{range .items[*]}{.metadata.name} {.spec.template.metadata.annotations.autoscaling\.knative\.dev/min-scale} {.spec.template.metadata.annotations.autoscaling\.knative\.dev/window}{"\n"}{end}')
[ $patched = 0 ] || kubectl wait ksvc -l pulse/role=$ROLE --for=condition=Ready --timeout=600s >/dev/null

# ---- arguments for the runner
[ "$CONFIG" = C3 ] && LABEL="$LABEL-$PREDICTOR"     # e.g. ..._cloud-lightgbm
[ "$CONFIG" = C3 ] && [ "$PREWARM_POLICY" = topk ] && LABEL="$LABEL-top$TOP_K"
[ "$CONTROLLER" = lease ] && LABEL="$LABEL-lease"
[ "$CONFIG" = C3 ] && [ "$REFRESH_POLICY" != always ] && LABEL="$LABEL-refresh_$REFRESH_POLICY"
[ "$CONFIG" = C3 ] && [ "$SHED_P" != 0 ] && LABEL="$LABEL-shed$SHED_P"
[ "$TOOL_WINDOWS" = per-tool ] && LABEL="$LABEL-tw"
[ "$CONFIG" = C3 ] && [ "$K_ADAPT" = 1 ] && LABEL="$LABEL-kadapt"
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
    lightgbm|lightgbm_adapt|markov) ;;
    speculator|pulse_v3) [ "$MODE" = replay ] && ARGS+=(--hint-timeout 10) ;;   # slower answers (LLM / embeddings)
    *) echo "Unknown PREDICTOR=$PREDICTOR (lightgbm, lightgbm_adapt, markov, speculator or pulse_v3)"; exit 1 ;;
  esac
  # the hint senders read the predictor's /info and send the inputs it needs (signals and/or history)
  ARGS+=(--predictor-url http://pulse-predictor:8000)
  # fresh state for this run: no old PrewarmRequests, a new controller and predictor (clean logs and counters);
  # set env switches the one pulse-predictor Deployment to this predictor (a rollout either way)
  kubectl delete pwr --all --ignore-not-found >/dev/null
  # DOMAIN: the retail lightgbm / markov model, and PrewarmRequests for the retail-<tool> services
  kubectl set env deploy/pulse-predictor PREDICTOR="$PREDICTOR" DOMAIN="$DOMAIN" PREWARM_POLICY="$PREWARM_POLICY" \
    TOP_K="$TOP_K" P_MIN="$P_MIN" THETA- >/dev/null
  # pulse_v3 loads its model from S3: PULSE_MODEL_URI from here, else the pulse-config ConfigMap (GitOps)
  [ "$PREDICTOR" = pulse_v3 ] && [ -n "${PULSE_MODEL_URI:-}" ] && \
    kubectl set env deploy/pulse-predictor PULSE_MODEL_URI="$PULSE_MODEL_URI" >/dev/null
  # the controller: ping (Option A) or lease (TTLs, reserve, eviction)
  kubectl set env deploy/pulse-controller CONTROLLER_MODE="$CONTROLLER" LEASE_RESERVE_PODS="$LEASE_RESERVE_PODS" \
    REFRESH_POLICY="$REFRESH_POLICY" SHED_P="$SHED_P" K_ADAPT="$K_ADAPT" PROM_URL="$PROM_URL" >/dev/null
  kubectl rollout restart deploy/pulse-controller deploy/pulse-predictor >/dev/null
  kubectl rollout status deploy/pulse-controller --timeout=300s >/dev/null
  kubectl rollout status deploy/pulse-predictor --timeout=300s >/dev/null
fi
ARGS+=("$@")
echo "Run $POD | $MODE $CONFIG$([ "$CONFIG" = C3 ] && echo " ($PREDICTOR, $CONTROLLER controller)") | $DOMAIN | context $CTX"
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
