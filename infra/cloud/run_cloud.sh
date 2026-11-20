#!/usr/bin/env bash
# Run a replay, a live run, a predictor load test or a cold-start check INSIDE the cluster (a pod with the runner
# image) and fetch its result folder. The laptop only starts the run, streams its log and copies the results back.
#
#   CONFIG=C0 bash infra/cloud/run_cloud.sh --fanout                       # replay, no prewarming
#   CONFIG=C3 PREDICTOR=lightgbm_adapt bash infra/cloud/run_cloud.sh --fanout
#   MODE=live CONFIG=C3 bash infra/cloud/run_cloud.sh --tasks 30 --agent-model qwen/qwen3.8-27b
#   MODE=load PREDICTOR=markov bash infra/cloud/run_cloud.sh --rates 1 5 10 20 --duration 60   # predictor API load
#   MODE=coldstart bash infra/cloud/run_cloud.sh --reps 5                                       # per-tool cold start
#
# Settings: MODE replay|live|load|coldstart (replay) · CONFIG C0|C1|C3 (C3; load: always C3, coldstart: always C0)
#           PREDICTOR lightgbm|lightgbm_adapt|markov|speculator|pulse_v3 (lightgbm): switches deploy/pulse-predictor to it
#           PREWARM_POLICY adaptive|topk (adaptive) · TOP_K (2, topk) · P_MIN (0.05): which tools a step prewarms
#           (pulse/predictor/policy.py: adaptive = top 3 candidates, the controller warms those worth a slot now)
#           WORKLOAD heldout|taubench|none (heldout; replay only: --tasks 6 20 22 26 30 or all tau-bench runs)
#           DOMAIN airline|retail (airline): the retail tools (Setup -> Deploy the retail tools; pass --tasks, e.g.
#           the held-out retail tasks); the speculator is airline only
#           CONTROLLER ping|lease (ping): C3's prewarm controller. ping = Option A, every prediction starts its k
#           pods; lease = admission control, only the new pods that fit next to a reserve of free pods
#           (LEASE_RESERVE_PODS, 4; pulse/controller/leases.py)
#           how long prewarmed pods live (pulse/controller/refresh.py; ping controller): REFRESH_POLICY
#           always|never|value (always) · SHED_P (0 = off) · TOOL_WINDOWS 60|per-tool (60) · WINDOW (60s): the
#           tools' idle window when TOOL_WINDOWS=60 (the shortest TTL a prewarm can have; the TTL sweep uses 20s)
#           K_ADAPT 0|1 (0): learn the k rule's warm time (PROM_URL) and budget at runtime (pulse/controller/adaptive.py)
#           TOPK_PRICE fixed|learned (fixed): adaptive top-k's slot price when full, learned from tool pods' slot waits
#           PREDICTOR_ENV / CONTROLLER_ENV: any other setting, "KEY=VALUE KEY=VALUE" (e.g. PREDICTOR_ENV="TTL_MODE=fixed
#           TTL_S=30", CONTROLLER_ENV="K_RULE=0"). Every C3 run first clears the settings in KNOBS_* below, so a
#           setting from an earlier run never leaks into the next one
#           LABEL (cloud) · PYTHON (for deploy.sh if the tools must be redeployed) · TARGET kind|ecr (auto)
#           COLLECT=<run pod>: fetch an already running/finished run instead of starting one (after a lost connection)
# Needs cloud_up.sh done. Results: results/cloud_runs/ (EKS) or results/local_runs/ (kind); load tests and
# cold-start checks land in the same place. C3 runs get controller.jsonl; the pod's log goes to infra/eks/logs/.
# Exit code: 0 ok, 1 failed, 2 lost the cluster for 30 min (the run pod keeps its results for 1 h: use COLLECT).
set -euo pipefail
export MSYS_NO_PATHCONV=1    # Git Bash on Windows would rewrite the pod paths (/home/app/...) into C:/Program Files/Git/...
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=infra/cloud
MODE=${MODE:-replay}
CONFIG=${CONFIG:-C3}
case "$MODE" in load) CONFIG=C3 ;; coldstart) CONFIG=C0 ;; esac
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
TOPK_PRICE=${TOPK_PRICE:-fixed}
PROM_URL=${PROM_URL:-http://prometheus-operated.monitoring.svc:9090}   # the in-cluster Prometheus (infra/monitoring)
SHED_P=${SHED_P:-0}
TOOL_WINDOWS=${TOOL_WINDOWS:-60}
want_window=${WINDOW:-60s}
PREDICTOR_ENV=${PREDICTOR_ENV:-}
CONTROLLER_ENV=${CONTROLLER_ENV:-}
# settings an experiment may pass in PREDICTOR_ENV / CONTROLLER_ENV: cleared before every C3 run (back to defaults)
KNOBS_PREDICTOR="TOP_K_MAX TTL_MODE TTL_S TTL_STEP_S TTL_AHEAD_S LOOKAHEAD THETA_NEXT LOOKAHEAD_MIN_RESPOND
  CALIB_PRIOR_N SESSION_TIMEOUT_S ADAPT_LR ADAPT_L2 ADAPT_PASSES ADAPT_BUFFER THETA"
KNOBS_CONTROLLER="K_RULE K_BUDGET_S TOOL_WARM_S TOOL_WARM_S_JSON TTL_KEEPALIVE TOPK_PRICE_S TOPK_FULL_X WARM_POLL_S
  WARM_MIN_CALLS REF_COLD_S K_FREE_HIGH WAIT_MIN_S WAIT_MAX_S REFRESH_FREE_HIGH REFRESH_P_MAX LEASE_K_MAX
  LEASE_TTL_MIN_S LEASE_TTL_MAX_S TOOL_WINDOW_S MAX_REQUEST_AGE_S"
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

# COLLECT=<pod>: only fetch the results of a run pod that is already running (e.g. after the laptop lost its
# connection): skips the set-up below and goes straight to waiting + copying.
if [ -n "${COLLECT:-}" ]; then
  POD=$COLLECT
  echo "Collecting $POD (CONFIG=$CONFIG) | context $CTX"
  kubectl get pod "$POD" >/dev/null || { echo "No pod $POD"; exit 1; }
else
# ---- tools: scale-to-zero for C0 / C3, one warm replica for C1, and the idle window: WINDOW (60 s), or each
#      tool's own (TOOL_WINDOWS=per-tool: data/tool_windows.json, pulse/controller/refresh.py). Patched only where
#      it differs; the rest of each tool's settings stays as infra/knative/deploy.sh made it
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
# A changed min-scale / window is a new Knative revision, and by default a new revision must first start one pod
# per tool. Going from C1 (one pod each) back to 0 would then need two pods per tool for a moment: more than two
# small workers hold, and the revisions fail ("Initial scale was never achieved"). So new revisions may start at
# zero pods (initial-scale 0; it changes nothing about cold starts, only how a revision is created).
kubectl patch cm config-autoscaler -n knative-serving --type merge -p '{"data":{"allow-zero-initial-scale":"true"}}' >/dev/null
tools_of() {   # ROLE -> "name min-scale window initial-scale" lines
  kubectl get ksvc -l pulse/role=$1 -o jsonpath='{range .items[*]}{.metadata.name} {.spec.template.metadata.annotations.autoscaling\.knative\.dev/min-scale} {.spec.template.metadata.annotations.autoscaling\.knative\.dev/window} {.spec.template.metadata.annotations.autoscaling\.knative\.dev/initial-scale}{"\n"}{end}'
}
patch_tool() {   # name min-scale window
  kubectl patch ksvc "$1" --type merge -p \
    "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"autoscaling.knative.dev/min-scale\":\"$2\",\"autoscaling.knative.dev/window\":\"$3\",\"autoscaling.knative.dev/initial-scale\":\"$2\"}}}}}" >/dev/null
}
# the other domain's tools back at zero (a C1 run of it leaves one pod per tool): they would take the workers'
# CPU and memory from this run
OTHER_ROLE=$([ "$ROLE" = mcp-tool ] && echo mcp-tool-retail || echo mcp-tool)
other=0
while read -r s min win init; do
  [ -n "$s" ] || continue
  if [ "${min:-0}" != 0 ] || [ "${init:-}" != 0 ]; then
    [ $other = 0 ] && echo "Setting the other domain's tools ($OTHER_ROLE) back to min-scale 0 ..."
    patch_tool "$s" 0 "${win:-60s}"; other=1
  fi
done < <(tools_of $OTHER_ROLE 2>/dev/null)
patched=0
while read -r s min win init; do
  [ -n "$s" ] || continue
  want=$(window_for "$s")
  if [ "$min/$win/$init" != "$want_min/$want/$want_min" ]; then
    [ $patched = 0 ] && echo "Setting the $DOMAIN tools to min-scale $want_min, idle window(s) $([ "$TOOL_WINDOWS" = per-tool ] && echo per-tool || echo "$want_window") ..."
    patch_tool "$s" "$want_min" "$want"
    patched=1
  fi
done < <(tools_of $ROLE)
[ $patched = 0 ] || kubectl wait ksvc -l pulse/role=$ROLE --for=condition=Ready --timeout=600s >/dev/null
if [ $other = 1 ]; then
  kubectl wait ksvc -l pulse/role=$OTHER_ROLE --for=condition=Ready --timeout=600s >/dev/null
  for i in $(seq 1 60); do      # their pods gone (the idle window, then Knative's grace period): at most 10 min
    [ -z "$(kubectl get pods -l pulse/role=$OTHER_ROLE --field-selector=status.phase=Running -o name 2>/dev/null)" ] && break
    sleep 10
  done
fi

# ---- arguments for the runner
[ "$CONFIG" = C3 ] && LABEL="$LABEL-$PREDICTOR"     # e.g. ..._cloud-lightgbm
if [ "$MODE" != load ] && [ "$MODE" != coldstart ]; then
  [ "$CONFIG" = C3 ] && [ "$PREWARM_POLICY" = topk ] && LABEL="$LABEL-top$TOP_K"
  [ "$CONTROLLER" = lease ] && LABEL="$LABEL-lease"
  [ "$CONFIG" = C3 ] && [ "$REFRESH_POLICY" != always ] && LABEL="$LABEL-refresh_$REFRESH_POLICY"
  [ "$CONFIG" = C3 ] && [ "$SHED_P" != 0 ] && LABEL="$LABEL-shed$SHED_P"
  [ "$TOOL_WINDOWS" = per-tool ] && LABEL="$LABEL-tw"
  [ "$TOOL_WINDOWS" != per-tool ] && [ "$want_window" != 60s ] && LABEL="$LABEL-w$want_window"
  [ "$CONFIG" = C3 ] && [ "$K_ADAPT" = 1 ] && LABEL="$LABEL-kadapt"
  [ "$CONFIG" = C3 ] && [ "$PREWARM_POLICY" = adaptive ] && [ "$TOPK_PRICE" = learned ] && LABEL="$LABEL-learnedprice"
  if [ "$CONFIG" = C3 ]; then for kv in $PREDICTOR_ENV $CONTROLLER_ENV; do LABEL="$LABEL-${kv//=/}"; done; fi
fi
LABEL=$(printf '%s' "$LABEL" | tr -c 'A-Za-z0-9._+-' '_' | cut -c1-120)
case "$MODE" in
  load) ARGS=(predictor --url http://pulse-predictor:8000 --config C3 --label "$LABEL")
        [ "$DOMAIN" = airline ] && [ "$WORKLOAD" = heldout ] && ARGS+=(--tasks "6 20 22 26 30") ;;
  coldstart) ARGS=(--label "$LABEL") ;;
  *) ARGS=(--label "$LABEL")
     if [ "$MODE" = replay ]; then
       case "$WORKLOAD" in
         heldout)  ARGS+=(--tasks 6 20 22 26 30) ;;
         taubench|none) ;;
         *) echo "Unknown WORKLOAD=$WORKLOAD"; exit 1 ;;
       esac
     fi ;;
esac
if [ "$CONFIG" = C3 ]; then
  case "$PREDICTOR" in
    lightgbm|lightgbm_adapt|markov) ;;
    speculator|pulse_v3) [ "$MODE" = replay ] && ARGS+=(--hint-timeout 10) ;;   # slower answers (LLM / embeddings)
    *) echo "Unknown PREDICTOR=$PREDICTOR (lightgbm, lightgbm_adapt, markov, speculator or pulse_v3)"; exit 1 ;;
  esac
  # the hint senders read the predictor's /info and send the inputs it needs (signals and/or history)
  [ "$MODE" = load ] || ARGS+=(--predictor-url http://pulse-predictor:8000)
  # fresh state for this run: no old PrewarmRequests, a new controller and predictor (clean logs and counters);
  # set env switches the one pulse-predictor Deployment to this predictor (a rollout either way)
  kubectl delete pwr --all --ignore-not-found >/dev/null
  # clear every experiment setting, then set this run's (set env persists on the Deployment between runs)
  # shellcheck disable=SC2046
  kubectl set env deploy/pulse-predictor $(for k in $KNOBS_PREDICTOR; do printf '%s- ' "$k"; done) >/dev/null
  # shellcheck disable=SC2046
  kubectl set env deploy/pulse-controller $(for k in $KNOBS_CONTROLLER; do printf '%s- ' "$k"; done) >/dev/null
  # DOMAIN: the retail lightgbm / markov model, and PrewarmRequests for the retail-<tool> services
  # shellcheck disable=SC2086
  kubectl set env deploy/pulse-predictor PREDICTOR="$PREDICTOR" DOMAIN="$DOMAIN" PREWARM_POLICY="$PREWARM_POLICY" \
    TOP_K="$TOP_K" P_MIN="$P_MIN" $PREDICTOR_ENV >/dev/null
  # pulse_v3 loads its model from S3: PULSE_MODEL_URI from here, else the pulse-config ConfigMap (GitOps)
  [ "$PREDICTOR" = pulse_v3 ] && [ -n "${PULSE_MODEL_URI:-}" ] && \
    kubectl set env deploy/pulse-predictor PULSE_MODEL_URI="$PULSE_MODEL_URI" >/dev/null
  # the controller: ping (Option A) or lease (TTLs, reserve, eviction)
  # shellcheck disable=SC2086
  kubectl set env deploy/pulse-controller CONTROLLER_MODE="$CONTROLLER" LEASE_RESERVE_PODS="$LEASE_RESERVE_PODS" \
    REFRESH_POLICY="$REFRESH_POLICY" SHED_P="$SHED_P" K_ADAPT="$K_ADAPT" TOPK_PRICE="$TOPK_PRICE" \
    PROM_URL="$PROM_URL" $CONTROLLER_ENV >/dev/null
  kubectl rollout restart deploy/pulse-controller deploy/pulse-predictor >/dev/null
  kubectl rollout status deploy/pulse-controller --timeout=300s >/dev/null
  kubectl rollout status deploy/pulse-predictor --timeout=300s >/dev/null
fi
ARGS+=("$@")
echo "Run $POD | $MODE $CONFIG$([ "$CONFIG" = C3 ] && echo " ($PREDICTOR, $CONTROLLER controller)") | $DOMAIN | context $CTX"
[ -z "$PREDICTOR_ENV$CONTROLLER_ENV" ] || echo "  predictor env: ${PREDICTOR_ENV:-none} | controller env: ${CONTROLLER_ENV:-none}"
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
fi

kubectl wait --for=condition=Ready "pod/$POD" --timeout=900s >/dev/null
LOG="infra/eks/logs/${STAMP}_${POD}.log"
kubectl logs -f "$POD" > "$LOG" 2>&1 &
LOGPID=$!
echo "  running; log: $LOG"
# A failed kubectl call is not a gone pod: the laptop may be offline for a while (sleep, Wi-Fi, VPN, a dropped
# connection like "http2: server sent GOAWAY"). Keep waiting; the pod holds its results for HOLD_S (1 h).
fails=0
until kubectl exec "$POD" -- test -f "$RDIR/.done" 2>/dev/null; do
  if out=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>&1); then
    fails=0
    case "$out" in Failed|Succeeded) echo "Pod ended ($out) before finishing; see $LOG"; exit 1 ;; esac
  elif echo "$out" | grep -qi "notfound"; then
    echo "Pod $POD is gone before finishing; see $LOG"; exit 1
  else
    fails=$((fails + 1))
    [ $((fails % 4)) = 1 ] && echo "  (no answer from the cluster: $(echo "$out" | tail -1 | cut -c1-120); still waiting)"
    if [ "$fails" -ge 120 ]; then
      echo "No answer from the cluster for 30 min. Resume with: COLLECT=$POD CONFIG=$CONFIG MODE=$MODE bash $HERE/run_cloud.sh"
      exit 2
    fi
  fi
  sleep 15
done
CODE=""
for try in 1 2 3 4 5; do
  CODE=$(kubectl exec "$POD" -- cat "$RDIR/.exit_code" 2>/dev/null) && break
  sleep 10
done

# ---- copy the results out (relative path: kubectl cp reads "C:" as a pod name)
INCOMING="_incoming_$POD"
for try in 1 2 3 4 5; do                    # a dropped connection mid-copy: start the copy again
  rm -rf "${RESULTS:?}/$INCOMING"
  ( cd "$RESULTS" && kubectl cp "default/$POD:$RDIR" "$INCOMING" >/dev/null ) && break
  echo "  copy failed (try $try of 5); retrying in 20 s"
  sleep 20
done
RUNS=()
for d in "$RESULTS/$INCOMING"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mv "$d" "$RESULTS/$name"
  RUNS+=("$RESULTS/$name")
done
rm -rf "$RESULTS/$INCOMING"
if [ "$CONFIG" = C3 ] && [ "$MODE" != load ] && [ ${#RUNS[@]} -gt 0 ]; then
  kubectl logs deploy/pulse-controller | grep '^{' > "${RUNS[0]}/controller.jsonl" || true
fi
kubectl exec "$POD" -- touch /tmp/collected >/dev/null 2>&1 || true
kill "$LOGPID" 2>/dev/null || true
kubectl delete pod "$POD" --wait=false >/dev/null
echo "Done (runner exit code $CODE). Result: ${RUNS[*]:-none}"
[ "$CODE" = 0 ]
