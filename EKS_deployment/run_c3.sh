#!/usr/bin/env bash
# C3 on EKS: scale-to-zero tools + predictor + prewarm controller (the proposed method).
# Starts the predictor (CRD_MODE=k8s) and the controller in the background, replays the workload with
# hints, then stops both. Their logs go to EKS_deployment/logs/; the controller's action log is also
# copied into the run's result folder as controller.jsonl.
#
#   bash EKS_deployment/run_c3.sh                       # task-9 trace x3 (see common.sh for WORKLOAD)
#   THETA=0.5 bash EKS_deployment/run_c3.sh             # another prewarm threshold (E6)
#   WORKLOAD=taubench bash EKS_deployment/run_c3.sh
#   PREDICTOR=speculator bash EKS_deployment/run_c3.sh  # the LLM speculator (speculative-actions baseline)
#
# PREDICTOR  lightgbm (default) | markov | speculator | pulse_v3: predictor/service.py with PREDICTOR=<name>.
#            The replay reads its /info and sends the inputs it needs (signals for lightgbm, the last 10
#            messages for speculator / pulse_v3). lightgbm and markov need predictor/data/action_model.pkl;
#            speculator needs GROQ_API_KEY_1.. (or SPEC_BASE_URL) in agent/.env; pulse_v3 a model (PULSE_MODEL_URI)
# The label gets -<PREDICTOR>, e.g. results/cloud_runs/<date>_<time>_C3_W1-replay_<workload>_eks-lightgbm/
source "$(dirname "$0")/common.sh"
THETA=${THETA:-0.3}
PORT=${PREDICTOR_PORT:-8000}
PREDICTOR=${PREDICTOR:-lightgbm}
case "$PREDICTOR" in gbt|lgbm) PREDICTOR=lightgbm ;; ngram_action) PREDICTOR=markov ;; esac   # old names
LABEL="$LABEL-$PREDICTOR"
STAMP=$(date +%Y-%m-%d_%H%M)

case "$PREDICTOR" in
  lightgbm|markov)
    [ -f predictor/data/action_model.pkl ] || {
      echo "No predictor/data/action_model.pkl: run predictor/build_action_data.py and predictor/train_action.py"; exit 1; }
    HINT_ARGS=() ;;
  speculator|pulse_v3) HINT_ARGS=(--hint-timeout 10) ;;
  *) echo "Unknown PREDICTOR=$PREDICTOR (use lightgbm, markov, speculator or pulse_v3)"; exit 1 ;;
esac
SERVICE=(-m uvicorn service:app --app-dir predictor --port "$PORT")

ensure_min_scale 0
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
kubectl delete pwr --all --ignore-not-found      # start from no PrewarmRequests

CONTROLLER_LOG="$LOGS/${STAMP}_controller.jsonl"
CONTROLLER_LOG="$CONTROLLER_LOG" "$PY" controller/prewarm_controller.py > "$LOGS/${STAMP}_controller.out" 2>&1 &
BG_PIDS+=($!)
CRD_MODE=k8s THETA="$THETA" PREDICTOR="$PREDICTOR" "$PY" "${SERVICE[@]}" \
  > "$LOGS/${STAMP}_predictor.out" 2>&1 &
BG_PIDS+=($!)

echo "Waiting for the $PREDICTOR predictor on port $PORT ..."
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  [ "$i" = 60 ] && { echo "Predictor did not start; see $LOGS/${STAMP}_predictor.out"; exit 1; }
  sleep 1
done
echo "Predictor up ($PREDICTOR, theta $THETA); controller running (log: $CONTROLLER_LOG)"

replay --predictor-url "http://127.0.0.1:$PORT" "${HINT_ARGS[@]}" "$@"
RUN=$(ls -td "$RESULTS"/*_C3_W1-replay_* | head -1)
[ -f "$CONTROLLER_LOG" ] && cp "$CONTROLLER_LOG" "$RUN/controller.jsonl" && echo "Controller log copied to $RUN/controller.jsonl"
curl -sf "http://127.0.0.1:$PORT/healthz" > "$RUN/predictor_health.json" || true   # e.g. the speculator's LLM call counts
