#!/usr/bin/env bash
# C3 on EKS: scale-to-zero tools + predictor + prewarm controller (the proposed method).
# Starts the predictor (CRD_MODE=k8s) and the controller in the background, replays the workload with
# hints, then stops both. Their logs go to EKS_deployment/logs/; the controller's action log is also
# copied into the run's result folder as controller.jsonl.
#
#   bash EKS_deployment/run_c3.sh                       # task-9 trace x3 (see common.sh for WORKLOAD)
#   THETA=0.5 bash EKS_deployment/run_c3.sh             # another prewarm threshold (E6)
#   WORKLOAD=taubench bash EKS_deployment/run_c3.sh
#
# Result: test/test_knative/results/<date>_<time>_C3_W1-replay_<workload>_eks/
source "$(dirname "$0")/common.sh"
THETA=${THETA:-0.3}
PORT=${PREDICTOR_PORT:-8000}
STAMP=$(date +%Y-%m-%d_%H%M)

if [ ! -f predictor/data/model.pkl ]; then
  echo "No trained predictor yet; training it (predictor/data/model.pkl) ..."
  "$PY" predictor/export_traces.py && "$PY" predictor/build_graphs.py && "$PY" predictor/train.py
fi

ensure_min_scale 0
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
kubectl delete pwr --all --ignore-not-found      # start from no PrewarmRequests

CONTROLLER_LOG="$LOGS/${STAMP}_controller.jsonl"
CONTROLLER_LOG="$CONTROLLER_LOG" "$PY" controller/prewarm_controller.py > "$LOGS/${STAMP}_controller.out" 2>&1 &
BG_PIDS+=($!)
CRD_MODE=k8s THETA="$THETA" "$PY" -m uvicorn service:app --app-dir predictor --port "$PORT" \
  > "$LOGS/${STAMP}_predictor.out" 2>&1 &
BG_PIDS+=($!)

echo "Waiting for the predictor on port $PORT ..."
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  [ "$i" = 60 ] && { echo "Predictor did not start; see $LOGS/${STAMP}_predictor.out"; exit 1; }
  sleep 1
done
echo "Predictor up (theta $THETA); controller running (log: $CONTROLLER_LOG)"

replay --predictor-url "http://127.0.0.1:$PORT" "$@"
RUN=$(ls -td "$RESULTS"/*_C3_W1-replay_* | head -1)
[ -f "$CONTROLLER_LOG" ] && cp "$CONTROLLER_LOG" "$RUN/controller.jsonl" && echo "Controller log copied to $RUN/controller.jsonl"
