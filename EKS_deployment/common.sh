# Shared settings and helpers for the EKS run scripts (run_c0.sh, run_c1.sh, run_c3.sh, run_all.sh).
# Sourced by them; not meant to be run on its own.
#
# Settings (environment variables, all optional):
#   PYTHON     Python with agent/requirements.txt + predictor/requirements.txt   (default: python)
#   WORKLOAD   task9    our agent's task-9 live trace, replayed 3 times, one episode at a time (~10 min)
#              taubench the 46 recorded tau-bench episodes, overlapping, 2 per minute      (~30 min)
#              heldout  46 tau-bench episodes of tasks 6, 20, 22, 26, 30 only: the W1 tasks the
#                       predictor never saw in training (its test split), same pacing      (~30 min)
#                                                                                   (default: task9)
#   LABEL      text at the end of every result folder name                       (default: eks)
#   CONCURRENCY  requests one tool pod serves at once (default 1: parallel calls make the autoscaler
#                start more pods; 0 = no limit, the setup of the runs before fan-out)
#   MAX_SCALE    most pods per tool                                              (default: 10)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python}
WORKLOAD=${WORKLOAD:-task9}
LABEL=${LABEL:-eks}
RESULTS="$ROOT/results/cloud_runs"          # new runs; runs up to 25 Oct 2026 stay in test/test_knative/results
export PULSE_RESULTS_DIR="$RESULTS"          # the Python runners write here too
LOGS="$ROOT/EKS_deployment/logs"
mkdir -p "$LOGS"

"$PY" -c "import mcp, kubernetes, sklearn, fastapi" 2>/dev/null || {
  echo "'$PY' is missing packages. Activate your environment (e.g. conda activate pulse) or set PYTHON=..., then:"
  echo "  $PY -m pip install -r agent/requirements.txt -r predictor/requirements.txt"
  exit 1
}

# Point kubectl, KUBE_CONTEXT and MCP_URL_TEMPLATE at the EKS cluster
source "$ROOT/EKS_deployment/use_eks.sh"
case "$(kubectl config current-context)" in
  *pulse-eks*) ;;
  *) echo "kubectl is not on the EKS cluster; run EKS_deployment/eks_up.sh first"; exit 1 ;;
esac

case "$WORKLOAD" in
  task9)    REPLAY_ARGS=(--traces test/test_knative/results/2026-10-21_1527_C0_W1-live_task9_gpt-oss-120b
                         --repeat 3 --sequential) ;;
  taubench) REPLAY_ARGS=() ;;
  heldout)  REPLAY_ARGS=(--tasks 6 20 22 26 30) ;;
  *) echo "Unknown WORKLOAD=$WORKLOAD (use task9, taubench or heldout)"; exit 1 ;;
esac

# Deploy the 14 tools with this min-scale (0 = C0/C3, 1 = C1), unless they already have it
# and with CONCURRENCY requests per pod (default 1) and at most MAX_SCALE pods per tool (default 10),
# see deploy/knative/deploy.sh. Compared as "min-scale concurrency max-scale"; a missing setting counts as 0.
ensure_min_scale() {
  local want="$1 ${CONCURRENCY:-1} ${MAX_SCALE:-10}" have
  have=$(kubectl get ksvc -l pulse/role=mcp-tool \
    -o jsonpath='{range .items[*]}{.spec.template.metadata.annotations.autoscaling\.knative\.dev/min-scale}|{.spec.template.spec.containerConcurrency}|{.spec.template.metadata.annotations.autoscaling\.knative\.dev/max-scale}{"\n"}{end}' \
    | awk -F'|' '{print ($1 == "" ? 0 : $1), ($2 == "" ? 0 : $2), ($3 == "" ? 0 : $3)}' | sort -u)
  if [ "$have" = "$want" ]; then
    echo "Tools already at min-scale / concurrency / max-scale $want"
  else
    echo "Deploying the tools with min-scale / concurrency / max-scale $want ..."
    SKIP_BUILD=1 bash deploy/knative/deploy.sh "$1"    # CONCURRENCY / MAX_SCALE reach it via the environment
  fi
}

# Replay the workload; extra arguments go to w1_replay.py. Prints the new result folder last.
replay() {
  "$PY" test/test_knative/w1_replay.py "${REPLAY_ARGS[@]}" --label "$LABEL" "$@"
  echo "Result: $(ls -td "$RESULTS"/*_W1-replay_* | head -1)"
}

# Background processes started by a script; stopped when it exits (also on Ctrl+C)
BG_PIDS=()
stop_background() {
  for pid in "${BG_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
}
trap stop_background EXIT
