#!/usr/bin/env bash
# E1 (tool-call latency + startup breakdown) on EKS, unattended: C0, C3 with every predictor, then C1, the same
# replay workload each time (held-out tasks, fan-out, seed 252), then the E1 plots. Meant to run in the background:
#
#   nohup bash infra/cloud/run_e1.sh > infra/eks/logs/e1_$(date +%Y%m%d_%H%M).log 2>&1 &
#   tail -f infra/eks/logs/e1_*.log            # follow it
#
# Resumable: every finished run is written to results/cloud_runs/e1_<E1_ID>/manifest.tsv, and running the same
# command again (same E1_ID, default: today's date) skips what is done and continues. A run whose laptop
# connection dropped is fetched again from its pod (it keeps its results for 1 h).
#
# Settings (environment):
#   PREDICTORS   which C3 predictors, in order (default "lightgbm markov speculator pulse_v3"); one that cannot
#                run here is skipped with the reason (speculator: no Groq keys; pulse_v3: no model / embeddings)
#   CONFIGS      which configurations (default "C0 C3 C1"; C1 last: it needs the tools redeployed always-warm)
#   TRIALS       runs per configuration (default 1; the protocol says 3: about 3.5 h per trial)
#   WORKLOAD     heldout (default) | taubench          ARGS  extra replay arguments (default "--fanout")
#   POLICY_ENV   prewarm policy for every C3 run, e.g. "PREWARM_POLICY=topk TOP_K=2 P_MIN=0.1 LOOKAHEAD=0"
#                (default: the predictor's own, theta 0.3)
#   E1_ID        name of this E1 (default: today's date)   AUTO_DOWN=1  run eks_down.sh at the end
#   PYTHON       Python for deploy.sh (default .venv/Scripts/python, else python); PLOT_PYTHON one with matplotlib
# Needs: the EKS cluster up (infra/eks/eks_up.sh), the image (infra/cloud/build_image.sh) and cloud_up.sh done.
set -uo pipefail
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PREDICTORS=${PREDICTORS:-lightgbm markov speculator pulse_v3}
CONFIGS=${CONFIGS:-C0 C3 C1}
TRIALS=${TRIALS:-1}
WORKLOAD=${WORKLOAD:-heldout}
ARGS=${ARGS:---fanout}
E1_ID=${E1_ID:-$(date +%Y-%m-%d)}
RUN_CLOUD=${RUN_CLOUD:-infra/cloud/run_cloud.sh}
PY=""
for p in "${PYTHON:-}" .venv/Scripts/python.exe .venv/bin/python agent/.venv/Scripts/python.exe python; do
  [ -n "$p" ] && command -v "$p" >/dev/null 2>&1 && { PY=$p; break; }
done
export PYTHON="$PY"
PLOT_PY=""                                   # the plots need matplotlib (not every venv here has it)
for p in "${PLOT_PYTHON:-}" "$PY" .venv/Scripts/python.exe /c/Python311/python.exe python3 python; do
  [ -n "$p" ] && "$p" -c "import matplotlib" >/dev/null 2>&1 && { PLOT_PY=$p; break; }
done
OUT=results/cloud_runs/e1_$E1_ID
MANIFEST=$OUT/manifest.tsv
mkdir -p "$OUT" infra/eks/logs
touch "$MANIFEST"
say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- preflight
CTX=$(kubectl config current-context 2>/dev/null)
kubectl get nodes --request-timeout=15s >/dev/null 2>&1 || { say "Cluster $CTX is not reachable: run infra/eks/eks_up.sh first"; exit 1; }
kubectl get deploy pulse-predictor pulse-controller >/dev/null 2>&1 || { say "No Pulse services: run infra/cloud/cloud_up.sh first"; exit 1; }
say "E1 $E1_ID | context $CTX | configs: $CONFIGS | predictors: $PREDICTORS | trials: $TRIALS | workload: $WORKLOAD $ARGS"
[ -n "${POLICY_ENV:-}" ] && say "C3 prewarm policy: $POLICY_ENV"
say "results: $OUT (manifest: $MANIFEST)"

done_already() { grep -q "^$1	$2	$3	done	" "$MANIFEST"; }
record() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$MANIFEST"; }
record_once() { grep -qF "$(printf '%s\t%s\t%s\t%s' "$1" "$2" "$3" "$4")" "$MANIFEST" || record "$@"; }   # skips: once

# health of the in-cluster predictor after switching it to $1 (does it load, does it answer?)
predictor_ok() {
  kubectl set env deploy/pulse-predictor PREDICTOR="$1" >/dev/null || return 1
  kubectl rollout status deploy/pulse-predictor --timeout=600s >/dev/null 2>&1 || return 1
  kubectl exec deploy/pulse-predictor -- python -c "
import json, sys, time, urllib.request
for _ in range(60):
    try:
        h = json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=10))
        sys.exit(0 if h.get('status') == 'ok' and h.get('predictor') == '$1' else 3)
    except Exception:
        time.sleep(5)
sys.exit(4)" >/dev/null 2>&1
}

# one run through run_cloud.sh; a lost connection -> fetch the same pod again (it keeps its results for 1 h)
one_run() {   # trial config predictor
  local trial=$1 config=$2 pred=$3 log rc pod result
  log="infra/eks/logs/e1_${E1_ID}_t${trial}_${config}_${pred}.log"
  say "start: trial $trial $config ${pred:+($pred)} -> $log"
  CONFIG=$config PREDICTOR=${pred:-lightgbm} WORKLOAD=$WORKLOAD LABEL="e1-t$trial" \
    bash "$RUN_CLOUD" $ARGS > "$log" 2>&1
  rc=$?
  pod=$(grep -o 'pulse-run-[0-9-]*' "$log" | head -1)
  for retry in 1 2 3 4 5 6 7 8 9 10; do
    [ "$rc" = 2 ] || break                 # 2 = the cluster stopped answering while waiting
    say "  lost the cluster during $pod; fetching it again in 2 min (try $retry)"
    sleep "${COLLECT_WAIT_S:-120}"
    COLLECT=$pod CONFIG=$config bash "$RUN_CLOUD" >> "$log" 2>&1
    rc=$?
  done
  result=$(grep -o 'Result: [^ ]*' "$log" | tail -1 | cut -d' ' -f2)
  if [ "$rc" = 0 ] && [ -n "$result" ] && [ -d "$result" ]; then
    record "$trial" "$config" "$pred" done "$result"
    say "  done: $result"
  else
    record "$trial" "$config" "$pred" "failed(rc=$rc)" "${pod:-}"
    say "  FAILED (rc=$rc); see $log"
  fi
}

for trial in $(seq 1 "$TRIALS"); do
  for config in $CONFIGS; do
    if [ "$config" != C3 ]; then
      done_already "$trial" "$config" "" && { say "skip (done): trial $trial $config"; continue; }
      one_run "$trial" "$config" ""
      continue
    fi
    [ -n "${POLICY_ENV:-}" ] && kubectl set env deploy/pulse-predictor $POLICY_ENV >/dev/null
    for pred in $PREDICTORS; do
      done_already "$trial" C3 "$pred" && { say "skip (done): trial $trial C3 $pred"; continue; }
      if [ "$pred" = speculator ] && ! kubectl get secret pulse-llm-keys >/dev/null 2>&1; then
        record_once "$trial" C3 "$pred" "skipped(no Groq keys: secret pulse-llm-keys)" ""
        say "skip: speculator needs Groq keys (agent/.env -> cloud_up.sh makes the secret)"; continue
      fi
      if ! predictor_ok "$pred"; then
        record_once "$trial" C3 "$pred" "skipped(predictor did not become healthy)" ""
        say "skip: $pred did not load in the cluster (kubectl logs deploy/pulse-predictor)"
        [ "$pred" = pulse_v3 ] && say "  pulse_v3 needs its model in S3 (PULSE_MODEL_URI / pulse-config) and pulse-embed"
        continue
      fi
      one_run "$trial" C3 "$pred"
    done
  done
done

# ---- the E1 plots over every finished run
mapfile -t RUNS < <(awk -F'\t' '$4 == "done" {print $5}' "$MANIFEST")
say "finished runs: ${#RUNS[@]}"
column -t -s $'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
if [ ${#RUNS[@]} -gt 0 ] && [ -z "$PLOT_PY" ]; then
  say "no Python with matplotlib for the plots: PLOT_PYTHON=<python> bash $0 (it only plots, the runs are done)"
elif [ ${#RUNS[@]} -gt 0 ]; then
  "$PLOT_PY" -m pulse.experiments.plot_e1 "${RUNS[@]}" --out "$OUT" && say "E1 plots: $OUT/e1_latency.png, $OUT/e1_startup.png, $OUT/e1.md"
fi
if [ "${AUTO_DOWN:-0}" = 1 ]; then
  say "AUTO_DOWN=1: deleting the EKS cluster"
  bash infra/eks/eks_down.sh
fi
say "E1 finished."
