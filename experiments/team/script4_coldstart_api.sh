#!/usr/bin/env bash
# Script 4: per-tool cold start and the predictor API's latency (airline and retail).
#
#   TEAM_MEMBER=<your name> bash experiments/team/script4_coldstart_api.sh
#
# Experiments:
#   coldstart      every tool (airline 14, retail 16): a cold call (no pod) and right after it a warm call, from a
#                  pod in the cluster, COLDSTART_REPS (5) rounds; plus each new pod's start-up breakdown
#                  (scheduling, image pull, init, readiness). pulse/experiments/coldstart.py
#   predictor_api  the prediction service under load: hints at 1 / 5 / 10 / 20 per second for 60 s each (open
#                  loop, from a pod in the cluster) for markov and lightgbm_adapt; NFR-4: p95 <= 50 ms at 10/s.
#                  pulse/loadtest/async_load.py
# TRIALS (3) for predictor_api; about 2.5 h, plus about 1 h of set-up. Script 4's owner also gathers the four
# scripts' results at the end (experiments/team/README.md, "After the runs").
SCRIPT=script4_coldstart_api
source "$(dirname "$0")/lib.sh"
COLDSTART_REPS=${COLDSTART_REPS:-5}
RATES=${RATES:-1 5 10 20}

experiments() {
  local domain trial pred tasks
  if wanted coldstart; then
    for domain in airline retail; do
      run_one coldstart "$domain" "reps$COLDSTART_REPS" 1 MODE=coldstart -- --reps "$COLDSTART_REPS"
    done
  fi
  if wanted predictor_api; then
    for trial in $(seq 1 "$TRIALS"); do
      for domain in airline retail; do
        tasks=$([ "$domain" = retail ] && echo "$RETAIL_REPLAY" || echo "$AIRLINE_REPLAY")
        for pred in $PREDICTORS_USED; do
          # shellcheck disable=SC2086
          run_one predictor_api "$domain" "$pred" "$trial" MODE=load PREDICTOR="$pred" WORKLOAD=none \
            -- --tasks "$tasks" --rates $RATES --duration 60
        done
      done
    done
  fi
}

run_script experiments coldstart predictor_api
