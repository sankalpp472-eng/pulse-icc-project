#!/usr/bin/env bash
# Script 1: main latency benchmark + resource usage (airline and retail).
#
#   TEAM_MEMBER=<your name> bash experiments/team/script1_latency_resources.sh
#
# Experiments (the same runs serve both; experiment id latency_resources):
#   main latency    tool-call latency p50 / p95 / p99 and cold calls for
#                   C0 (scale to zero), C3 markov, C3 lightgbm_adapt (adaptive top-k, k rule), C1 (always warm)
#   resource usage  what each configuration costs: pod-seconds and new pods (from the runs), and from Prometheus
#                   the tool pods' CPU core-seconds and memory, the Pulse services' CPU, node CPU, peak tool pods
# Workload: the held-out tasks' recorded conversations, replayed with fan-out (airline 6 20 22 26 30, retail all 23
# held-out tasks; 46 episodes, 2 per minute, seed 252). TRIALS (3) x 4 configurations x 2 domains, ~25 min each:
# about 10 h, plus about 1 h of set-up. Resumable: run the same command again.
SCRIPT=script1_latency_resources
source "$(dirname "$0")/lib.sh"

experiments() {
  local domain trial
  wanted latency_resources || return 0
  for domain in airline retail; do
    for trial in $(seq 1 "$TRIALS"); do
      # shellcheck disable=SC2046
      run_one latency_resources "$domain" C0 "$trial" CONFIG=C0 -- --fanout $(replay_tasks "$domain")
      # shellcheck disable=SC2046
      run_one latency_resources "$domain" C3-markov "$trial" CONFIG=C3 PREDICTOR=markov -- --fanout $(replay_tasks "$domain")
      # shellcheck disable=SC2046
      run_one latency_resources "$domain" C3-lightgbm_adapt "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt -- --fanout $(replay_tasks "$domain")
      # shellcheck disable=SC2046
      run_one latency_resources "$domain" C1 "$trial" CONFIG=C1 -- --fanout $(replay_tasks "$domain")
    done
  done
}

run_script experiments latency_resources
