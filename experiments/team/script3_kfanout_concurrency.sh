#!/usr/bin/env bash
# Script 3: k fan-out (the k rule vs the predictor's static k) and concurrency (airline).
#
#   TEAM_MEMBER=<your name> bash experiments/team/script3_kfanout_concurrency.sh
#
# Experiments:
#   k_fanout     how many pods a prediction gets: the k rule (K_RULE=1, the default: m = min(k, ceil(k x warm_s /
#                budget_s)), pulse/controller/leases.py) vs the predictor's own k unchanged (K_RULE=0: "static" k =
#                the pending-ID count, capped at 5). C3 lightgbm_adapt (markov has no fan-out: always k = 1).
#                With ~45 ms tools the rule gives 1 pod where the static k gives k, so the comparison is fewer pods
#                vs the latency of the fan-out bursts
#   concurrency  load: new episodes per minute 3 / 7 / 14 (69 / 161 / 322 episodes, so every run lasts ~23 min):
#                about 4 / 10 / 20 agents at once (the PRD's E5: 5 / 10 / 20; at the usual 2 per minute it is ~3,
#                measured; calls.csv records concurrent_episodes) for C0 and C3 lightgbm_adapt
# Workload: the airline held-out replay (tasks 6 20 22 26 30, fan-out, seed 252); DOMAINS="airline retail" adds
# retail. TRIALS (3) x (2 + 6) arms, ~25 min each: about 10 h, plus about 1 h of set-up.
SCRIPT=script3_kfanout_concurrency
source "$(dirname "$0")/lib.sh"
DOMAINS=${DOMAINS:-airline}
RATES=${RATES:-3 7 14}

experiments() {
  local domain trial rate eps
  for domain in $DOMAINS; do
    for trial in $(seq 1 "$TRIALS"); do
      if wanted k_fanout; then
        # shellcheck disable=SC2046
        run_one k_fanout "$domain" k-rule "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt CONTROLLER_ENV="K_RULE=1" \
          -- --fanout $(replay_tasks "$domain")
        # shellcheck disable=SC2046
        run_one k_fanout "$domain" k-static "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt CONTROLLER_ENV="K_RULE=0" \
          -- --fanout $(replay_tasks "$domain")
      fi
      if wanted concurrency; then
        for rate in $RATES; do
          eps=$(( 46 * rate / 2 ))
          run_one concurrency "$domain" "C0-rate$rate" "$trial" CONFIG=C0 \
            -- --fanout --tasks $([ "$domain" = retail ] && echo "$RETAIL_REPLAY" || echo "$AIRLINE_REPLAY") \
            --episodes "$eps" --rate "$rate"
          run_one concurrency "$domain" "C3-lightgbm_adapt-rate$rate" "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt \
            -- --fanout --tasks $([ "$domain" = retail ] && echo "$RETAIL_REPLAY" || echo "$AIRLINE_REPLAY") \
            --episodes "$eps" --rate "$rate"
        done
      fi
    done
  done
}

run_script experiments k_fanout concurrency
