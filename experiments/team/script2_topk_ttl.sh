#!/usr/bin/env bash
# Script 2: top-k sweep + adaptive top-k / adaptive k, and the TTL sweep (C3 lightgbm_adapt, airline).
#
#   TEAM_MEMBER=<your name> bash experiments/team/script2_topk_ttl.sh
#
# Experiments:
#   topk_adaptive  how many tools a step prewarms: fixed top-k (PREWARM_POLICY=topk, TOP_K 1 / 2 / 3) vs adaptive
#                  top-k (the default: the predictor offers its top 3, the controller warms those whose hit rate is
#                  worth a slot: hit x cold >= (1 - hit) x price) vs adaptive top-k + adaptive k (K_ADAPT=1: the k
#                  rule learns its warm time and budget from Prometheus)
#   ttl            how long a prediction keeps its tool warm (FR-7, issue #37): TTL_MODE=fixed with TTL_S 30 / 60 /
#                  120 s. The tools' idle window is 20 s for these runs (the window is the shortest possible TTL;
#                  with the usual 60 s a 30 s TTL could not differ from 60 s). The controller keeps a tool warm past
#                  the window with keep-alive pings (TTL_KEEPALIVE=1, its default)
# Workload: the airline held-out replay (tasks 6 20 22 26 30, 46 episodes, fan-out, seed 252); DOMAINS="airline
# retail" adds retail. TRIALS (3) x (5 + 3) arms, ~25 min each: about 10 h, plus about 1 h of set-up.
SCRIPT=script2_topk_ttl
source "$(dirname "$0")/lib.sh"
DOMAINS=${DOMAINS:-airline}

experiments() {
  local domain trial k ttl
  for domain in $DOMAINS; do
    for trial in $(seq 1 "$TRIALS"); do
      if wanted topk_adaptive; then
        for k in 1 2 3; do
          # shellcheck disable=SC2046
          run_one topk_adaptive "$domain" "top$k" "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt PREWARM_POLICY=topk \
            TOP_K="$k" -- --fanout $(replay_tasks "$domain")
        done
        # shellcheck disable=SC2046
        run_one topk_adaptive "$domain" adaptive "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt PREWARM_POLICY=adaptive \
          -- --fanout $(replay_tasks "$domain")
        # shellcheck disable=SC2046
        run_one topk_adaptive "$domain" adaptive-kadapt "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt \
          PREWARM_POLICY=adaptive K_ADAPT=1 -- --fanout $(replay_tasks "$domain")
      fi
      if wanted ttl; then
        for ttl in 30 60 120; do
          # shellcheck disable=SC2046
          run_one ttl "$domain" "ttl$ttl" "$trial" CONFIG=C3 PREDICTOR=lightgbm_adapt WINDOW=20s \
            PREDICTOR_ENV="TTL_MODE=fixed TTL_S=$ttl" CONTROLLER_ENV="TTL_KEEPALIVE=1" -- --fanout $(replay_tasks "$domain")
        done
      fi
    done
  done
}

run_script experiments topk_adaptive ttl
