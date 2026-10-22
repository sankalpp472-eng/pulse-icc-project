#!/usr/bin/env bash
# C0 on EKS: scale-to-zero tools, no prewarming.
#
#   bash EKS_deployment/run_c0.sh                       # task-9 trace x3 (see common.sh for WORKLOAD)
#   WORKLOAD=taubench bash EKS_deployment/run_c0.sh     # the 46 recorded tau-bench episodes
#   bash EKS_deployment/run_c0.sh --seed 7              # extra arguments go to w1_replay.py
#
# Result: test/test_knative/results/<date>_<time>_C0_W1-replay_<workload>_eks/
source "$(dirname "$0")/common.sh"

ensure_min_scale 0
replay "$@"
