#!/usr/bin/env bash
# C1 on EKS: one always-warm replica per tool. Switches the tools back to scale-to-zero (C0) at the end.
#
#   bash EKS_deployment/run_c1.sh                       # task-9 trace x3 (see common.sh for WORKLOAD)
#   WORKLOAD=taubench bash EKS_deployment/run_c1.sh
#
# Result: results/cloud_runs/<date>_<time>_C1_W1-replay_<workload>_eks/
source "$(dirname "$0")/common.sh"

back_to_c0() { echo "Switching the tools back to scale-to-zero ..."; ensure_min_scale 0; }
trap 'back_to_c0; stop_background' EXIT

ensure_min_scale 1
replay "$@"
