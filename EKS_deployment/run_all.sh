#!/usr/bin/env bash
# C0, C3 and C1 on EKS with the same workload and seed, then the comparison plots.
#
#   bash EKS_deployment/run_all.sh                      # task-9 trace x3 per config (~30-40 min)
#   WORKLOAD=taubench bash EKS_deployment/run_all.sh    # 46 tau-bench episodes per config (~1.5-2 h)
#
# Result: three run folders + results/cloud_runs/compare_<date>_<workload>_C0-C3-C1_eks/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$HERE/../results/cloud_runs"
newest() { ls -td "$RESULTS"/*_"$1"_W1-replay_* | head -1; }

bash "$HERE/run_c0.sh" "$@"; C0=$(newest C0)
bash "$HERE/run_c3.sh" "$@"; C3=$(newest C3)
bash "$HERE/run_c1.sh" "$@"; C1=$(newest C1)

OUT="$RESULTS/compare_$(date +%Y-%m-%d)_${WORKLOAD:-task9}_C0-C3-C1_${LABEL:-eks}"
"${PYTHON:-python}" "$HERE/../test/test_knative/plot_compare.py" "$C0" "$C3" "$C1" --out "$OUT"
echo
echo "Runs:  $C0"
echo "       $C3"
echo "       $C1"
echo "Plots: $OUT"
