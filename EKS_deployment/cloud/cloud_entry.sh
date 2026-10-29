#!/usr/bin/env bash
# Entry point of a run pod (started by EKS_deployment/cloud/run_cloud.sh):
#   cloud_entry.sh replay <w1_replay.py args>     cloud_entry.sh live <w1_live.py args>
# Writes the run folder under $PULSE_RESULTS_DIR, then marks it done and waits (up to HOLD_S) for the
# laptop to copy it out with kubectl cp and create /tmp/collected; then the pod exits.
set -u
mode=${1:?replay or live}; shift
mkdir -p "$PULSE_RESULTS_DIR"
cd /app
case "$mode" in
  replay) python test/test_knative/w1_replay.py "$@" ;;
  live)   python test/test_knative/w1_live.py "$@" ;;
  *) echo "unknown mode $mode (replay or live)"; exit 2 ;;
esac
echo $? > "$PULSE_RESULTS_DIR/.exit_code"
touch "$PULSE_RESULTS_DIR/.done"
echo "PULSE_RUN_DONE exit=$(cat "$PULSE_RESULTS_DIR/.exit_code")"
for _ in $(seq 1 $(( ${HOLD_S:-3600} / 5 ))); do
  [ -f /tmp/collected ] && exit 0
  sleep 5
done
echo "results were not collected within ${HOLD_S:-3600} s"
