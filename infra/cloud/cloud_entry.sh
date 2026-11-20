#!/usr/bin/env bash
# Entry point of a run pod (started by infra/cloud/run_cloud.sh):
#   cloud_entry.sh replay <pulse.experiments.replay args>     cloud_entry.sh live <pulse.experiments.live args>
#   cloud_entry.sh load <pulse.loadtest.async_load args>      (hints at fixed rates against the predictor's API)
#   cloud_entry.sh coldstart <pulse.experiments.coldstart args>   (cold and warm call to every tool, N times)
# Writes the run folder under $PULSE_RESULTS_DIR, then marks it done and waits (up to HOLD_S) for the
# laptop to copy it out with kubectl cp and create /tmp/collected; then the pod exits.
set -u
mode=${1:?replay, live, load or coldstart}; shift
mkdir -p "$PULSE_RESULTS_DIR"
cd /app
case "$mode" in
  replay)    python -m pulse.experiments.replay "$@" ;;
  live)      python -m pulse.experiments.live "$@" ;;
  load)      PULSE_LOAD_DIR="$PULSE_RESULTS_DIR" python -m pulse.loadtest.async_load "$@" ;;
  coldstart) python -m pulse.experiments.coldstart "$@" ;;
  *) echo "unknown mode $mode (replay, live, load or coldstart)"; exit 2 ;;
esac
echo $? > "$PULSE_RESULTS_DIR/.exit_code"
touch "$PULSE_RESULTS_DIR/.done"
echo "PULSE_RUN_DONE exit=$(cat "$PULSE_RESULTS_DIR/.exit_code")"
for _ in $(seq 1 $(( ${HOLD_S:-3600} / 5 ))); do
  [ -f /tmp/collected ] && exit 0
  sleep 5
done
echo "results were not collected within ${HOLD_S:-3600} s"
