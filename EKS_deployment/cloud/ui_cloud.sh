#!/usr/bin/env bash
# The demo UI inside the EKS cluster: the laptop only opens a browser tab through kubectl port-forward.
#
#   bash EKS_deployment/cloud/ui_cloud.sh up      # deploy / update the pulse-ui pod, then open it (port-forward)
#   bash EKS_deployment/cloud/ui_cloud.sh open    # open it again (port-forward only), e.g. after a disconnect
#   bash EKS_deployment/cloud/ui_cloud.sh fetch   # copy the runs made in the UI to results/cloud_runs/
#   bash EKS_deployment/cloud/ui_cloud.sh down    # remove the UI (fetch first: its result folder goes with it)
#
# Needs eks_up.sh and cloud_up.sh done (the Secret, RBAC, pulse-controller) and the image from build_image.sh
# (rebuild it after changing ui/ or any other code). Open http://127.0.0.1:8765 while the port-forward runs.
set -euo pipefail
export MSYS_NO_PATHCONV=1    # Git Bash would rewrite the pod paths (/home/app/...)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=EKS_deployment/cloud
PORT=${UI_PORT:-8765}
RDIR=/home/app/results/cloud_runs
CMD=${1:-up}

open_ui() {
  echo "Pulse UI: http://127.0.0.1:$PORT   (Ctrl+C to close the tunnel; the UI keeps running in the cluster)"
  until kubectl port-forward svc/pulse-ui "$PORT:8765"; do   # reconnect if the tunnel drops
    echo "port-forward ended; reconnecting in 3 s (Ctrl+C to stop)"
    sleep 3
  done
}

case "$CMD" in
  up)
    IMAGE=${IMAGE:-$(cat "$HERE/.image-ecr" 2>/dev/null || true)}
    [ -n "$IMAGE" ] || { echo "No image: bash $HERE/build_image.sh first"; exit 1; }
    kubectl get secret pulse-llm-keys >/dev/null 2>&1 || echo "Note: no Secret pulse-llm-keys (cloud_up.sh): live episodes need Groq keys"
    kubectl get deploy pulse-controller >/dev/null 2>&1 || { echo "No pulse-controller: run $HERE/cloud_up.sh first"; exit 1; }
    PLACE=""
    if [ -n "$(kubectl get nodes -l pulse/role=client -o name)" ]; then   # next to the run pods, away from the tools
      PLACE='      nodeSelector: {pulse/role: client}
      tolerations: [{key: pulse/role, operator: Equal, value: client, effect: NoSchedule}]'
    fi
    TMP=$(mktemp)
    awk -v img="$IMAGE" -v place="$PLACE" '{ gsub("__IMAGE__", img); if ($0 == "__PLACE__") print place; else print }' \
      "$HERE/pulse-ui.yaml" > "$TMP"
    kubectl apply -f "$TMP"
    rm -f "$TMP"
    kubectl rollout status deploy/pulse-ui --timeout=600s
    open_ui ;;
  open)
    open_ui ;;
  fetch)
    POD=$(kubectl get pod -l app=pulse-ui -o jsonpath='{.items[0].metadata.name}')
    kubectl exec "$POD" -- test -d "$RDIR" || { echo "No runs in the UI pod yet"; exit 0; }
    mkdir -p results/cloud_runs
    INCOMING="_incoming_ui"
    ( cd results/cloud_runs && rm -rf "$INCOMING" && kubectl cp "default/$POD:$RDIR" "$INCOMING" >/dev/null )
    n=0
    for d in results/cloud_runs/$INCOMING/*/; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      if [ -e "results/cloud_runs/$name" ]; then echo "  kept (already here): $name"; else mv "$d" "results/cloud_runs/$name"; n=$((n + 1)); echo "  + $name"; fi
    done
    rm -rf "results/cloud_runs/$INCOMING"
    echo "$n new folder(s) in results/cloud_runs/" ;;
  down)
    kubectl delete deploy/pulse-ui svc/pulse-ui --ignore-not-found ;;
  *)
    echo "usage: $0 up|open|fetch|down"; exit 1 ;;
esac
