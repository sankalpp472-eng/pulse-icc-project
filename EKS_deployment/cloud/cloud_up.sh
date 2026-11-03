#!/usr/bin/env bash
# Deploy Pulse's own components into the current cluster (after eks_up.sh, or on kind for a test):
# the Groq-key Secret, RBAC, pulse-predictor, pulse-speculator and pulse-controller.
#
#   bash EKS_deployment/cloud/cloud_up.sh                 # EKS, image from build_image.sh
#   TARGET=kind bash EKS_deployment/cloud/cloud_up.sh     # local kind cluster
#   THETA=0.5 bash EKS_deployment/cloud/cloud_up.sh       # another prewarm threshold
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=EKS_deployment/cloud
TARGET=${TARGET:-ecr}
IMAGE=${IMAGE:-$(cat "$HERE/.image-$TARGET" 2>/dev/null || true)}
[ -n "$IMAGE" ] || { echo "No image yet: TARGET=$TARGET bash $HERE/build_image.sh"; exit 1; }
echo "Context: $(kubectl config current-context) | image: $IMAGE"

kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml

# Groq / OpenRouter keys from agent/.env -> Secret pulse-llm-keys (values never printed or written to the repo)
if [ -f agent/.env ]; then
  TMP=$(mktemp)
  trap 'rm -f "$TMP"' EXIT
  grep -E '^(GROQ_API_KEY(_[0-9]+)?|OPEN_ROUTER_API_KEY|LLM_BACKEND)=' agent/.env | tr -d '\r' > "$TMP"
  echo "Secret pulse-llm-keys: $(wc -l < "$TMP") keys/settings from agent/.env"
  kubectl create secret generic pulse-llm-keys --from-env-file="$TMP" --dry-run=client -o yaml | kubectl apply -f -
  rm -f "$TMP"
else
  echo "No agent/.env: the speculator and live runs will have no Groq keys"
fi

sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__THETA__|${THETA:-0.3}|g" "$HERE/pulse-cloud.yaml" | kubectl apply -f -
kubectl rollout status deploy/pulse-predictor deploy/pulse-speculator deploy/pulse-controller --timeout=600s 2>/dev/null || {
  for d in pulse-predictor pulse-speculator pulse-controller; do kubectl rollout status deploy/$d --timeout=600s; done; }
kubectl get pods -l pulse/role=system -o wide
