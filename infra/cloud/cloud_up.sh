#!/usr/bin/env bash
# Deploy Pulse's own components into the current cluster (after eks_up.sh, or on kind for a test):
# the Groq-key Secret, RBAC, pulse-predictor (any of the four predictors), pulse-embed (pulse_v3's embedding
# server; its first start downloads the model, ~1 min) and pulse-controller.
#
#   bash infra/cloud/cloud_up.sh                     # EKS, image from build_image.sh
#   TARGET=kind bash infra/cloud/cloud_up.sh         # local kind cluster
#   THETA=0.5 bash infra/cloud/cloud_up.sh           # another prewarm threshold
#   PREDICTOR=markov bash infra/cloud/cloud_up.sh    # lightgbm (default) | markov | speculator | pulse_v3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=infra/cloud
TARGET=${TARGET:-ecr}
IMAGE=${IMAGE:-$(cat "$HERE/.image-$TARGET" 2>/dev/null || true)}
[ -n "$IMAGE" ] || { echo "No image yet: TARGET=$TARGET bash $HERE/build_image.sh"; exit 1; }
echo "Context: $(kubectl config current-context) | image: $IMAGE"

kubectl apply -f infra/prewarm/prewarmrequest-crd.yaml

# Groq / OpenRouter keys from .env -> Secret pulse-llm-keys (values never printed or written to the repo)
if [ -f .env ]; then
  TMP=$(mktemp)
  command -v cygpath >/dev/null && TMP=$(cygpath -m "$TMP")   # Git Bash on Windows: a path kubectl.exe can open
  trap 'rm -f "$TMP"' EXIT
  grep -E '^(GROQ_API_KEY(_[0-9]+)?|OPEN_ROUTER_API_KEY|LLM_BACKEND|SPEC_BASE_URL|SPEC_API_KEY|SPEC_MODEL)=' .env | tr -d '\r' > "$TMP"
  echo "Secret pulse-llm-keys: $(wc -l < "$TMP") keys/settings from .env"
  kubectl create secret generic pulse-llm-keys --from-env-file="$TMP" --dry-run=client -o yaml | kubectl apply -f -
  rm -f "$TMP"
else
  echo "No .env: the speculator and live runs will have no Groq keys"
fi

# The model registry the cluster's predictors load from (ConfigMap pulse-config): from the UI's registry settings
# (PULSE_MODEL_URI / ACTION_MODEL_URI). With GitOps, Argo CD owns pulse-config (infra/cloud/gitops/config.yaml): kept.
if kubectl get configmap pulse-config -o jsonpath='{.metadata.annotations.argocd\.argoproj\.io/tracking-id}' 2>/dev/null \
     | grep -q .; then
  echo "ConfigMap pulse-config: managed by Argo CD (GitOps), kept"
elif [ -n "${PULSE_MODEL_URI:-}${ACTION_MODEL_URI:-}" ]; then
  kubectl create configmap pulse-config --from-literal=PULSE_MODEL_URI="${PULSE_MODEL_URI:-}" \
    --from-literal=ACTION_MODEL_URI="${ACTION_MODEL_URI:-}" --dry-run=client -o yaml | kubectl apply -f -
  echo "ConfigMap pulse-config: the model registry (${PULSE_MODEL_URI:-no pulse-v3} · ${ACTION_MODEL_URI:-no action})"
fi

sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__THETA__|${THETA:-0.3}|g" -e "s|value: lightgbm}|value: ${PREDICTOR:-lightgbm}}|" \
  "$HERE/pulse-cloud.yaml" | kubectl apply -f -
kubectl delete deploy/pulse-speculator svc/pulse-speculator --ignore-not-found >/dev/null   # replaced by PREDICTOR=speculator
for d in pulse-predictor pulse-controller pulse-embed; do kubectl rollout status deploy/$d --timeout=600s; done
kubectl get pods -l pulse/role=system -o wide
