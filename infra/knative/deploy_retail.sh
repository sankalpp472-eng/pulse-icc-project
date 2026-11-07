#!/usr/bin/env bash
# The 16 retail tools next to the airline ones, on kind or EKS. The published tools image predates retail, so the
# image is built here from infra/knative/Dockerfile.mcp:
#   kind  loaded into the nodes as kind.local/pulse-mcp-tools:retail (a registry Knative never looks up)
#   EKS   pushed to this account's private ECR repository pulse-tools (the nodes may pull from ECR)
# then DOMAIN=retail infra/knative/deploy.sh: Redis DB 1 seeded with the retail data, services retail-<tool>.
#
#   bash infra/knative/deploy_retail.sh               # the current kubectl context (kind-pulse-kn or the EKS one)
#   KUBE_CONTEXT=kind-pulse-kn bash infra/knative/deploy_retail.sh
# The UI: Setup -> Local cluster / AWS EKS -> Deploy the retail tools.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.arkade/bin:$PATH"
CTX=${KUBE_CONTEXT:-$(kubectl config current-context)}
REGION=${REGION:-ap-south-1}
TAG=retail-$(date +%Y%m%d-%H%M)

case "$CTX" in
  kind-*)
    IMAGE=kind.local/pulse-mcp-tools:retail
    docker build -f infra/knative/Dockerfile.mcp -t "$IMAGE" .
    kind load docker-image "$IMAGE" --name "${CTX#kind-}"
    ;;
  *)
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
    IMAGE="$REGISTRY/pulse-tools:$TAG"
    aws ecr describe-repositories --region "$REGION" --repository-names pulse-tools >/dev/null 2>&1 || \
      aws ecr create-repository --region "$REGION" --repository-name pulse-tools >/dev/null
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
    docker build -f infra/knative/Dockerfile.mcp -t "$IMAGE" .
    docker push "$IMAGE"
    ;;
esac
echo "retail tools image: $IMAGE"
DOMAIN=retail SKIP_BUILD=1 IMAGE="$IMAGE" KUBE_CONTEXT="$CTX" bash infra/knative/deploy.sh 0
