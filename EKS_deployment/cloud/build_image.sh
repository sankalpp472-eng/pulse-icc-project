#!/usr/bin/env bash
# Build the runner image (EKS_deployment/cloud/Dockerfile) and make it available to the cluster.
#
#   bash EKS_deployment/cloud/build_image.sh              # EKS: push to the private Amazon ECR repo pulse-runner
#   TARGET=kind bash EKS_deployment/cloud/build_image.sh  # local kind cluster: load it into the nodes
#
# Needs Docker running; for ECR also `aws` logged in (same account as the EKS cluster: its nodes may pull
# from ECR). The image tag (date + git commit) is written to EKS_deployment/cloud/.image-<target>, where
# cloud_up.sh and run_cloud.sh read it.
# What goes in: the code the pods run, the trained models (predictor/data/*.pkl), the tau-bench airline
# trajectories (hint signals) and the replay traces. Never agent/.env: Groq keys go in as a Secret.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
TARGET=${TARGET:-ecr}
REGION=${REGION:-ap-south-1}
REPO=pulse-runner
HERE=EKS_deployment/cloud
STAGE=$HERE/.build

for f in predictor/data/model.pkl predictor/data/action_model.pkl \
         predictor/data/tau-bench/historical_trajectories/gpt-4o-airline.json \
         predictor/data/tau-bench/historical_trajectories/sonnet-35-new-airline.json; do
  [ -f "$f" ] || { echo "Missing $f (train the predictor / download the tau-bench trajectories first)"; exit 1; }
done

rm -rf "$STAGE"
mkdir -p "$STAGE/app/agent" "$STAGE/app/predictor/data/tau-bench/historical_trajectories" "$STAGE/app/controller" \
         "$STAGE/app/test/test_knative/data" "$STAGE/app/baselines/llm_speculator"
cp agent/{agent,hints,llm,prompts,tools,user}.py "$STAGE/app/agent/"
cp -r agent/tau_airline agent/tau_retail "$STAGE/app/agent/"
cp predictor/*.py "$STAGE/app/predictor/"
cp predictor/data/*.pkl "$STAGE/app/predictor/data/"
cp predictor/data/tau-bench/historical_trajectories/*airline.json "$STAGE/app/predictor/data/tau-bench/historical_trajectories/"
cp controller/prewarm_controller.py "$STAGE/app/controller/"
cp test/test_knative/{w1_replay,w1_live,c0_common}.py "$STAGE/app/test/test_knative/"
cp test/test_knative/data/*.jsonl "$STAGE/app/test/test_knative/data/"
cp baselines/llm_speculator/{service,common,prompt,speculate}.py "$STAGE/app/baselines/llm_speculator/"
cp "$HERE/cloud_entry.sh" "$STAGE/app/"
cp "$HERE/requirements.txt" "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name ".env*" -exec rm -f {} +          # belt and braces: no secrets in the image
echo "staged $(find "$STAGE/app" -type f | wc -l) files ($(du -sh "$STAGE" | cut -f1))"

COMMIT=$(git rev-parse --short HEAD)
DIRTY=$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)
TAG="$(date +%Y%m%d-%H%M)-$COMMIT$([ "$DIRTY" = 1 ] && echo -dirty || true)"

if [ "$TARGET" = kind ]; then
  IMAGE="$REPO:$TAG"
else
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
  REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
  IMAGE="$REGISTRY/$REPO:$TAG"
  aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 || \
    aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
fi

docker build -f "$HERE/Dockerfile" --build-arg GIT_COMMIT="$COMMIT" --build-arg GIT_DIRTY="$DIRTY" \
  --build-arg IMAGE_TAG="$IMAGE" -t "$IMAGE" "$STAGE"
if [ "$TARGET" = kind ]; then
  export PATH="$HOME/.arkade/bin:$PATH"
  kind load docker-image "$IMAGE" --name pulse-kn
else
  docker push "$IMAGE"
fi
echo "$IMAGE" > "$HERE/.image-$TARGET"
echo "Image: $IMAGE  (saved to $HERE/.image-$TARGET)"
