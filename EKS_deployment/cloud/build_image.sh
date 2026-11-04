#!/usr/bin/env bash
# Build the runner image (EKS_deployment/cloud/Dockerfile) and make it available to the cluster.
#
#   bash EKS_deployment/cloud/build_image.sh              # EKS: push to the private Amazon ECR repo pulse-runner
#   TARGET=kind bash EKS_deployment/cloud/build_image.sh  # local kind cluster: load it into the nodes
#   TARGET=local bash EKS_deployment/cloud/build_image.sh # build only (check the image / run it with docker)
#
# Needs Docker running; for ECR also `aws` logged in (same account as the EKS cluster: its nodes may pull
# from ECR). The image tag (date + git commit) is written to EKS_deployment/cloud/.image-<target>, where
# cloud_up.sh and run_cloud.sh read it.
# What goes in: the code the pods run, the trained models (predictor/data/*.pkl), the tau-bench airline
# trajectories (hint signals), the replay traces, the demo UI (ui/; pulse-ui pod, ui_cloud.sh) with the plot
# scripts, the measured per-tool cold starts and the result folders so far (read-only, for comparisons).
# Never agent/.env: Groq keys go in as a Secret.
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
cp test/test_knative/{w1_replay,w1_live,c0_common,plot_compare,plot_c0,score_hints,pulse_metrics}.py "$STAGE/app/test/test_knative/"
mkdir -p "$STAGE/app/loadtest"                                    # load tests (Locust / asyncio), run by the UI pod
cp loadtest/*.py "$STAGE/app/loadtest/"
# pulse-v3 (code only: its model comes from the S3 registry, its data/ stays out): the UI pod and the training Job
tar -cf - --exclude=data --exclude=__pycache__ --exclude=tests pulse_pred | tar -xf - -C "$STAGE/app"
cp test/test_knative/data/*.jsonl "$STAGE/app/test/test_knative/data/"
cp baselines/llm_speculator/{service,common,prompt,speculate}.py "$STAGE/app/baselines/llm_speculator/"
cp "$HERE/cloud_entry.sh" "$STAGE/app/"
# the demo UI (pulse-ui pod) + what it reads: measured cold starts (simulated cluster), earlier results (Results tab)
mkdir -p "$STAGE/app/ui/static" "$STAGE/app/deploy/knative/results" "$STAGE/app/results"
cp ui/{server,backend,session,sim_cluster}.py "$STAGE/app/ui/"
cp ui/static/* "$STAGE/app/ui/static/"
cp deploy/knative/results/*_tools-cold-warm.csv "$STAGE/app/deploy/knative/results/" 2>/dev/null || true
for d in cloud_runs local_runs; do if [ -d "results/$d" ]; then cp -r "results/$d" "$STAGE/app/results/"; fi; done
cp "$HERE/requirements.txt" "$STAGE/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name ".env*" -exec rm -f {} +          # belt and braces: no secrets in the image
echo "staged $(find "$STAGE/app" -type f | wc -l) files ($(du -sh "$STAGE" | cut -f1))"

COMMIT=$(git rev-parse --short HEAD)
DIRTY=$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)
TAG="$(date +%Y%m%d-%H%M)-$COMMIT$([ "$DIRTY" = 1 ] && echo -dirty || true)"

if [ "$TARGET" = kind ] || [ "$TARGET" = local ]; then
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
elif [ "$TARGET" != local ]; then
  docker push "$IMAGE"
fi
echo "$IMAGE" > "$HERE/.image-$TARGET"
echo "Image: $IMAGE  (saved to $HERE/.image-$TARGET)"
