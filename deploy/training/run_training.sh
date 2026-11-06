#!/usr/bin/env bash
# Retrain a model family in the EKS cluster: start its training Job and follow its log.
#
#   PULSE_MODEL_URI=s3://<bucket>/pulse-v3 bash deploy/training/run_training.sh                   # pulse-v3
#   MODEL=action ACTION_MODEL_URI=s3://<bucket>/action-model bash deploy/training/run_training.sh  # lightgbm + markov
#   PROMOTE=never NOTES="more data" VERSION=2026-11-05_1200 bash deploy/training/run_training.sh
#
# MODEL pulse_v3 (default): train-job.yaml (pulse_pred/train.py, with an embedding sidecar)
#       action:             train-action-job.yaml (predictor/train_action.py)
# Needs: the runner image (EKS_deployment/cloud/build_image.sh), the training data in the registry once
# (pulse-v3: python pulse_pred/model_store.py data push "$PULSE_MODEL_URI"; action: ... data push "$ACTION_MODEL_URI"
# --local predictor/data --files action_rows.csv), and the service account pulse-trainer with S3 publish rights
# (checked below). Prints the gate's verdict at the end ("GATE: ...").
set -euo pipefail
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/../.."
MODEL=${MODEL:-pulse_v3}
case "$MODEL" in
  pulse_v3) TEMPLATE=deploy/training/train-job.yaml
            URI=${PULSE_MODEL_URI:?set PULSE_MODEL_URI=s3://<bucket>/pulse-v3 (terraform -chdir=terraform output -raw model_uri)} ;;
  action)   TEMPLATE=deploy/training/train-action-job.yaml
            URI=${ACTION_MODEL_URI:?set ACTION_MODEL_URI=s3://<bucket>/action-model (terraform -chdir=terraform output -raw action_model_uri)} ;;
  *) echo "Unknown MODEL=$MODEL (pulse_v3 or action)"; exit 1 ;;
esac
IMAGE=${IMAGE:-$(cat EKS_deployment/cloud/.image-ecr 2>/dev/null || true)}
[ -n "$IMAGE" ] || { echo "No image: bash EKS_deployment/cloud/build_image.sh first"; exit 1; }
VERSION=${VERSION:-$(date +%Y-%m-%d_%H%M)}
JOB="pulse-train-$([ "$MODEL" = action ] && echo "action-")$(echo "$VERSION" | tr '_A-Z' '-a-z')"

if ! kubectl get serviceaccount pulse-trainer >/dev/null 2>&1; then
  echo "No service account pulse-trainer. Create it once per cluster, bound to the publisher policy:"
  echo "  eksctl create iamserviceaccount --cluster <cluster> --region ap-south-1 --namespace default \\"
  echo "    --name pulse-trainer --attach-policy-arn \$(terraform -chdir=terraform output -raw model_publisher_policy_arn) --approve"
  echo "(or, with the Terraform cluster + GitOps: terraform/eks/irsa.tf makes the role, deploy/gitops/ the account)"
  exit 1
fi

sed -e "s|__IMAGE__|$IMAGE|; s|__URI__|$URI|; s|__PROMOTE__|${PROMOTE:-auto}|; s|__NOTES__|${NOTES:-}|" \
    -e "s|pulse-train-__VERSION__|$JOB|; s|__VERSION__|$VERSION|" "$TEMPLATE" | kubectl apply -f -
echo "job $JOB: waiting for it to start$([ "$MODEL" = pulse_v3 ] && echo " (the embedding sidecar downloads its model first)") ..."
kubectl wait --for=condition=Ready "pod" -l "job-name=$JOB" --timeout=900s >/dev/null 2>&1 || true
kubectl logs -f "job/$JOB" -c train
if kubectl wait --for=condition=complete "job/$JOB" --timeout=60s >/dev/null 2>&1; then
  echo "training job $JOB finished"
else
  echo "training job $JOB failed: kubectl describe job/$JOB"; exit 1
fi
