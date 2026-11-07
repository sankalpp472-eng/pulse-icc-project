#!/usr/bin/env bash
# Give Pulse's pods on an eksctl cluster access to the S3 model registry (IRSA: short-lived credentials, no keys):
#   default/pulse          predictors, controller, run pods: read models    role pulse-eks-pulse          (pulse-model-reader)
#   default/pulse-trainer  the retraining Jobs (infra/training/): publish    role pulse-eks-pulse-trainer  (pulse-model-publisher)
# The same role names as the Terraform cluster's (infra/terraform/eks/irsa.tf), so the GitOps manifests
# (infra/cloud/gitops/) annotate the service accounts correctly either way. eksctl delete cluster removes them.
#
#   bash infra/eks/registry_access.sh            # eks_up.sh runs it; the UI: Setup -> AWS EKS -> Model registry access
#
# Needs the registry's policies (Setup -> Cloud extras -> Terraform apply); without them it says so and stops.
# Safe to rerun.
set -euo pipefail
REGION=${REGION:-ap-south-1}
NAME=${CLUSTER:-pulse-eks}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
READER=arn:aws:iam::$ACCOUNT:policy/pulse-model-reader
PUBLISHER=arn:aws:iam::$ACCOUNT:policy/pulse-model-publisher

if ! aws iam get-policy --policy-arn "$READER" >/dev/null 2>&1; then
  echo "No model registry in this AWS account yet (policy pulse-model-reader): create it first"
  echo "(Setup -> Cloud extras -> Terraform apply), then run this again. Without it the predictors use the models"
  echo "in the image."
  exit 0
fi

# the cluster's OIDC identity provider, which lets IAM trust the cluster's service-account tokens (once per cluster)
eksctl utils associate-iam-oidc-provider --cluster "$NAME" --region "$REGION" --approve

for pair in "pulse:$READER" "pulse-trainer:$PUBLISHER"; do
  sa=${pair%%:*}; policy=${pair#*:}
  eksctl create iamserviceaccount --cluster "$NAME" --region "$REGION" --namespace default --name "$sa" \
    --role-name "$NAME-$sa" --attach-policy-arn "$policy" --override-existing-serviceaccounts --approve
done

# pods pick up a new role only when they start: restart Pulse's services if they are already deployed
CTX=$(kubectl config get-contexts -o name | grep -m1 "$NAME" || true)
if [ -n "$CTX" ]; then
  for d in pulse-predictor pulse-controller; do
    kubectl --context "$CTX" get deploy "$d" >/dev/null 2>&1 && kubectl --context "$CTX" rollout restart deploy "$d"
  done
  kubectl --context "$CTX" get serviceaccount pulse pulse-trainer \
    -o custom-columns=ACCOUNT:.metadata.name,ROLE:.metadata.annotations.eks\\.amazonaws\\.com/role-arn
fi
echo "Model registry access: done (pods read with $NAME-pulse, training Jobs publish with $NAME-pulse-trainer)"
