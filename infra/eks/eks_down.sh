#!/usr/bin/env bash
# Delete the EKS cluster and everything in it (nodes, load balancer, VPC), ~10 min.
# Result folders on your laptop are kept. Afterwards check that nothing is left running (the UI's Setup tab:
# "Check for leftovers", or the AWS console: EKS, EC2 -> Instances and Load Balancers, in the cluster's region).
#
#   bash infra/eks/eks_down.sh
#   REGION=us-east-1 bash infra/eks/eks_down.sh   # the region eks_up.sh used (default ap-south-1)
#   USE_TERRAFORM=1 bash infra/eks/eks_down.sh   # a cluster made by infra/terraform/eks/ (eks_up.sh USE_TERRAFORM=1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
REGION=${REGION:-ap-south-1}

CTX=$(kubectl config get-contexts -o name | grep -m1 pulse-eks || true)
[ -n "$CTX" ] && kubectl --context "$CTX" delete pwr --all --ignore-not-found || true
if [ "${USE_TERRAFORM:-0}" = 1 ]; then
  # Kourier's load balancer was made by Kubernetes, not Terraform: remove it first, or the VPC can't be deleted
  [ -n "$CTX" ] && kubectl --context "$CTX" delete svc -n kourier-system kourier --ignore-not-found --wait || true
  TF_STATE_BUCKET=${TF_STATE_BUCKET:-pulse-tfstate-$(aws sts get-caller-identity --query Account --output text)}
  terraform -chdir=infra/terraform/eks init -input=false -backend-config="bucket=$TF_STATE_BUCKET" >/dev/null
  terraform -chdir=infra/terraform/eks destroy -input=false -auto-approve
else
  eksctl delete cluster --name pulse-eks --region "$REGION" --wait
fi
kubectl config use-context kind-pulse-kn >/dev/null 2>&1 || true
echo
echo "Cluster deleted. In terminals where you sourced use_eks.sh, run: source infra/eks/use_local.sh"
