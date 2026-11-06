#!/usr/bin/env bash
# Delete the EKS cluster and everything in it (nodes, load balancer, VPC), ~10 min.
# Result folders on your laptop are kept. Afterwards, check the AWS console (ap-south-1:
# EKS, EC2 -> Instances and Load Balancers) that nothing is left running.
#
#   bash infra/eks/eks_down.sh
#   USE_TERRAFORM=1 bash infra/eks/eks_down.sh   # a cluster made by infra/terraform/eks/ (eks_up.sh USE_TERRAFORM=1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CTX=$(kubectl config get-contexts -o name | grep -m1 pulse-eks || true)
[ -n "$CTX" ] && kubectl --context "$CTX" delete pwr --all --ignore-not-found || true
if [ "${USE_TERRAFORM:-0}" = 1 ]; then
  # Kourier's load balancer was made by Kubernetes, not Terraform: remove it first, or the VPC can't be deleted
  [ -n "$CTX" ] && kubectl --context "$CTX" delete svc -n kourier-system kourier --ignore-not-found --wait || true
  TF_STATE_BUCKET=${TF_STATE_BUCKET:-pulse-tfstate-$(aws sts get-caller-identity --query Account --output text)}
  terraform -chdir=infra/terraform/eks init -input=false -backend-config="bucket=$TF_STATE_BUCKET" >/dev/null
  terraform -chdir=infra/terraform/eks destroy -input=false -auto-approve
else
  eksctl delete cluster -f infra/eks/cluster.yaml --wait
fi
kubectl config use-context kind-pulse-kn >/dev/null 2>&1 || true
echo
echo "Cluster deleted. In terminals where you sourced use_eks.sh, run: source infra/eks/use_local.sh"
