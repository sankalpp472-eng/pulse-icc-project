#!/usr/bin/env bash
# Delete the EKS cluster and everything in it (nodes, load balancer, VPC), ~10 min.
# Result folders on your laptop are kept. Afterwards, check the AWS console (ap-south-1:
# EKS, EC2 -> Instances and Load Balancers) that nothing is left running.
#
#   bash EKS_deployment/eks_down.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CTX=$(kubectl config get-contexts -o name | grep -m1 pulse-eks || true)
# Runs made in the demo UI live only in its pod: copy them home before the cluster goes
if [ -n "$CTX" ] && kubectl --context "$CTX" get deploy pulse-ui >/dev/null 2>&1; then
  echo "Copying the demo UI's runs to results/cloud_runs/ first ..."
  kubectl config use-context "$CTX" >/dev/null
  bash EKS_deployment/cloud/ui_cloud.sh fetch || echo "(could not fetch the UI's runs; continuing)"
fi
[ -n "$CTX" ] && kubectl --context "$CTX" delete pwr --all --ignore-not-found || true
eksctl delete cluster -f EKS_deployment/cluster.yaml --wait
kubectl config use-context kind-pulse-kn >/dev/null 2>&1 || true
echo
echo "Cluster deleted. In terminals where you sourced use_eks.sh, run: source EKS_deployment/use_local.sh"
