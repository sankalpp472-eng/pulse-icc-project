#!/usr/bin/env bash
# Bring up everything on EKS in one go: cluster (if missing), Knative + Kourier (load balancer open
# to your current IP only), Redis + the 14 tools at scale-to-zero, the PrewarmRequest CRD, and a
# cold/warm check of every tool (also caches the tool image on the nodes).
# Safe to rerun, e.g. after your IP changes (it reapplies the IP restriction).
#
#   bash EKS_deployment/eks_up.sh          # ~25 min the first time, ~5 min when the cluster exists
#
# Needs: aws (logged in, `aws configure`), eksctl, kubectl, and the project's Python packages.
# The cluster costs money by the hour until you run EKS_deployment/eks_down.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python}
REGION=ap-south-1
NAME=pulse-eks

for t in aws eksctl kubectl curl; do
  command -v "$t" >/dev/null || { echo "$t is not installed (see EKS_deployment/README.md, step 1)"; exit 1; }
done
aws sts get-caller-identity >/dev/null || { echo "Not logged in to AWS: run aws configure"; exit 1; }

if eksctl get cluster --name "$NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "Cluster $NAME exists"
  aws eks update-kubeconfig --region "$REGION" --name "$NAME" >/dev/null
else
  echo "Creating cluster $NAME (15-20 min) ..."
  eksctl create cluster -f EKS_deployment/cluster.yaml
fi
CTX=$(kubectl config get-contexts -o name | grep -m1 "$NAME")
kubectl config use-context "$CTX"
kubectl get nodes

ALLOW_CIDR="$(curl -s https://checkip.amazonaws.com)/32" KUBE_CONTEXT="$CTX" bash EKS_deployment/setup_knative.sh

KUBE_CONTEXT="$CTX" SKIP_BUILD=1 bash deploy/knative/deploy.sh 0
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml

source EKS_deployment/use_eks.sh
echo "Waiting for the load balancer's DNS name to resolve ..."
LB_HOST=${MCP_URL_TEMPLATE#http://}; LB_HOST=${LB_HOST%%/*}
for i in $(seq 1 60); do
  curl -s -o /dev/null --max-time 5 "http://$LB_HOST/" && break
  sleep 10
done
"$PY" deploy/knative/check_tools.py

echo
echo "EKS is ready. Run: bash EKS_deployment/run_all.sh   (or run_c0.sh / run_c3.sh / run_c1.sh)"
echo "When done for the day: bash EKS_deployment/eks_down.sh"
