#!/usr/bin/env bash
# Bring up everything on EKS in one go: cluster (if missing), Knative + Kourier (load balancer open
# to your current IP only), Redis + the 14 tools at scale-to-zero, the PrewarmRequest CRD, and a
# cold/warm check of every tool (also caches the tool image on the nodes).
# Safe to rerun, e.g. after your IP changes (it reapplies the IP restriction).
#
#   bash infra/eks/eks_up.sh          # ~25 min the first time, ~5 min when the cluster exists
#   USE_TERRAFORM=1 bash infra/eks/eks_up.sh   # the cluster from infra/terraform/eks/ instead of eksctl (same nodes)
#
# Needs: aws (logged in, `aws configure`), eksctl (or terraform), kubectl, and the project's Python packages.
# The cluster costs money by the hour until you run infra/eks/eks_down.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python}
REGION=ap-south-1
NAME=pulse-eks
CLUSTER_TOOL=$([ "${USE_TERRAFORM:-0}" = 1 ] && echo terraform || echo eksctl)

for t in aws "$CLUSTER_TOOL" kubectl curl; do
  command -v "$t" >/dev/null || { echo "$t is not installed (see infra/eks/README.md, step 1)"; exit 1; }
done
aws sts get-caller-identity >/dev/null || { echo "Not logged in to AWS: run aws configure"; exit 1; }

if [ "$CLUSTER_TOOL" = terraform ]; then        # infrastructure as code: infra/terraform/eks/ (README there)
  TF_STATE_BUCKET=${TF_STATE_BUCKET:-pulse-tfstate-$(aws sts get-caller-identity --query Account --output text)}
  terraform -chdir=infra/terraform/eks init -input=false -backend-config="bucket=$TF_STATE_BUCKET" >/dev/null
  echo "Creating / updating cluster $NAME with Terraform (15-20 min the first time) ..."
  terraform -chdir=infra/terraform/eks apply -input=false -auto-approve
  aws eks update-kubeconfig --region "$REGION" --name "$NAME" >/dev/null
elif eksctl get cluster --name "$NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "Cluster $NAME exists"
  aws eks update-kubeconfig --region "$REGION" --name "$NAME" >/dev/null
else
  echo "Creating cluster $NAME (15-20 min) ..."
  eksctl create cluster -f infra/eks/cluster.yaml
fi
CTX=$(kubectl config get-contexts -o name | grep -m1 "$NAME")
kubectl config use-context "$CTX"
kubectl get nodes

ALLOW_CIDR="$(curl -s https://checkip.amazonaws.com)/32" KUBE_CONTEXT="$CTX" bash infra/eks/setup_knative.sh

KUBE_CONTEXT="$CTX" SKIP_BUILD=1 bash infra/knative/deploy.sh 0
kubectl apply -f infra/prewarm/prewarmrequest-crd.yaml

source infra/eks/use_eks.sh
echo "Waiting for the load balancer's DNS name to resolve ..."
LB_HOST=${MCP_URL_TEMPLATE#http://}; LB_HOST=${LB_HOST%%/*}
for i in $(seq 1 60); do
  curl -s -o /dev/null --max-time 5 "http://$LB_HOST/" && break
  sleep 10
done
"$PY" -m pulse.agent.check_tools
if [ "${USE_GITOPS:-0}" = 1 ]; then   # Argo CD then deploys Pulse's services and monitoring from Git (infra/gitops/)
  bash infra/gitops/install.sh
fi

echo
echo "EKS is ready. In the UI (python -m pulse.ui.server): Setup -> Deploy Pulse services, then Knative + Experiments."
echo "Or: bash infra/cloud/cloud_up.sh, then CONFIG=C3 bash infra/cloud/run_cloud.sh --fanout"
echo "When done for the day: bash infra/eks/eks_down.sh"
