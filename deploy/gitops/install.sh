#!/usr/bin/env bash
# GitOps for the EKS cluster: install Argo CD and point it at this repository (apps.yaml). From then on the cluster
# follows Git: merge a change to EKS_deployment/cloud or deploy/monitoring and Argo CD applies it.
#
#   bash deploy/gitops/install.sh                    # after eks_up.sh (any cluster: eksctl or Terraform)
#   GITOPS_REVISION=my-branch bash deploy/gitops/install.sh   # follow a branch instead of main
#
# Before the first time: bash deploy/gitops/configure.sh (fills in the AWS account), commit, push.
# Needs kubectl and helm (for the monitoring chart's version). Argo CD's own UI:
#   kubectl port-forward -n argocd svc/argocd-server 18080:80   -> http://127.0.0.1:18080  (the Pulse UI does this)
#   user admin, password: kubectl get secret -n argocd argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
# A private repository also needs read access for Argo CD: argocd repo add, or a repository Secret (Argo CD docs).
set -euo pipefail
cd "$(dirname "$0")/../.."
ARGOCD_VERSION=${ARGOCD_VERSION:-v2.13.0}
REPO=${GITOPS_REPO:-$(git remote get-url origin)}
REVISION=${GITOPS_REVISION:-main}

if grep -rq ACCOUNT_ID EKS_deployment/cloud/kustomization.yaml EKS_deployment/cloud/gitops/; then
  echo "EKS_deployment/cloud still says ACCOUNT_ID: run bash deploy/gitops/configure.sh, commit and push first"; exit 1
fi
command -v helm >/dev/null || { echo "helm not found (needed for the monitoring chart's version)"; exit 1; }
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update >/dev/null
CHART_VERSION=${CHART_VERSION:-$(helm show chart prometheus-community/kube-prometheus-stack | sed -n 's/^version: //p')}

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd --server-side --force-conflicts \
  -f "https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/install.yaml"
# plain HTTP inside the cluster (reached only through port-forward), so the Pulse UI can link to it
kubectl patch configmap argocd-cmd-params-cm -n argocd --type merge -p '{"data":{"server.insecure":"true"}}'
kubectl rollout restart deploy/argocd-server -n argocd
kubectl rollout status deploy/argocd-server -n argocd --timeout=300s

sed -e "s|__REPO__|$REPO|g; s|__REVISION__|$REVISION|g; s|__CHART_VERSION__|$CHART_VERSION|g" deploy/gitops/apps.yaml |
  kubectl apply -f -
echo
echo "Argo CD $ARGOCD_VERSION follows $REPO @ $REVISION (monitoring chart $CHART_VERSION)"
kubectl get applications -n argocd
