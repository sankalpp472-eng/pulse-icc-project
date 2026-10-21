#!/usr/bin/env bash
# Install Knative Serving + Kourier on the EKS cluster (the EKS version of test/test_knative/setup.sh).
# Run AFTER `eksctl create cluster -f EKS_deployment/cluster.yaml`, from the project root:
#
#   ALLOW_CIDR="$(curl -s https://checkip.amazonaws.com)/32" bash EKS_deployment/setup_knative.sh
#
# Differences from the local setup: no kind cluster to create, and Kourier gets an AWS load balancer
# (a public hostname) instead of a NodePort on localhost:8081.
# ALLOW_CIDR (recommended): only this IP range may reach the load balancer; without it the tools
# are open to the whole internet.
set -euo pipefail
cd "$(dirname "$0")"
KN=knative-v1.23.0
CONTEXT=${KUBE_CONTEXT:-$(kubectl config current-context)}
kubectl config use-context "$CONTEXT"
echo "Installing Knative $KN on context: $CONTEXT"
case "$CONTEXT" in *kind-*) echo "This looks like the LOCAL kind cluster, not EKS. Aborting."; exit 1;; esac

# 1. Knative Serving
kubectl apply -f "https://github.com/knative/serving/releases/download/$KN/serving-crds.yaml"
kubectl wait --for=condition=Established --all crd --timeout=180s
kubectl apply -f "https://github.com/knative/serving/releases/download/$KN/serving-core.yaml"

# 2. Kourier, the networking layer. On EKS its Service is type LoadBalancer -> an AWS ELB.
kubectl apply -f "https://github.com/knative-extensions/net-kourier/releases/download/$KN/kourier.yaml"
kubectl patch configmap/config-network -n knative-serving --type merge \
  -p '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'
kubectl patch configmap/config-domain -n knative-serving --type merge \
  -p '{"data":{"example.com":""}}'
if [ -n "${ALLOW_CIDR:-}" ]; then
  kubectl patch svc kourier -n kourier-system --type merge \
    -p "{\"spec\":{\"loadBalancerSourceRanges\":[\"$ALLOW_CIDR\"]}}"
  echo "Load balancer restricted to $ALLOW_CIDR"
else
  echo "WARNING: ALLOW_CIDR not set -> the tools are reachable from the whole internet"
fi

kubectl wait --for=condition=Available deploy --all -n knative-serving --timeout=600s
kubectl wait --for=condition=Available deploy --all -n kourier-system --timeout=600s

# 3. Wait for AWS to hand out the load balancer's hostname (1-3 min)
echo "Waiting for the Kourier load balancer hostname..."
for i in $(seq 1 60); do
  LB=$(kubectl get svc kourier -n kourier-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)
  [ -n "$LB" ] && break
  sleep 5
done
[ -n "${LB:-}" ] || { echo "No load balancer hostname after 5 min: kubectl describe svc kourier -n kourier-system"; exit 1; }
echo
echo "Knative ready. Ingress: http://$LB   (DNS for a new ELB can take a few more minutes to resolve)"
echo "Next: KUBE_CONTEXT=$CONTEXT SKIP_BUILD=1 bash deploy/knative/deploy.sh"
