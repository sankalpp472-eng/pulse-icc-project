#!/usr/bin/env bash
# Local kind cluster + Knative Serving + Kourier + the C0 test service.
# Needs: Docker running; kind and kubectl on PATH (arkade get kind --os mingw).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.arkade/bin:$PATH"

CLUSTER=pulse-kn
KN=knative-v1.23.0

# 1. Cluster
if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --config kind-config.yaml
fi
kubectl config use-context "kind-$CLUSTER"

# 2. Knative Serving
kubectl apply -f "https://github.com/knative/serving/releases/download/$KN/serving-crds.yaml"
kubectl wait --for=condition=Established --all crd --timeout=120s
kubectl apply -f "https://github.com/knative/serving/releases/download/$KN/serving-core.yaml"

# 3. Kourier as the networking layer, exposed on NodePort 31080 (-> localhost:8081)
kubectl apply -f "https://github.com/knative-extensions/net-kourier/releases/download/$KN/kourier.yaml"
kubectl patch configmap/config-network -n knative-serving --type merge \
  -p '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'
kubectl patch configmap/config-domain -n knative-serving --type merge \
  -p '{"data":{"example.com":""}}'
HTTP_PORT_IDX=$(kubectl get svc kourier -n kourier-system -o jsonpath='{range .spec.ports[*]}{.name}{"\n"}{end}' | grep -n '^http2$' | cut -d: -f1)
kubectl patch svc kourier -n kourier-system --type json -p "[
  {\"op\":\"replace\",\"path\":\"/spec/type\",\"value\":\"NodePort\"},
  {\"op\":\"add\",\"path\":\"/spec/ports/$((HTTP_PORT_IDX-1))/nodePort\",\"value\":31080}]"

kubectl wait --for=condition=Available deploy --all -n knative-serving --timeout=300s
kubectl wait --for=condition=Available deploy --all -n kourier-system --timeout=300s

# 4. The function (pulled from Docker Hub once, then cached on the node)
kubectl apply -f service.yaml
kubectl wait ksvc/cold-tool --for=condition=Ready --timeout=300s

echo
echo "Ready. Ingress: http://127.0.0.1:8081  (Host: cold-tool.default.example.com)"
echo "Run:   python c0_baseline.py --trials 10"
