#!/usr/bin/env bash
# Local kind cluster + OpenFaaS CE + one test function.
# Needs: Docker running; kind, helm, faas-cli, kubectl on PATH (arkade get kind helm faas-cli).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.arkade/bin:$PATH"

CLUSTER=pulse
IMAGE=sankalps2003/pulse-cold-tool:0.2   # CE only allows public images (Docker Hub)

# 1. Cluster
if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --config kind-config.yaml
fi
kubectl config use-context "kind-$CLUSTER"

# 2. OpenFaaS CE
kubectl apply -f https://raw.githubusercontent.com/openfaas/faas-netes/master/namespaces.yml
helm repo add openfaas https://openfaas.github.io/faas-netes/ >/dev/null 2>&1 || true
helm repo update openfaas >/dev/null
helm upgrade --install openfaas openfaas/openfaas -n openfaas \
  --set functionNamespace=openfaas-fn \
  --set generateBasicAuth=true \
  --set serviceType=NodePort \
  --set gateway.scaleFromZero=true \
  --set functions.imagePullPolicy=IfNotPresent \
  --wait --timeout 5m

# 3. Function image: build, push (public, required by CE), and side-load into kind
#    so cold starts never include an image pull. Needs `docker login` as sankalps2003.
docker build -t "$IMAGE" fn/
docker push "$IMAGE"
kind load docker-image "$IMAGE" --name "$CLUSTER"

# 4. Deploy
PASSWORD=$(kubectl get secret -n openfaas basic-auth -o jsonpath='{.data.basic-auth-password}' | base64 -d)
until curl -sf http://127.0.0.1:8080/healthz >/dev/null; do sleep 2; done
echo -n "$PASSWORD" | faas-cli login -g http://127.0.0.1:8080 -u admin --password-stdin
faas-cli deploy -f stack.yml
kubectl rollout status -n openfaas-fn deploy/cold-tool --timeout=120s

echo
echo "Ready. Gateway: http://127.0.0.1:8080  (user admin, password in secret openfaas/basic-auth)"
echo "Run:   python cold_start.py --trials 20"
