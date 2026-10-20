#!/usr/bin/env bash
# Deploy the shared database (Redis) and the 14 MCP tools as Knative services
# (optionally building + pushing the image first).
# Needs: the Knative cluster from test/test_knative/setup.sh running.
#
#   SKIP_BUILD=1 bash deploy.sh      # use the existing public image (teammates: use this)
#   bash deploy.sh                   # build + push the image first (needs `docker login` as the image owner)
#   SKIP_BUILD=1 bash deploy.sh 1    # C1: min-scale 1 (always warm) instead of C0 (min-scale 0)
#
# IMAGE=<repo:tag> overrides the image (e.g. your own Docker Hub account).
# NO_REDIS=1 skips the shared database (each tool pod keeps its own in-memory copy).
# WINDOW=<time> idle time before a tool scales down (default 60s = Knative default; e.g. WINDOW=6s)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.arkade/bin:$PATH"

MIN_SCALE=${1:-0}
IMAGE=${IMAGE:-docker.io/sankalps2003/pulse-mcp-tools:0.2}
AGENT_DIR=../../agent

# First Python that actually runs (on Windows `python3` can be a Microsoft Store placeholder)
PY=""
for c in python python3 py; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then PY=$c; break; fi
done
[ -n "$PY" ] || { echo "No working Python found"; exit 1; }

kubectl config use-context kind-pulse-kn

# 1. Image (public on Docker Hub; .dockerignore keeps .env, .venv and runs/ out)
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  docker build -f "$AGENT_DIR/Dockerfile.mcp" -t "$IMAGE" "$AGENT_DIR"
  docker push "$IMAGE"
fi

# 2. Shared database: Redis, then load the airline data into it (one-off Job, same image)
REDIS_URL=redis://redis.default.svc.cluster.local:6379/0
if [ "${NO_REDIS:-0}" != "1" ]; then
  kubectl apply -f redis.yaml
  kubectl rollout status deploy/redis --timeout=180s
  kubectl delete pod redis-seed --ignore-not-found
  kubectl run redis-seed --image="$IMAGE" --restart=Never --env="REDIS_URL=$REDIS_URL" \
    --command -- python mcp_server.py --seed
  kubectl wait pod/redis-seed --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s
  kubectl logs pod/redis-seed
  kubectl delete pod redis-seed
else
  REDIS_URL=""
fi

# 3. Services
"$PY" gen_services.py --image "$IMAGE" --min-scale "$MIN_SCALE" --window "${WINDOW:-60s}" --redis-url "$REDIS_URL" -o tools.yaml
kubectl apply -f tools.yaml
kubectl wait ksvc -l pulse/role=mcp-tool --for=condition=Ready --timeout=600s
kubectl get ksvc -l pulse/role=mcp-tool

echo
echo "Agent settings for these tools:"
echo "  TOOL_BACKEND=mcp"
echo "  MCP_URL_TEMPLATE=http://127.0.0.1:8081/mcp"
echo "  MCP_HOST_TEMPLATE={svc}.default.example.com"
