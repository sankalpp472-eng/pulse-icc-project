#!/usr/bin/env bash
# Deploy the shared database (Redis) and the 14 airline MCP tools (or the 16 retail ones) as Knative
# services (optionally building + pushing the image first).
# Needs: the Knative cluster from infra/kind/setup.sh running.
#
#   SKIP_BUILD=1 bash infra/knative/deploy.sh      # use the existing public image (teammates: use this)
#   bash infra/knative/deploy.sh                   # build + push the image first (needs `docker login` as the image owner)
#   SKIP_BUILD=1 bash infra/knative/deploy.sh 1    # C1: min-scale 1 (always warm) instead of C0 (min-scale 0)
#   DOMAIN=retail bash infra/knative/deploy.sh     # the retail tools: services retail-<tool>, pods pulse/role=mcp-tool-retail,
#                                    #   Redis DB 1, image :0.3 (build + push it once; :0.2 has no retail tools)
#
# IMAGE=<repo:tag> overrides the image (e.g. your own Docker Hub account).
# NO_REDIS=1 skips the shared database (each tool pod keeps its own in-memory copy).
# WINDOW=<time> idle time before a tool scales down (default 60s = Knative default; e.g. WINDOW=6s)
# SCALE_DOWN_DELAY=<time> keep pods this much longer after Knative decides to scale down (default 0s = off;
#               it applies to every config, so run C0 and C3 with the same value)
# CONCURRENCY=<n> requests one tool pod serves at once (default 1: N parallel calls -> the autoscaler starts
#               N pods; 0 = no limit, one pod takes every call, the setup of the runs before fan-out)
# MAX_SCALE=<n>  most pods per tool (default 10; 0 = no limit)
# KUBE_CONTEXT=<context> cluster to deploy to (default kind-pulse-kn, the local cluster; see infra/eks/)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.arkade/bin:$PATH"

MIN_SCALE=${1:-0}
DOMAIN=${DOMAIN:-airline}
if [ "$DOMAIN" = retail ]; then
  IMAGE=${IMAGE:-docker.io/sankalps2003/pulse-mcp-tools:0.3}; DB=1; ROLE=mcp-tool-retail; OUT=tools-retail.yaml
else
  IMAGE=${IMAGE:-docker.io/sankalps2003/pulse-mcp-tools:0.2}; DB=0; ROLE=mcp-tool; OUT=tools.yaml
fi
ROOT=../..

# First Python that actually runs (on Windows `python3` can be a Microsoft Store placeholder); PYTHON overrides
PY="${PYTHON:-}"
[ -n "$PY" ] || for c in python python3 py; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then PY=$c; break; fi
done
[ -n "$PY" ] || { echo "No working Python found"; exit 1; }

kubectl config use-context "${KUBE_CONTEXT:-kind-pulse-kn}"   # KUBE_CONTEXT=<eks context> for EKS

# 1. Image (public on Docker Hub; Dockerfile.mcp.dockerignore lets only the tool code in)
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.mcp -t "$IMAGE" "$ROOT"
  docker push "$IMAGE"
fi

# 2. Shared database: Redis, then load the domain's data into it (one-off pod, same image)
REDIS_URL=redis://redis.default.svc.cluster.local:6379/$DB
if [ "${NO_REDIS:-0}" != "1" ]; then
  kubectl apply -f redis.yaml
  kubectl rollout status deploy/redis --timeout=180s
  # After a Docker restart cluster DNS comes up last; the seed pod must resolve "redis"
  kubectl -n kube-system rollout status deploy/coredns --timeout=180s
  kubectl delete pod redis-seed --ignore-not-found
  kubectl run redis-seed --image="$IMAGE" --restart=Never --env="REDIS_URL=$REDIS_URL" --env="DOMAIN=$DOMAIN" \
    --command -- python mcp_server.py --seed
  kubectl wait pod/redis-seed --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s
  kubectl logs pod/redis-seed
  kubectl delete pod redis-seed
else
  REDIS_URL=""
fi

# 3. Services
(cd "$ROOT" && PYTHONPATH=. "$PY" -m pulse.agent.gen_services --domain "$DOMAIN" --image "$IMAGE" --min-scale "$MIN_SCALE" --window "${WINDOW:-60s}" --scale-down-delay "${SCALE_DOWN_DELAY:-0s}" \
  --concurrency "${CONCURRENCY:-1}" --max-scale "${MAX_SCALE:-10}" --redis-url "$REDIS_URL" -o "infra/knative/$OUT")
kubectl apply -f "$OUT"
kubectl wait ksvc -l pulse/role=$ROLE --for=condition=Ready --timeout=600s
kubectl get ksvc -l pulse/role=$ROLE

echo
echo "Agent settings for these tools:"
echo "  TOOL_BACKEND=mcp"
echo "  MCP_URL_TEMPLATE=http://127.0.0.1:8081/mcp"
if [ "$DOMAIN" = retail ]; then
  echo "  MCP_HOST_TEMPLATE=retail-{svc}.default.example.com"
else
  echo "  MCP_HOST_TEMPLATE={svc}.default.example.com"
fi
