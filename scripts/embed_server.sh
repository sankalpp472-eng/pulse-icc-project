#!/usr/bin/env bash
# The embedding server pulse-v3 reads (PULSE_EMBED_BASE, default http://127.0.0.1:8091/v1): llama.cpp's server with
# nomic-embed-text-v1.5, in Docker. The same image and settings as the cluster's pulse-embed Deployment and the
# training Job's sidecar, so local, cluster and training embeddings match.
#
#   bash scripts/embed_server.sh          # start (first start downloads the 140 MB model into a Docker volume)
#   bash scripts/embed_server.sh stop
set -euo pipefail
NAME=pulse-embed
if [ "${1:-}" = stop ]; then docker rm -f "$NAME" >/dev/null && echo "stopped"; exit 0; fi
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "already running on :8091"; exit 0; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p 127.0.0.1:8091:8091 -v pulse-embed-models:/models -e LLAMA_CACHE=/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 --alias nomic-embed --embedding --pooling mean \
  -c 8192 --parallel 4 -b 2048 -ub 2048 --host 0.0.0.0 --port 8091 >/dev/null
echo -n "starting the embedding server (model download on the first start) "
for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8091/health >/dev/null; then echo " ready on http://127.0.0.1:8091/v1"; exit 0; fi
  echo -n "."; sleep 5
done
echo; echo "not ready after 10 min: docker logs $NAME"; exit 1
