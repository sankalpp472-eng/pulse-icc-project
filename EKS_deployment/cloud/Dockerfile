# One image for everything Pulse runs in the cluster besides the tools:
#   pulse-predictor   uvicorn service:app --app-dir predictor                 (LightGBM gbt)
#   pulse-speculator  uvicorn service:app --app-dir baselines/llm_speculator  (LLM speculator)
#   pulse-controller  python controller/prewarm_controller.py
#   run pods          bash cloud_entry.sh replay|live <args>                 (w1_replay.py / w1_live.py)
# Built from a staging folder by build_image.sh (code + trained models + tau-bench airline trajectories;
# never agent/.env: the Groq keys reach the pods as the Kubernetes Secret pulse-llm-keys).
FROM python:3.11-slim

# kubectl: the runners use it (pod watcher, Redis reset, cluster settings), with the pod's service account
ARG KUBECTL_VERSION=v1.37.1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd -m app
WORKDIR /app
COPY app/ /app/
RUN chmod +x /app/cloud_entry.sh
USER app

ARG GIT_COMMIT=""
ARG GIT_DIRTY="0"
ARG IMAGE_TAG=""
# MCP_HOST_TEMPLATE is empty on purpose: in the cluster the URL itself names the Knative service
# TZ: run folders are named with local (India) time, like the laptop runs (POSIX form: no tzdata needed)
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=IST-5:30 \
    PULSE_GIT_COMMIT=${GIT_COMMIT} \
    PULSE_GIT_DIRTY=${GIT_DIRTY} \
    PULSE_IMAGE=${IMAGE_TAG} \
    TOOL_BACKEND=mcp \
    MCP_URL_TEMPLATE=http://{svc}.default.svc.cluster.local/mcp \
    MCP_HOST_TEMPLATE= \
    PREDICTOR_DIR=/app/predictor
