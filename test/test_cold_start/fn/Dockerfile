FROM ghcr.io/openfaas/of-watchdog:0.10.7 AS watchdog

FROM python:3.11-slim
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN useradd -m app
WORKDIR /home/app
COPY handler.py .
USER app

# of-watchdog in http mode: it starts handler.py and proxies requests to it.
ENV fprocess="python -u handler.py" \
    mode="http" \
    upstream_url="http://127.0.0.1:5000" \
    ready_path="/healthz" \
    exec_timeout="30s"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1
CMD ["fwatchdog"]
