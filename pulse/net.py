"""
Which network address Pulse's own HTTP listeners bind to: the predictor the UI starts (:8000), the prewarm
controller's metrics (:9102) and a run's live metrics (:9105).

  on a laptop / VM   127.0.0.1: only this machine can reach them (none of them asks for credentials, and the
                     predictor's /predict writes PrewarmRequests on a real cluster)
  in a pod           0.0.0.0: Prometheus scrapes the pod's IP; only the cluster can reach it

PULSE_BIND overrides it. Set it only to let Prometheus in a local kind cluster scrape the laptop's processes
(infra/monitoring/values-kind.yaml): Docker Desktop reaches 127.0.0.1 through host.docker.internal already;
native Docker on Linux / WSL needs the Docker bridge address, e.g. PULSE_BIND=172.17.0.1 (`ip -4 addr show docker0`),
or 0.0.0.0, which also opens the ports to the local network.
"""

import os


def bind_host() -> str:
    if os.getenv("PULSE_BIND"):
        return os.environ["PULSE_BIND"]
    return "0.0.0.0" if os.getenv("KUBERNETES_SERVICE_HOST") else "127.0.0.1"
