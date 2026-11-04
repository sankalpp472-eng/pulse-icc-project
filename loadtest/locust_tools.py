"""
Locust: the Knative tool services under load. Each user calls a random recorded READ-ONLY call (real arguments)
once a second through the ingress, so Knative scales the tools (containerConcurrency 1: one pod per in-flight
call). Calls slower than 1 s count as cold. Watch the pods in the UI or Grafana while it runs.

    locust -f loadtest/locust_tools.py --headless                          # 2, 5, 10, 20 users, 60 s each
    locust -f loadtest/locust_tools.py --headless --stages "20:60:20"      # a burst: 20 users at once

The tools' address comes from MCP_URL_TEMPLATE / MCP_HOST_TEMPLATE (kind: http://127.0.0.1:8081/mcp + Host header;
the UI sets them for the current cluster).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locust import HttpUser, constant_throughput, task  # noqa: E402

import common  # noqa: E402
import locust_common as lc  # noqa: E402
from locust_common import StagesShape  # noqa: E402,F401

lc.TARGET.update(name="tools", default_stages="2:60,5:60,10:60,20:60")


class ToolUser(HttpUser):
    wait_time = constant_throughput(1)          # one call per second per user
    host = "http://127.0.0.1"                   # unused: every call has its full URL (one per tool)

    def on_start(self):
        self.calls = common.read_only_calls(lc.episodes(self.environment))
        self.rng = random.Random(id(self))

    @task
    def call(self):
        tool, args = self.rng.choice(self.calls)
        url, headers, body = common.mcp_request(tool, args)
        with self.client.post(url, json=body, headers=headers, name=f"tool:{tool}", catch_response=True,
                              timeout=120) as r:
            why = common.mcp_failed(r.status_code, r.text)
            if why:
                r.failure(why)
