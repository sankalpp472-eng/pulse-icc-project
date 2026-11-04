"""
Locust: whole agent episodes under rising concurrency. Each user is one agent replaying recorded episodes back
to back, exactly as w1_replay.py does: think / customer-reply gaps from the seed, fan-out groups fired together,
and (C3, --predictor-url) the agent's hints before every LLM step, sent without waiting, as the agent does.
users = concurrent agents. Default steps 2, 5, 10 agents, 180 s each.

    locust -f loadtest/locust_episodes.py --headless --pulse-config C0
    locust -f loadtest/locust_episodes.py --headless --pulse-config C3 --predictor-url http://127.0.0.1:8000

Run C0 and C3 with the same --seed and --stages and compare them (Results tab, or loadtest/plot_load.py).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gevent  # noqa: E402
from locust import HttpUser, constant, task  # noqa: E402

import common  # noqa: E402
import locust_common as lc  # noqa: E402
from locust_common import StagesShape  # noqa: E402,F401

lc.TARGET.update(name="episodes", default_stages="2:180,5:180,10:180")


class AgentUser(HttpUser):
    wait_time = constant(0)                     # the gaps are inside the episode
    host = "http://127.0.0.1"

    def on_start(self):
        self.eps = lc.episodes(self.environment)
        self.k = abs(hash(id(self)))
        o = self.environment.parsed_options
        self.predictor, self.scale = o.predictor_url.rstrip("/"), o.gap_scale

    def hint(self, h):
        with self.client.post(self.predictor + "/predict", json=h, name="hint", catch_response=True, timeout=10) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    def tool(self, call):
        url, headers, body = common.mcp_request(call["tool"], call["args"])
        with self.client.post(url, json=body, headers=headers, name=f"tool:{call['tool']}", catch_response=True,
                              timeout=120) as r:
            why = common.mcp_failed(r.status_code, r.text)
            if why:
                r.failure(why)

    @task
    def episode(self):
        ep = self.eps[self.k % len(self.eps)]
        eid = f"load-{id(self)}-{self.k}"
        self.k += 1
        hints = iter(common.episode_hints(ep, eid)) if self.predictor else None
        tr = ep["trace"]
        for idx in ep["groups"]:
            i = idx[0]
            start = time.time()
            for off, _, _ in ep["hints"][i]:            # one hint per LLM step in the gap before this group
                gevent.sleep(max(0.0, start + off * self.scale - time.time()))
                if hints:
                    gevent.spawn(self.hint, next(hints))  # the agent never waits for the predictor
            gevent.sleep(max(0.0, start + ep["gaps"][i] * self.scale - time.time()))
            gevent.joinall([gevent.spawn(self.tool, tr["calls"][j]) for j in idx])   # a fan-out group at once
