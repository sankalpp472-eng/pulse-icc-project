"""
Locust: the predictor API under rising load (NFR-4: p95 < 50 ms at 10 hints/s).

Every user replays recorded sessions' hints in order (signals + conversation history, so stateful predictors such
as pulse_v3 see real sessions), one hint per second: users = hints/s. Default steps 1, 5, 10, 20 users, 60 s each.

    locust -f loadtest/locust_predictor.py --headless --host http://127.0.0.1:8000
    locust -f loadtest/locust_predictor.py --host http://127.0.0.1:8000 --stages "10:120"   # web UI on :8089
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locust import HttpUser, constant_throughput, task  # noqa: E402

import common  # noqa: E402
import locust_common as lc  # noqa: E402
from locust_common import StagesShape  # noqa: E402,F401  (Locust runs the shape it finds here)

lc.TARGET.update(name="predictor", default_stages="1:60,5:60,10:60,20:60")


class PredictorUser(HttpUser):
    wait_time = constant_throughput(1)          # one hint per second per user
    host = "http://127.0.0.1:8000"

    def on_start(self):
        self.hints = common.hint_stream(lc.episodes(self.environment), f"u{id(self)}")

    @task
    def hint(self):
        with self.client.post("/predict", json=next(self.hints), name="hint", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
