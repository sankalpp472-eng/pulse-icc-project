"""
Locust plumbing shared by locust_predictor.py, locust_tools.py and locust_episodes.py:

  extra options   --stages "users:seconds[:spawn rate],..."   the load steps (StagesShape runs them, then stops)
                  --pulse-config C0|C1|C3, --label, --seed, --tasks, --predictor-url, --gap-scale
  StagesShape     steps through --stages; the test ends after the last one (headless or web UI)
  recording       every request -> common.Recorder (load = the user count when it was sent), written to the run
                  folder at the end (requests.csv, summary.json, run_config.json)

Request names carry their kind: "hint", "tool:<name>".
"""

import time

from locust import LoadTestShape, events

import pulse.loadtest.common as common

TARGET = {"name": "load"}                # set by the locustfile
STATE: dict = {}


@events.init_command_line_parser.add_listener
def _options(parser):
    parser.add_argument("--stages", default="", help='load steps "users:seconds[:spawn rate],..." '
                                                     "(default: the locustfile's own)")
    parser.add_argument("--pulse-config", default="C0", help="C0 / C1 / C3: recorded with the results")
    parser.add_argument("--label", default="", help="end of the result folder name")
    parser.add_argument("--seed", type=int, default=252, help="replay seed: which recorded episodes, in which order")
    parser.add_argument("--tasks", default="", help="W1 task ids (default: all)")
    parser.add_argument("--predictor-url", default="", help="episodes: also send the agent's hints here (C3)")
    parser.add_argument("--gap-scale", type=float, default=1.0, help="episodes: multiply think / customer gaps")


def stages(env) -> list[tuple[int, float, float]]:
    spec = env.parsed_options.stages or TARGET.get("default_stages", "1:60,5:60,10:60,20:60")
    out = []
    for part in spec.split(","):
        users, secs, *rate = part.split(":")
        out.append((int(users), float(secs), float(rate[0]) if rate else max(1.0, int(users) / 5)))
    return out


class StagesShape(LoadTestShape):
    def tick(self):
        t = self.get_run_time()
        for users, secs, rate in stages(self.runner.environment):
            if t < secs:
                return users, rate
            t -= secs
        return None                         # after the last step: stop the test


def episodes(env) -> list[dict]:
    """The replay's episodes, loaded once per test."""
    if "eps" not in STATE:
        o = env.parsed_options
        STATE["eps"] = common.episodes(o.seed, common.parse_tasks(o.tasks))
    return STATE["eps"]


@events.test_start.add_listener
def _start(environment, **_):
    o = environment.parsed_options
    STATE["env"] = environment
    STATE["rec"] = common.Recorder(TARGET["name"], "locust", o.pulse_config,
                                   {"stages": [list(s) for s in stages(environment)], "seed": o.seed, "tasks": o.tasks,
                                    "host": o.host, "predictor_url": o.predictor_url, "gap_scale": o.gap_scale})
    STATE["dir"] = common.run_dir(TARGET["name"], "locust", o.pulse_config, o.label)


@events.request.add_listener
def _request(request_type, name, response_time, exception, start_time=None, **_):
    rec, env = STATE.get("rec"), STATE.get("env")
    if rec is None:
        return
    kind = name.split(":")[0]
    rec.add(start_time or time.time() - response_time / 1000, env.runner.user_count if env and env.runner else None, kind, name,
            response_time / 1000, exception is None, str(exception)[:120] if exception else "")


@events.test_stop.add_listener
def _stop(environment, **_):
    if STATE.get("rec") and STATE["rec"].rows:
        STATE["rec"].write(STATE["dir"])
        STATE["rec"] = None
