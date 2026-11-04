# Load testing (Locust + asyncio)

These tests put Pulse under rising load. They use the same recorded W1 episodes as the replay runner
(`test/test_knative/w1_replay.py`): the same seed gives the same traces, arguments, gaps, fan-out
groups and hints.

| Test | Engine | What it does | Needs |
|---|---|---|---|
| `locust_predictor.py` | Locust | Each user replays sessions' hints in order (signals + history), one per second, so users = hints/s. Default 1 → 5 → 10 → 20, 60 s each | a predictor (any: gbt, speculator, pulse_v3) |
| `locust_tools.py` | Locust | Each user makes one recorded **read-only** MCP call per second, so the shared Redis database is never changed. Calls slower than 1 s count as cold | Knative tools |
| `locust_episodes.py` | Locust | Each user is one agent replaying whole episodes with their real gaps and fan-out groups; with `--predictor-url` (C3) it also sends hints without waiting. Users = concurrent agents | Knative tools (+ predictor for C3) |
| `async_load.py predictor` | asyncio | Open loop: exactly R hints/s for each rate, whether or not answers come back (the NFR-4 check) | a predictor |
| `async_load.py tools` | asyncio | Open loop: bursts of N simultaneous calls per tool; `--pause 90` makes every burst start cold | Knative tools |

Locust is **closed loop**: each user waits for its answer before the next request. asyncio is **open
loop**: requests go out on a fixed schedule. Use Locust to find where things bend as users grow, and
asyncio to check a target rate exactly.

## Run

```bash
"$PY" -m pip install -r loadtest/requirements.txt
"$PY" -m locust -f loadtest/locust_predictor.py --headless --host http://127.0.0.1:8000
"$PY" -m locust -f loadtest/locust_tools.py --headless --stages "20:60:20"          # 20 users at once
"$PY" -m locust -f loadtest/locust_episodes.py --headless --pulse-config C3 --predictor-url http://127.0.0.1:8000
"$PY" loadtest/async_load.py predictor --url http://127.0.0.1:8000 --rates 1 5 10 20 --duration 60
"$PY" loadtest/async_load.py tools --burst 1 5 10 --pause 90
"$PY" loadtest/plot_load.py results/load_runs/<run> results/load_runs/<run> --out results/load_runs/compare_<...>
```

- **Locust options:** `--stages "users:seconds[:spawn rate],..."`, `--pulse-config C0|C1|C3` (recorded
  with the result), `--seed`, `--tasks "6 20 22"`, `--label`, and for episodes `--gap-scale`. Drop
  `--headless` for Locust's web UI on :8089.
- **Tool address:** taken from `MCP_URL_TEMPLATE` / `MCP_HOST_TEMPLATE`, the same as the runners and
  the controller. kind: `http://127.0.0.1:8081/mcp` with a Host header. The UI sets them for the
  current cluster.
- **Tools and episodes:** set the configuration first: tools at min-scale 0 for C0 and C3, or 1 for
  C1 (`deploy/knative/deploy.sh`). For C3 run the controller and a predictor with `CRD_MODE=k8s`.
  The UI does all of this for you.

## From the UI

**Experiments tab → Mode: Load test.**
- **Choose:** the target, engine, stages (Locust), or rates and bursts (asyncio), the configuration
  and the predictor.
- **What the UI does:** sets min-scale, starts the predictor and controller as needed, runs the test,
  streams its log, and shows the summary.
- **Where:** the predictor test also works in simulated mode, because it needs no cluster; tools and
  episodes need Knative. In the EKS UI pod the tests run next to the tools on the client node.
- **Results tab:** load runs show requests, failed requests, latency and requests/s at peak load.
  Compare and Make plots (`plot_load.py`) work on load runs, and the Grafana link shows the run's
  window.
- **Live demo:** **Background load (agents)** (Knative mode) runs that many extra agents
  (`locust_episodes.py`) while your episode plays, so you can watch prewarming under pressure.

## Output

`results/load_runs/<date>_load-<target>_<engine>_<config>[_label]/` (`PULSE_LOAD_DIR` overrides it):

| File | Contents |
|---|---|
| `requests.csv` | one row per request: start, load (users / rate / burst), kind (hint / tool), latency, ok |
| `summary.json` | totals, latency p50/p95/p99, `cold_share` (tools), and `stages`: the same per load level, for the plots |
| `run_config.json` | settings, tool address templates, runner |

Prometheus (`deploy/monitoring/`) sees the predictor, the controller and the pods during a run. The
Monitoring tab follows a running load test automatically.
