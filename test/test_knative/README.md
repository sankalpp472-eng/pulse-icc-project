# C0 baseline on Knative (local kind cluster)

> **Start here instead:** [C0_BASELINE_README.md](C0_BASELINE_README.md) is the full end-to-end guide
> (install → cluster → MCP tools + shared database → W1 workload baseline). This page covers only the
> single-function C0 test (its step 4).

**C0 = scale-to-zero, no prewarming.** A tool that sits idle is removed by Knative itself.
The next call is a real cold start: Knative's *activator* holds the request, starts a pod,
and forwards the request once the pod is ready.

This replaces the OpenFaaS CE test (`../test_cold_start/`). CE could not stay at zero
replicas, so it could not produce repeated cold starts.

---

## 0. Prerequisites (one-time)

Run everything in **Git Bash**, from this folder.

| Tool | Check |
|---|---|
| Docker Desktop (running, not paused) | `docker info` |
| kubectl | `kubectl version --client` |
| kind | `kind version` (install: `arkade get kind --os mingw`) |
| Python 3.11 | `python --version` |

```bash
export PATH="$HOME/.arkade/bin:$PATH"
```

**Memory:** the laptop has about 7.6 GB of RAM and Docker gets 4 GB of it. Close heavy apps
(especially many Chrome tabs) first. Otherwise Windows can fail with *"The paging file is too
small for this operation to complete"*.

## 1. Create the cluster + Knative + the test service

```bash
cd test/test_knative
bash setup.sh
```

This takes about 3–6 minutes. It:
1. creates a 2-node kind cluster `pulse-kn` (ingress on http://127.0.0.1:8081)
2. installs **Knative Serving v1.23** and **Kourier** (the networking layer)
3. deploys `service.yaml`: the same test image as before (`sankalps2003/pulse-cold-tool:0.2`)
   with `min-scale: 0` (may go to zero) and a 6 s idle window

Quick check. Knative routes by *Host header*, so it has to be passed:

```bash
curl -s -X POST -H "Host: cold-tool.default.example.com" http://127.0.0.1:8081/ -d '{}'
```

If the service was idle, this first call takes about 2 s and returns `"first_request": true`
(a cold start). Call it again right away: about 0.1 s and `"first_request": false` (warm).

Watch Knative scale it to zero by itself (about 40 s after the last call):

```bash
kubectl get pods -w
```

## 2. Run the C0 baseline

```bash
python c0_baseline.py --trials 10
```

Takes about 8 minutes: every trial waits until Knative has removed the pod (nothing is forced).
For each trial it prints:
- **cold**: latency of the first call after scale-to-zero, its HTTP status, and whether a fresh
  pod answered (`fresh pod=True` means a real cold start)
- **warm median**: 5 calls right after, served by the now-warm pod
- **waited**: how long the idle service took to reach zero pods
- the pod timeline in seconds after the request: created, scheduled, container started, ready

Results go to `results/<date>_<time>_C0_single-function.csv` (per trial) and `..._summary.json`
(p50/p95 cold and warm, cold-start penalty).

## Results (2026-10-15, laptop, 10 trials)

`results/2026-10-15_2000_C0_single-function*`. Setup: kind with 2 nodes, Docker 4 CPU / 4 GB, image cached on the node.

| | p50 | p95 | min | max |
|---|---|---|---|---|
| **Cold call** (after scale-to-zero) | **1687 ms** | 3248 ms | 1258 ms | 3248 ms |
| **Warm call** | **111 ms** | 119 ms | | |
| **Cold-start penalty** (p50 cold − p50 warm) | **≈1.58 s** | | | |

- **10/10** cold calls succeeded (HTTP 200, answered by a fresh pod). The activator held every request until the pod was ready.
- Knative scaled the idle service to zero after **38–49 s** (6 s window + 30 s grace period + pod shutdown).
- The pod timeline is coarse: Kubernetes timestamps have **1 s resolution**, so "created" can show up to −1 s.
  `ready_s` is often empty because Knative's queue-proxy forwards the request before Kubernetes marks the pod Ready.
- A first run (deleted, still in git history) had 4/10 HTTP 500s. Cause: the readiness probe used the watchdog's
  `/_/health`, which passes before `handler.py` listens. Fixed by probing `/_/ready` with a fast
  `/healthz` handler path (image `0.2`).

## 3. Clean up

```bash
bash teardown.sh
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused` on 127.0.0.1:8081 or the kube API after a reboot or sleep | Docker lost the port mappings. Run `bash teardown.sh && bash setup.sh` |
| `Docker Desktop is manually paused` | Unpause it from the whale icon |
| `paging file is too small` | The laptop is out of memory: close apps, then rerun |
| `curl` returns 404 | Missing or wrong `Host:` header. Check `kubectl get ksvc cold-tool` for the URL |

## Knative settings that matter for experiments

| Setting | Where | Value here | Effect |
|---|---|---|---|
| `autoscaling.knative.dev/min-scale` | `service.yaml` | `0` | 0 = C0 (scale to zero); 1 = C1 (always warm) |
| `autoscaling.knative.dev/window` | `service.yaml` | `6s` | How long without traffic counts as idle |
| `scale-to-zero-grace-period` | ConfigMap `config-autoscaler` | 30 s (default) | Extra wait before the last pod is removed |
| `scale-to-zero-pod-retention-period` | annotation or ConfigMap | 0 (default) | Keeps the last pod for a while after going idle (a "TTL") |

## Files

| File | What it is |
|---|---|
| `setup.sh` / `teardown.sh` | create / delete everything |
| `kind-config.yaml` | 2-node cluster, Kourier mapped to port 8081 |
| `service.yaml` | the Knative service (C0 settings) |
| `c0_baseline.py` | the C0 measurement |
| `results/` | output |
| The function image | built from `../test_cold_start/fn/` |
