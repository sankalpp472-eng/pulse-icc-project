# Cold-start test on OpenFaaS CE (local kind cluster)

**Question:** can OpenFaaS Community Edition on Kubernetes scale a function to zero
and wake it on the next request (the faasd method), so we can measure repeated cold starts?

**Answer we got (2026-10-14):** No. CE puts replicas back to 1 within a second, so the
function is never at zero. Details and numbers are in [Results](#results). Follow the
steps below to reproduce it yourself.

---

## 0. Prerequisites (one-time)

Run everything in **Git Bash** (not PowerShell), from this folder.

| Tool | Check | Install |
|---|---|---|
| Docker Desktop (running) | `docker info` | already installed |
| kubectl | `kubectl version --client` | already installed |
| kind, helm, faas-cli | `kind version` | `arkade get kind helm faas-cli --os mingw` |
| Python 3.11 | `python --version` | already installed |

arkade puts the tools in `~/.arkade/bin`. Add them to your PATH for this terminal:

```bash
export PATH="$HOME/.arkade/bin:$PATH"
```

You also need to be logged in to Docker Hub as `sankalps2003` (`docker login`), because
**CE only accepts public images**.

## 1. Create the cluster + OpenFaaS + test function

```bash
cd test/test_cold_start
bash setup.sh
```

This takes about 3–5 minutes. It:
1. creates a 2-node kind cluster called `pulse` (gateway on http://127.0.0.1:8080)
2. installs OpenFaaS CE with Helm
3. builds `fn/` as `sankalps2003/pulse-cold-tool:0.1`, pushes it and loads it into kind
4. deploys the function `cold-tool` (sleeps 100 ms, reports if it is a pod's first request)

Check that it works:

```bash
curl -s -X POST http://127.0.0.1:8080/function/cold-tool -d '{}'
```

You should see something like
`{"pod": "cold-tool-...", "first_request": false, "requests_served": 3, ...}`.

> If the deploy prints `200 OK` but the pod still runs an old image, check the provider log:
> `kubectl logs -n openfaas deploy/gateway -c faas-netes --tail=20`.
> A rejected deploy (e.g. "only allows public images") still prints 200.

## 2. Test A: does CE allow scale-to-zero?

Scale to 0, then watch the replica count every second:

```bash
kubectl scale deploy cold-tool -n openfaas-fn --replicas=0
for i in 1 2 3 4 5; do kubectl get deploy cold-tool -n openfaas-fn -o jsonpath='{.spec.replicas}{"\n"}'; sleep 1; done
```

**Expected on CE:** `1` every time. `faas-netes` has already put it back.

You can also try the official OpenFaaS API:

```bash
P=$(kubectl get secret -n openfaas basic-auth -o jsonpath='{.data.basic-auth-password}' | base64 -d)
curl -s -u admin:$P -X POST http://127.0.0.1:8080/system/scale-function/cold-tool -d '{"serviceName":"cold-tool","replicas":0}'
kubectl logs -n openfaas deploy/gateway -c faas-netes --tail=3
```

**Expected:** the log shows `Set replicas - cold-tool openfaas-fn, 1 => 1` (clamped to 1).

To see who changed the replicas, run:
`kubectl get deploy cold-tool -n openfaas-fn -o yaml --show-managed-fields | grep manager`

## 3. Test B: the full faasd-style script (`cold_start.py`)

```bash
python cold_start.py --trials 10
```

On OpenFaaS CE this **stops at trial 1 after ~2 minutes** with
`RuntimeError: cold-tool: pods still present after scaling to 0`. That error means the same
thing as Test A. The script is kept because it should work, with small changes, on a platform
that really scales to zero (e.g. Knative). Results go to `results/`.

## 4. Test C: kill the pod and call immediately (`kill_pod_probe.py`)

This is the closest CE gets: delete the pod, then call right away while Kubernetes restarts it.

```bash
python kill_pod_probe.py 5
```

Then repeat with the image pull policy changed from CE's forced `Always` to `IfNotPresent`:

```bash
kubectl patch deploy cold-tool -n openfaas-fn --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"IfNotPresent"}]'
kubectl rollout status deploy/cold-tool -n openfaas-fn
python kill_pod_probe.py 5
```

Read the output like this:
- `HTTP Error 500/503`: the gateway did **not** wait for the new pod. The call just failed.
- `first_request=True` with a few seconds of latency: a real cold start (request served by a brand-new pod).
- The "warm median" line includes failed calls, so ignore it here.

## Troubleshooting

**`dial tcp 127.0.0.1:<port>: connection refused` (from setup.sh or kubectl)**
Docker Desktop was restarted (laptop reboot or sleep). The kind containers come back
without their port mappings, so neither the API server nor the gateway can be reached.
Check with `docker ps --filter name=pulse`: the control-plane line shows `6443/tcp`
without a `127.0.0.1:...->` mapping. Fix: recreate the cluster.

```bash
bash teardown.sh && bash setup.sh
```

## 5. Clean up

```bash
bash teardown.sh
```

This deletes the kind cluster and frees Docker's memory. The image stays on Docker Hub.

---

## Results

Laptop setup: Docker Desktop with 4 CPU / 4 GB, kind 2 nodes, OpenFaaS CE (Helm chart, Oct 2026).

| Finding | Evidence |
|---|---|
| CE reverts `replicas=0` to 1 within ~1 s | Test A; `managedFields` shows `faas-netes` set replicas |
| The scale API clamps 0 to 1 | `Set replicas - cold-tool openfaas-fn, 1 => 1` |
| CE only allows public images | deploy error: *"the Community Edition license agreement only allows public images"* |
| CE forces `imagePullPolicy: Always` (Helm setting ignored) | `kubectl get deploy cold-tool -n openfaas-fn -o yaml \| grep imagePullPolicy` |
| On pod kill, most calls fail instead of waiting | Test C: 3/5 failed (Always), 4/5 failed (IfNotPresent) |
| Rough cold start when it did succeed | **~5.7 s** with Always pull, **~3.7 s** with IfNotPresent (only 3 samples, so ballpark only) |

**Conclusion:** OpenFaaS CE on Kubernetes cannot give repeated scale-to-zero cold starts, so the
C0 baseline, TTL eviction and the miss fallback (FR-8) cannot be built on it. Options: Knative
Serving (free, real scale-to-zero with request buffering) or OpenFaaS Pro.

## Files

| File | What it is |
|---|---|
| `setup.sh` / `teardown.sh` | create / delete everything |
| `kind-config.yaml` | 2-node local cluster, gateway mapped to port 8080 |
| `stack.yml` | OpenFaaS function definition |
| `fn/` | the test function (Dockerfile + `handler.py`) |
| `cold_start.py` | faasd-style test: scale to 0, then cold call, then warm calls |
| `kill_pod_probe.py` | kill-pod test (Test C) |
| `results/` | output of `cold_start.py` |
