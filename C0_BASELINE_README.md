# Running the C0 baseline (Knative, local cluster)

End-to-end guide: install the tools, create a local Kubernetes cluster with Knative, deploy the
14 airline MCP tools as serverless functions, and measure cold vs warm calls.

**C0 = scale-to-zero, no prewarming.** An idle tool is removed by Knative after about 40 s; the
next call is a cold start. **C1** = one warm replica per tool (same steps, one flag changed).

> **Status (16 Oct 2026):** steps 1–7 work and are tested. The *workload* C0 baseline (replaying
> W1 airline traces against these tools) is the next piece to build. Step 7 shows the live agent
> calling the tools, which is a demo, not the baseline.

---

## 0. What you need

| | Minimum |
|---|---|
| Laptop | 8 GB RAM, 4 CPU. Close heavy apps (many Chrome tabs) before starting |
| Docker Desktop | Running, **not paused**, with ≥ 4 GB memory (Settings → Resources) |
| Shell | **Windows: Git Bash** (not PowerShell). macOS/Linux: any terminal |
| Python | 3.11 |

Commands below are for **Windows + Git Bash**. On macOS/Linux, replace
`agent/.venv/Scripts/python` with `agent/.venv/bin/python`.

## 1. Install the command-line tools (one time)

`kubectl` comes with Docker Desktop. Install `kind` (Kubernetes in Docker) with arkade:

```bash
curl -sLS https://get.arkade.dev | sh
```
```bash
arkade get kind --os mingw
```
```bash
export PATH="$HOME/.arkade/bin:$PATH"
```

macOS: `brew install kind`. Linux: `arkade get kind`. Check that everything is found:

```bash
docker info --format '{{.ServerVersion}}' && kubectl version --client && kind version && python --version
```

## 2. Set up the Python environment (one time)

From the project root:

```bash
python -m venv agent/.venv
```
```bash
agent/.venv/Scripts/python -m pip install -r agent/requirements.txt
```

If pip fails with an SSL or "invalid wheel" error, the network dropped a download. Rerun the
same command (add `--no-cache-dir` if it repeats).

For step 7 only (the live agent) you also need a Groq API key:

```bash
cp agent/.env.example agent/.env
```

Then edit `agent/.env` and set `GROQ_API_KEY`. **Never commit `.env`** (it is in `.gitignore`).

## 3. Create the cluster and install Knative (about 5 minutes)

```bash
bash test/test_knative/setup.sh
```

This creates a 2-node kind cluster `pulse-kn`, installs **Knative Serving v1.23 + Kourier**
(ingress on `http://127.0.0.1:8081`), and deploys a small test function `cold-tool`.
The script is safe to rerun if a download fails halfway.

Quick check (Knative routes by the `Host` header):

```bash
curl -s -X POST -H "Host: cold-tool.default.example.com" http://127.0.0.1:8081/ -d '{}'
```

## 4. Single-function C0 check (about 8 minutes)

```bash
cd test/test_knative && python c0_baseline.py --trials 10 && cd ../..
```

Each trial waits until Knative scales the function to zero by itself, then makes one cold call and
five warm calls. Results: `test/test_knative/results/c0_<time>.csv` and `_summary.json`.

**Our result:** 10/10 cold calls succeeded; cold p50 **1.69 s**, warm p50 **0.11 s**.

## 5. Deploy the 14 MCP tools to Knative (about 2 minutes)

Each τ-bench airline tool is its own MCP server and its own Knative service
(`get-user-details`, `cancel-reservation`, …), all from the public image
`sankalps2003/pulse-mcp-tools:0.1`.

```bash
SKIP_BUILD=1 bash deploy/knative/deploy.sh
```

- `SKIP_BUILD=1` uses the existing public image. Only the image owner runs it **without**
  `SKIP_BUILD` (that rebuilds and pushes; needs `docker login`). To use your own Docker Hub
  account: `IMAGE=docker.io/<you>/pulse-mcp-tools:0.1 bash deploy/knative/deploy.sh`.
- For **C1** (always warm) pass `1` at the end: `SKIP_BUILD=1 bash deploy/knative/deploy.sh 1`.
  Go back to C0 with the same command without `1`.

Check the services are Ready, and watch them scale to zero after about 40 s:

```bash
kubectl get ksvc -l pulse/role=mcp-tool
```
```bash
kubectl get pods -l pulse/role=mcp-tool -w
```

## 6. Per-tool cold vs warm check (about 3 minutes)

```bash
agent/.venv/Scripts/python deploy/knative/check_tools.py
```

For each tool: wait until it is at zero, one cold MCP call, one warm MCP call. Results:
`deploy/knative/results/tools_cold_warm_<time>.csv`.

**Our results (two runs, 16 Oct 2026):** 14/14 tools OK; cold p50 **3.9–5.8 s**
(range 2.7–12.2 s), warm p50 **51–64 ms**. Real MCP tools start much slower than the small
test function in step 4 (the MCP SDK and the 5 MB airline database load at start-up).

## 7. Optional: the live agent calling the tools on Knative

```bash
cd agent
export TOOL_BACKEND=mcp MCP_URL_TEMPLATE="http://127.0.0.1:8081/mcp" MCP_HOST_TEMPLATE="{svc}.default.example.com"
.venv/Scripts/python run_tasks.py --tasks 1 --tools mcp
cd ..
```

Expect several minutes per task: Groq's free tier makes the agent wait 10–50 s between calls.
Those waits exceed Knative's ~40 s idle window, so most tool calls come out cold. That is why the
workload baseline will **replay** traces with realistic gaps instead of running the live agent.
In MCP mode the run shows `reward=None` (the database lives in the tool servers).

## 8. Clean up

```bash
bash test/test_knative/teardown.sh
```

Deletes the cluster and frees Docker's memory. Rerun steps 3 and 5 next time.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Docker Desktop is manually paused` | Docker paused | Whale icon → Resume |
| `connection refused` / `TLS handshake timeout` on 127.0.0.1 after a reboot or sleep | kind lost its port mappings | `bash test/test_knative/teardown.sh && bash test/test_knative/setup.sh`, then step 5 |
| `The paging file is too small` | Laptop out of memory | Close apps, rerun the step |
| `error parsing https://github.com/...` or `connection forcibly closed` | Network dropped a download | Rerun the same script |
| `curl` returns 404 | Missing or wrong `Host:` header | `kubectl get ksvc` shows the right host |
| `Python was not found` (Windows) | `python3` is a Microsoft Store placeholder | Use `python`; `deploy.sh` already does |
| A tool never scales to zero | Something keeps calling it, or a broken old revision | `kubectl get revisions`; delete the broken one |

## Where things are

| Path | What it is |
|---|---|
| `test/test_knative/` | Cluster + Knative setup, single-function C0 test (`c0_baseline.py`) |
| `deploy/knative/` | Tool deployment: `deploy.sh`, `gen_services.py` (service manifests), `check_tools.py` |
| `agent/mcp_server.py` | Wraps each τ-bench tool as its own MCP server (`TOOLS=<tool>` per Knative service) |
| `agent/Dockerfile.mcp` | The tool image |
| `agent/tools.py` | Agent-side tool backends: local or MCP |
| `test/test_cold_start/` | The earlier OpenFaaS CE test (why we moved to Knative) |
