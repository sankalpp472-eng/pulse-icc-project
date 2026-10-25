# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pulse** — DS252: Introduction to Cloud Computing (IISc, Aug 2026)

Predictive prewarming for serverless MCP tools on Kubernetes (Knative Serving) in agentic (ReAct-style) workflows. The core idea: predict the next MCP tool an LLM agent will call and wake its scaled-to-zero pod before the call arrives, hiding the cold-start latency (1–5 s) that accumulates across multi-step agent loops.

The PRD (`docs/PRD.docx`) was written for OpenFaaS. The project moved to **Knative** because OpenFaaS Community Edition never keeps a function at zero replicas, so tools never go cold during a run (see `docs/Knative_decision.docx`).

## System Architecture

1. **Agent** (`agent/`) — A custom ReAct agent (reason → act → observe, one tool call at a time) on tau-bench's airline domain, with an LLM-simulated customer. LLMs via Groq / OpenRouter / Ollama. Sends a prediction hint (workflow ID, episode, trajectory so far) to the predictor before every LLM step.

2. **Prediction Service** (`predictor/`, Python + scikit-learn + LightGBM + FastAPI) — Two families. Tool-only models (graph / history / trajectory / ensemble / lr) rank the next tool from the trajectory. Next-action models (`gbt` = LightGBM gradient-boosted trees, main; `lr_action`; `ngram_action` baseline; `action_model.py`, `train_action.py`) predict the next LLM step among the 14 tools + RESPOND from the trajectory plus agent-side signals (numbers/flags only, `signals.py`), and a fan-out width k (pending-ID rule) that becomes `desiredReplicas`. Tools with P ≥ θ become `PrewarmRequest` objects (CRD); RESPOND/END are never prewarmed. Target: p95 < 50 ms at 10 hints/s.

3. **Prewarm Controller** (`controller/prewarm_controller.py`, Python) — Watches `PrewarmRequest` objects and sends the tool one MCP `ping`, which makes Knative start a pod if the tool is at zero. TTL = the tools' Knative idle window (`autoscaling.knative.dev/window`, 60 s); Knative scales unused tools back to zero, and the controller marks them `Evicted`. Contract and status fields: `deploy/prewarm/README.md`.

4. **MCP Tool Functions** (Knative) — 14 tau-bench airline tools, each its own MCP server (`agent/mcp_server.py`) and Knative Service, sharing one Redis copy of the airline database.

### Tier Boundaries

| From → To | Data |
|---|---|
| Agent → Predictor | Workflow ID, current tool, trajectory, graph ID, timestamp |
| Predictor → Controller | Tool ID, probability, threshold θ, TTL, desired replicas (via the `PrewarmRequest` CRD) |
| Knative tool → Agent | Tool result + latency |
| Any → Trace store | Tool/timing/resource/prediction fields (no raw prompts or credentials) |

## Technology Stack

| Layer | Choice |
|---|---|
| Orchestration | Kubernetes: kind (local, 2 nodes) or AWS EKS (`EKS_deployment/`) |
| Serverless | Knative Serving v1.23 + Kourier (min-scale 0 = scale-to-zero) |
| MCP | MCP Python SDK, Streamable HTTP, one server per tool |
| Prediction | Python + scikit-learn, FastAPI |
| Controller | Python (`kubernetes` client); a Go/controller-runtime port is possible later |
| Load generation | Trace replay (`test/test_knative/w1_replay.py`) and the live agent (`w1_live.py`) |
| Metrics | Prometheus (`/metrics` on the predictor and the runners) |
| Trace store | CSV / JSONL files per run (`test/test_knative/results/`) |

## Repository Layout

| Path | What |
|---|---|
| `agent/` | ReAct agent, LLM client, simulated user, MCP tool servers, hints, `tau_airline/` (tools, data, tasks) |
| `predictor/` | Steps 1–9: trace export, graphs, models, training, evaluation, API, load test, CRD writer |
| `controller/` | Prewarm controller and its action logs |
| `deploy/knative/` | Knative Services for the tools, Redis, `deploy.sh`, per-tool cold/warm check |
| `deploy/prewarm/` | `PrewarmRequest` CRD and the predictor ↔ controller contract |
| `test/test_knative/` | Local cluster setup, C0/C1/C3 runners, plots, `results/` |
| `test/test_cold_start/` | The OpenFaaS CE tests that led to the move to Knative |
| `EKS_deployment/` | EKS cluster, Knative setup, `run_c0/c1/c3/all.sh` |
| `workloads/` | Workload spreadsheet (prompts, graphs, real traces) |
| `docs/` | PRD and the Knative decision document |

Result folders are named `<date>_<time>_<config>_W1-<replay|live>_<details>[_label]`, e.g. `2026-10-22_1336_C3_W1-replay_task9-x3`.

## Evaluation Configurations

- **C0** — scale-to-zero (min-scale 0), no prewarming (cold-start baseline)
- **C1** — always-warm, min-scale 1 per tool (latency floor / resource ceiling)
- **C3** — scale-to-zero + predictor (θ) + prewarm controller, TTL = Knative idle window (proposed method)

Key NFR targets: predictive p50 ≥20% lower than C0; p99 ≥15% lower; ≥30% fewer warm-pod-seconds than C1.

## Key Design Decisions

- **Knative over OpenFaaS CE**: real scale-to-zero, the activator holds requests during a cold start (miss fallback, FR-8), private images allowed.
- **Prewarm by MCP ping ("Option A")**: changing a Knative Service's min-scale creates a new Revision, so the controller wakes tools with a request instead; Knative's idle window is the TTL.
- **CRD + controller over a script**: the predictor only writes desired state; the controller reconciles it.
- **Lightweight feature model first**: scikit-learn before LSTMs/Transformers — isolate prewarming impact from model complexity.
- **MCP over direct API calls**: standardized, discoverable tools with controllable startup behavior for reproducible measurement.
- **Replay for baselines**: seeded replay of recorded traces gives identical request sequences across C0/C1/C3 without LLM quota limits.

## Hardware Setup

- **Client tier**: laptop (2–4 vCPU, 8 GB) for load generation and agent execution.
- **Cloud tier**: PRD: 2 worker nodes (2–4 vCPU, 16 GB, 50 GB SSD each). EKS currently uses 2 × m7i-flex.large (2 vCPU, 8 GiB, Free Tier eligible).

## Workload Protocol

All experiments use: 3 branching tau-bench airline workflows, fixed seeds, 2-min warm-up + 10-min steady state, 3 trials (report median with min/max). So far only W1 (change/cancel a booking) has been run.
