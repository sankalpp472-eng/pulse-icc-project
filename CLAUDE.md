# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pulse** — DS252: Introduction to Cloud Computing (IISc, Aug 2026)

Predictive prewarming system for Kubernetes/OpenFaaS pods in agentic (ReAct-style) workflows. The core idea: predict the next MCP tool a LLM agent will call and proactively scale up its OpenFaaS pod before the call arrives, eliminating cold-start latency (1–3 s) that accumulates across multi-step agent loops.

## System Architecture

The system has four main components:

1. **Agent** — A custom ReAct agent (reason → act → observe) that emits a prediction hint (workflow ID, current node, recent trajectory) *before* each tool call.

2. **Prediction Service** (Python + scikit-learn) — Receives hints, uses workflow graph topology + historical telemetry + real-time trajectory to return ranked next-tool probabilities. Must respond at p95 < 50 ms at 10 hints/s.

3. **Prewarming Controller** (Go + controller-runtime) — Watches a Kubernetes CRD. When the predictor outputs a tool with confidence ≥ θ, the controller scales up the corresponding OpenFaaS replica. A TTL triggers automatic eviction if the tool is unused.

4. **MCP Tool Functions** (OpenFaaS) — 14 tools from tau-bench's airline environment (reads + writes). Controlled startup behavior (~50–150 ms execution time) is the measurement target.

### Tier Boundaries

| From → To | Data |
|---|---|
| Agent → Predictor | Workflow ID, current tool, trajectory, graph ID, timestamp |
| Predictor → Controller | Tool ID, probability, threshold θ, TTL, desired replicas (via CRD) |
| OpenFaaS → Agent | Tool result + latency |
| Any → Trace store | Tool/timing/resource/prediction fields (no raw prompts or credentials) |

## Technology Stack

| Layer | Choice |
|---|---|
| Orchestration | Kubernetes |
| FaaS | OpenFaaS (note: scale-to-zero requires Pro/paid tier) |
| MCP | MCP SDK microservers |
| Prediction | Python + scikit-learn |
| Controller | Go + controller-runtime / client-go |
| Load generation | Locust / asyncio |
| Metrics | Prometheus |
| Trace store | SQLite (dev) / PostgreSQL (prod) |

## Evaluation Configurations

- **C0** — scale-to-zero, no prewarming (cold-start baseline)
- **C1** — always-warm, one ready replica per tool (latency floor / resource ceiling)
- **C3** — graph + history + trajectory predictor with threshold θ and TTL eviction (proposed method)

Key NFR targets: predictive p50 ≥20% lower than C0; p99 ≥15% lower; ≥30% fewer warm-pod-seconds than C1.

## Key Design Decisions

- **CRD + Controller over script**: continuous reconciliation loop handles pod failures automatically.
- **Lightweight feature model first**: scikit-learn before LSTMs/Transformers — isolate prewarming impact from model complexity.
- **Go for controller**: native Kubernetes libraries (client-go, controller-runtime), lower overhead than Python.
- **MCP over direct API calls**: standardized, discoverable tools with controllable startup behavior needed for reproducible measurement.

## Hardware Setup

- **Client tier**: laptop (2–4 vCPU, 8 GB) for load generation and agent execution.
- **Cloud tier**: Kubernetes cluster, 2 worker nodes (2–4 vCPU, 16 GB, 50 GB SSD each).

## Workload Protocol

All experiments use: 3 branching tau-bench airline workflows, fixed seeds, 2-min warm-up + 10-min steady state, 3 trials (report median with min/max).
