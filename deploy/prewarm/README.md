# PrewarmRequest: predictor ↔ controller (Option A)

```
agent ──hint──▶ predictor /predict ──P ≥ θ──▶ PrewarmRequest ◀──watches── controller ──MCP ping──▶ Knative tool
                                            (one object per tool)        writes .status          (wakes from zero;
                                                                                                   Knative's 60 s idle
                                                                                                   window = the TTL)
```

Files: `prewarmrequest-crd.yaml` (the CRD), `predictor/crd_writer.py` (writes `.spec`),
`controller/prewarm_controller.py` (reads `.spec`, writes `.status`).

## Mechanism: Option A, "warm-up request + Knative's idle window as the TTL"

1. **Predictor:** for every tool with P ≥ θ it creates or updates the tool's `PrewarmRequest`
   (named after its Knative service, e.g. `get-reservation-details`) with a new `spec.requestedAt`.
2. **Controller:** for every **new** `requestedAt`, it sends that tool **one MCP `ping`** through the
   ingress, the same path the agent's calls take.
   - Tool at zero: Knative's activator starts a pod (`wasCold=true`). The ping returns when the pod
     is ready, so `pingLatencyMs` ≈ the cold start. It just happened **before** the agent needed the tool.
   - Tool already warm: the ping takes a few ms and counts as activity.
3. **TTL = Knative's idle window** (`autoscaling.knative.dev/window`, 60 s here), counted from the
   **last** request (ping or real call). The controller never evicts. Knative scales the tool to zero
   after the window, and the controller's sweeper then marks the object **Evicted**.
4. A repeated prediction updates `requestedAt` → a new ping → the window starts again.

`spec.ttlSeconds` and `spec.desiredReplicas` are still written by the predictor but **not used**.
`status.effectiveTtlSeconds` shows the window actually in force. To change the TTL (experiment E7),
redeploy the tools with a different window: `WINDOW=30s SKIP_BUILD=1 bash deploy/knative/deploy.sh`
(and run C0 at the same window, so C3 is compared like for like).

### Status fields (written by the controller)

| Field | Meaning |
|---|---|
| `phase` | `Warming` (ping sent) → `Ready` (a pod answered) → `Evicted` (back at zero); `Failed` if the ping failed |
| `wasCold` | the tool had no pod when the ping was sent, so this ping caused the cold start |
| `pingLatencyMs` | ping round trip: about a cold start if `wasCold`, a few ms if already warm |
| `readyAt`, `expiresAt` | when the pod answered; `readyAt` + window = earliest scale-down if nothing else calls it |
| `evictedAt` | first time the controller saw the tool back at zero pods |
| `effectiveTtlSeconds` | the service's idle window (the real TTL) |

Every action is also written to `controller/results/controller.jsonl`
(`warmed`: tool, was_cold, ping_ms, probability, episode; `evicted`: warm_for_s).

## Run it (local kind cluster; from the project root, Git Bash)

Prerequisites: the cluster and the 14 tools from `test/test_knative/C0_BASELINE_README.md`
(steps 3 and 5), and a trained predictor (`predictor/data/model.pkl`).

**1. Install the CRD** (once per cluster):
```bash
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
```

**2. Start the controller** (terminal 1):
```bash
KUBE_CONTEXT=kind-pulse-kn agent/.venv/Scripts/python controller/prewarm_controller.py
```

**3. Start the predictor writing real PrewarmRequests** (terminal 2):
```bash
CRD_MODE=k8s KUBE_CONTEXT=kind-pulse-kn agent/.venv/Scripts/python -m uvicorn service:app --app-dir predictor --port 8000
```

**4. Watch** (terminal 3):
```bash
kubectl get pwr -w
```

**5. Quick test with one hand-made hint:**
```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"workflow_id":"W1","episode_id":"demo","trajectory":["get_user_details"]}'
```
`get-reservation-details` appears in `kubectl get pwr`, goes `Warming` → `Ready` (with `COLD true` if
it was at zero), and about 1-1.5 minutes later `Evicted`.

**6. With the agent** (terminal 4): the agent sends hints whenever `PREDICTOR_URL` is set.
```bash
cd agent && PREDICTOR_URL=http://127.0.0.1:8000 WORKFLOW_ID=W1 TOOL_BACKEND=mcp MCP_URL_TEMPLATE=http://127.0.0.1:8081/mcp GROQ_MODEL=openai/gpt-oss-120b USER_MODEL=openai/gpt-oss-20b .venv/Scripts/python run_tasks.py --tasks 9
```

**On EKS:** `source EKS_deployment/use_eks.sh` in each terminal first. It sets the kubectl context
and `MCP_URL_TEMPLATE`, which the controller uses for its pings. Drop the `KUBE_CONTEXT=kind-pulse-kn`
prefixes.

## Measured on the local cluster (22 Oct 2026, one hand-made hint)

| | |
|---|---|
| Hint → `PrewarmRequest` → ping sent | well under a second |
| Warm-up of a tool at zero (`pingLatencyMs`) | 8.6 s (laptop under load; tool cold starts were 3-6 s in C0) |
| Real agent call after the prewarm | 0.42 s (warm; no cold start) |
| Pod evicted by Knative | ~95 s after `readyAt` (last real call + 60 s window + scale-down) |

## Notes

- **Lead time is the open problem.** The agent's LLM thinks for ~1-2 s after a hint, while a cold start
  takes ~4 s or more. A ping sent at the hint is often still running when the agent calls the tool.
  The agent's call then simply waits for the same pod (never fails), but only part of the cold start
  is hidden. Prewarms sent earlier (during customer turns, or predicting further ahead) hide all of it.
- The controller runs on the laptop (like the predictor). It can later run in the cluster as a
  Deployment, or be ported to Go (controller-runtime; an earlier Go skeleton is in git history,
  commit `4153875`): the logic is the 3 steps above.
- The ping is a real MCP request (`{"method": "ping"}` to `/mcp`), not `/healthz`: requests to
  `/healthz` through Kourier are answered by Knative itself and do not wake the tool.
