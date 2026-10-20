# PrewarmRequest: predictor ↔ controller contract

```
agent ──hint──▶ predictor (/predict) ──writes .spec──▶ PrewarmRequest ◀──watches── controller ──▶ Knative service
                                                                     ──writes .status──┘
```

Files: `prewarmrequest-crd.yaml` (the CRD), `predictor/crd_writer.py` (the writer).

## What the predictor guarantees

- **One object per tool**, named after its Knative service (`search-direct-flight`), in namespace `default`.
- An object is written only when that tool's probability is **≥ θ**. Nothing is written for `END`.
- When the tool is predicted again, the predictor **patches `.spec` only**: new `requestedAt`, `probability`, `episodeId`. It never touches `.status`, and it never deletes objects.
- Fields: `tool`, `service`, `probability`, `threshold`, `ttlSeconds`, `desiredReplicas` (always 1 for now), `requestedAt` (RFC 3339, UTC), `episodeId`, `workflowId`.

## What the controller is expected to do (to agree with Nithish)

1. On create or update: make `spec.service` have `desiredReplicas` ready pods. Set `status.phase` to `Warming`, then `Ready`, and set `status.readyAt`.
2. Eviction: once `now > max(spec.requestedAt, status.lastActivityAt) + ttlSeconds`, let the service go back to zero and set `status.phase: Evicted`. Keep the object; the next prediction re-warms it.
3. Never block a real call: a tool at zero must still answer (Knative's activator does that). This is FR-8.

Open question for the controller: changing `autoscaling.knative.dev/min-scale` on the Knative **Service template** creates a new Revision, which itself starts a pod and can be slow. Check whether patching the existing Revision or PodAutoscaler is faster, and measure it.

## Try it

```bash
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml
CRD_MODE=k8s python -m uvicorn service:app --app-dir predictor --port 8000
kubectl get pwr -w          # objects appear as the agent runs
```
