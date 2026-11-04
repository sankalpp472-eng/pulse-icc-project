# Plan: observability, load testing, retraining, IaC and GitOps

Status: proposed, nothing implemented yet. Each phase gets its own branch and PR, and updates
`ui/GUIDE.md` and `README.md`.

Rules for every phase:
- **Ask the owner first** before anything that:
  - costs AWS money (EKS time, `terraform apply`);
  - changes another team member's files (show the diff first);
  - runs tests or load runs;
  - commits.
- **Simulated mode keeps working without a cluster.** A new feature either works there too, or the
  UI says it needs Knative.
- **Fair measurements:** load generators and monitoring never run on the tool nodes. On EKS they run
  on the client node.
- **Both UI setups are supported:** the laptop (`ui/server.py`) and the UI pod in EKS
  (`EKS_deployment/cloud/ui_cloud.sh`).

UI terms used below:
- **Tabs:** Live demo, Experiments, Results, How it works.
- **Side cards:** Predictor, Tool pods, Prewarm activity.
- **Top bar:** Simulated vs Knative.
- **Backend:** starts and watches processes (predictor, controller), runs steps with a streamed log,
  and lists result folders.

## Phase 0: housekeeping

1. Review and commit the root `README.md`. Push `ajinkya-pulse-v3` and open a PR.
2. Cut one branch per phase from it (`observability`, `loadtest`, `retraining`, `iac-gitops`).

## Phase 1: observability (Prometheus + Grafana in the cluster)

Status: implemented on branch `observability`, but not yet installed on a cluster or tested
end to end.

Deviations from the plan:
- **Runner metrics:** the runners didn't publish any metrics before. `w1_replay.py` now publishes
  them through `--metrics-port`. `w1_live.py` and the UI's interactive episodes don't yet.
- **Knative panels:** Knative's metrics are scraped, but have no panels yet. Their names still need
  checking on a cluster.

### Cluster side

1. **Install:** `deploy/monitoring/install.sh` plus Helm values for kube-prometheus-stack.
   - **Components:** Prometheus Operator, Prometheus, Grafana, kube-state-metrics, node-exporter.
     Alertmanager off, short retention.
   - **Placement:** on EKS, the client node (nodeSelector/tolerations). On kind, a small profile,
     because Docker has 4 GB.
   - **Grafana:** anonymous read-only viewing, embedding allowed, served under a sub-path, so the UI
     can embed panels. Reached with `kubectl port-forward` only (ClusterIP, no public address).
2. **Scrape targets:**
   - `pulse-predictor` and `pulse-speculator` (`/metrics`);
   - the runner pods (`test/test_knative/pulse_metrics.py`, port 9105);
   - Knative activator, autoscaler and queue-proxy; Kourier;
   - kube-state-metrics and node-exporter.

   Check the actual Knative 1.23 metric names on a cluster before writing any query that uses them.
3. **New metrics:**
   - `controller/prewarm_controller.py`: prewarms, pings, `was_cold`, ping latency, evictions, warm
     duration. This is a team file, so show the diff first.
   - `pulse_pred/service.py`: a `/metrics` endpoint with the same metric names as
     `predictor/service.py`, so one dashboard covers every predictor.
4. **Dashboards as code** (`deploy/monitoring/dashboards/*.json`, loaded through ConfigMaps):
   - **Pulse overview:**
     - tool latency p50/p95/p99, cold vs warm;
     - cold calls per minute;
     - ready pods per tool and warm pod-seconds (NFR-7);
     - prewarms vs hits.
   - **Predictor:**
     - request latency p95 against a 50 ms line (NFR-4);
     - hints/s;
     - prewarm candidates per tool;
     - CRD writes and errors.
   - **Cluster:** node CPU and memory (to check the CPU-contention explanation for slow cold
     starts), Knative activator queue, autoscaler desired vs actual pods.
5. **EKS bring-up:** an optional `install.sh` step in `eks_up.sh` / `cloud_up.sh`.

### UI integration

6. **Top bar, Monitoring status:**
   - **Detection:** the backend detects Grafana in Knative mode.
   - **Laptop:** the UI starts and watches the Grafana port-forward itself, like the predictor and
     controller.
   - **EKS:** `ui_cloud.sh up` opens both tunnels (UI and Grafana).
   - **Install button:** shown when monitoring is missing. It runs `install.sh` through the
     existing step runner, with the log streamed.
7. **New Monitoring tab:**
   - **Panels:** embedded panels of the three dashboards, plus "open in Grafana".
   - **Time range:** follows the current context: the live episode, the running experiment, or the
     selected result.
   - **Simulated mode:** the tab says it needs Knative mode.
8. **Key figures in existing cards:**
   - **Backend:** a `/api/monitoring/query` endpoint passes a few Prometheus queries through.
   - **Predictor card:** live p95 vs 50 ms.
   - **Tool pods card:** cluster-measured pod-seconds next to the UI's own count.
9. **Results tab:** every run gets a Grafana link covering its start-to-end window, also stored in
   `run_config.json`.

## Phase 2: load testing (Locust + asyncio)

Locust runs closed-loop tests: a set number of users, ramped over time. asyncio runs open-loop
tests: requests arrive at an exact rate whether or not earlier ones have finished, which is the
right shape for an SLO check.

### Load test scripts

1. **`loadtest/` with three Locust tests:**
   - **`locust_predictor.py`:** each user replays one session's hints in order, including `history`
     (pulse-v3 keeps per-session state). A load-shape class steps through 1 → 5 → 10 → 20 hints/s.
   - **`locust_tools.py`:** MCP calls through Kourier. Read-only tools only, so the shared Redis
     database isn't changed. Scenarios: a burst against cold tools, a steady load, a ramp.
   - **`locust_episodes.py`:** each user is an agent replaying a recorded episode with its real think
     and customer-reply gaps, sending hints and calling tools. It reuses `w1_replay.py`'s episode data
     and timing. It ramps from 1 to N concurrent agents, run as C0 and as C3 for each predictor.
2. **asyncio:**
   - extend `predictor/loadtest.py`, which is already asyncio + httpx, as the precise NFR-4 check;
   - add an open-loop tool-burst generator for exactly repeatable cold-start storms.
3. **Outputs:** `results/load_runs/<date>_<target>_<engine>_<config>/` with `summary.json`, Locust's
   CSV and HTML reports, and `run_config.json`. `plot_load.py` draws latency, pods and errors against
   load.
4. **Where it runs:**
   - **EKS:** a pod on the client node (Locust added to the runner image).
   - **kind:** the laptop.

### UI integration

5. **Experiments tab, Load test mode** next to Replay and Live:
   - **Inputs:** target (predictor / tools / episodes), engine (Locust / asyncio), users and spawn
     rate or a staged shape, open-loop rate, duration, predictor and θ, C0/C3.
   - **Backend reuse:** the existing step machinery. It sets min-scale, starts the predictor and
     controller for C3, runs the test, streams the Run log, and shows the summary when done.
6. **Interactive option:** start Locust with its web UI and link or embed it. Headless is the default,
   for repeatable runs.
7. **Monitoring tab:** follows the running load test automatically.
8. **Results tab:**
   - **Listing:** load runs appear with their own columns (requests/s, p50/p95/p99, failures, peak
     users).
   - **Compare and plots:** Compare uses suitable charts, and Make plots calls `plot_load.py`.
   - **Zip download:** works as for other runs.
9. **Live demo, "background load" option** (Knative only): run a demo episode while N simulated agents
   load the cluster.
   - The predictor API load test also works in simulated mode, because the predictor runs on the
     laptop.

## Phase 3: retraining (MLOps)

### Pipeline

1. **Training and the gate:** `pulse_pred/train.py` trains, evaluates on held-out tasks (hit@1/3,
   warm@k) and publishes with `model_store.publish`, with the metrics in the manifest. It promotes the
   model to `latest` only if it beats the current one; otherwise it keeps it as a candidate
   (`--no-latest`).
2. **New `model_store.py` commands:** `list` (versions, metrics, which is latest) and `promote`.
3. **Traces:** runs and live sessions upload episodes and hints to `s3://pulse-models-<account>/traces/`.
4. **Training Job:**
   - a Kubernetes Job (on demand, optionally a nightly CronJob) with an embedding-server sidecar;
   - its own pod IAM role (IRSA) with `pulse-model-publisher`, defined in Terraform;
   - the embedding cache kept in S3.
5. **Rollout:** the predictor's manifest pins an exact model version. Promotion is a one-line change,
   which can later go through GitOps (Phase 4).

### UI integration

6. **New Models card or tab:**
   - **Versions:** every version from S3 with its date, held-out metrics and notes, plus which is
     `latest` and which the running predictor serves.
   - **Use this version:** restarts the pulse-v3 predictor pinned to that version.
   - **Promote to latest:** asks for confirmation first.
   - **Retrain:** launches the training Job, streams its log in Run log, and shows the gate's verdict.
7. **Predictor card:** shows the served model version, read from `/healthz`.
8. **S3 access:** the laptop UI uses the user's AWS credentials. The in-cluster UI pod needs its own
   IAM role (Phase 4); until then the Models card works from the laptop only, and the card says so.

## Phase 4: infrastructure as code + GitOps

### Infrastructure

1. **Terraform:**
   - remote state in S3 with locking (`versions.tf` already has the backend, commented out);
   - EKS, node groups and ECR in Terraform, replacing `eksctl`. This changes the team's `eks_up.sh`
     workflow, so it needs team agreement;
   - pod IAM roles for the predictor (reader), the training Job (publisher) and the UI.
2. **CI:** GitHub Actions authenticating to AWS through OIDC (no stored keys). Tests, runner and UI
   image builds to ECR, `terraform plan` on PRs.
3. **GitOps:** Argo CD syncs `deploy/` from git: tools, CRDs, controller, predictors, monitoring and
   the UI itself.

### UI integration

4. **Configuration:** the UI reads Terraform outputs (bucket URI, roles) from a ConfigMap instead of
   hand-set environment variables.
5. **Top bar, Deployment status:**
   - the deployed git commit and image tag, extending the existing `/api/version` stale-code warning;
   - Argo CD sync status and a link to Argo CD, port-forwarded the same way as Grafana;
   - in GitOps mode, the Models card's "Use this version" opens a PR that pins the version, instead
     of restarting the predictor directly.
6. **How it works tab:** updated architecture covering the model registry, monitoring, load testing,
   retraining and GitOps.

## Already done (branch `ajinkya-pulse-v3`)

- **`pulse_pred/`:** the pulse-v3 predictor behind the shared `/predict` API, selectable in the UI as
  `pulse_v3`.
- **S3 model registry:** `pulse_pred/model_store.py` and `terraform/` (bucket `pulse-models-<account>`,
  reader and publisher IAM policies). Applied in ap-south-1; version `2026-11-03` is published.
- **Root `README.md`:** how to run the project. Not committed yet.
