# pulse-v3 predictor (`pulse_pred/`)

A next-tool predictor that reads the conversation: the customer's messages, the tool calls with their
arguments, and the tool results. Its features don't depend on how a particular LLM writes:
- a tool graph built from the tool schemas (which IDs are known, so which tools are ready);
- environment-state flags (user identified, order or reservation state, the user said yes or no, ...);
- a sentence embedding of the latest user text and results.

A frozen model gives the next-step probabilities. An online adapter, one per agent model, corrects
them from finished sessions. It was trained on tau-bench's historical airline and retail trajectories.
Train it from a fresh clone with `python scripts/download_data.py`, then `bash scripts/embed_server.sh`, then
`python scripts/train_models.py pulse_v3`; the model lands in `pulse_pred/data/predictor.pkl`. The original
experiments are in the original pulse repository.

It plugs into this repository as one of the four predictors: `predictor/service.py` with
`PREDICTOR=pulse_v3` (adapter: `predictor/predictors/pulse_v3.py`). Its `/info` says it reads
`trajectory` and `history`, so the agent, the replay and the UI send it the conversation.

## Run

The trained model isn't kept in git. It lives in the S3 model registry (`terraform/model_store.tf`),
one folder per version, with a manifest and a `latest.json` pointer. The service downloads it once
into `pulse_pred/data/models/<version>/`:

```bash
pip install -r pulse_pred/requirements.txt
export PULSE_MODEL_URI=s3://<bucket>/pulse-v3                 # latest; or .../pulse-v3/<version>
PREDICTOR=pulse_v3 CRD_MODE=k8s uvicorn service:app --app-dir predictor --port 8000      # from the repo root
python test/test_knative/w1_replay.py --predictor-url http://127.0.0.1:8000 --hint-timeout 10 ...
```

- **Model versions:** a retrained model becomes a new version with
  `python pulse_pred/model_store.py publish <file> s3://<bucket>/pulse-v3 --version <v>`. Without
  `PULSE_MODEL_URI` the service loads a local file instead (`PULSE_MODEL`, default
  `pulse_pred/data/predictor.pkl`).
- **UI:** choose **pulse-v3** in the Predictor card. Hints then carry the conversation (`history`),
  as for the LLM speculator.
- **Policy:** the prewarm policy settings (`THETA`, `PREWARM_POLICY`, `TOP_K`, `P_MIN`, `TTL_S`) are
  shared by all predictors (top of `predictor/service.py`); pulse-v3's own settings are at the top of
  `predictor/predictors/pulse_v3.py`.
- **Embedding server:** text embeddings come from a llama.cpp server running nomic-embed-text-v1.5
  on `PULSE_EMBED_BASE` (default `http://127.0.0.1:8091/v1`):
  ```bash
  llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --pooling mean -c 16384 --parallel 8 -b 2048 -ub 2048 --port 8091
  ```
  Without it the predictor still answers with a zero embedding, which lowers accuracy;
  `/healthz` then shows `"embedding": "unavailable"`.

## How a hint becomes a session

pulse-v3 follows one session per `episode_id`:
1. The first hint starts the session with the first customer message.
2. Each new tool in the trajectory is replayed into the session as a call, with its arguments
   from `history`, followed by a result, with the tool outputs from `history`.
3. A new customer message without a new tool step updates the user text.
4. A session idle for `SESSION_TIMEOUT_S` (300 s) ends, and the adapter learns from it.

Hints without `history` still work from the tool names alone, with lower accuracy.

## Retraining (MLOps)

`train.py` runs the whole pipeline: data → train → evaluate → gate → publish.

```bash
python pulse_pred/model_store.py data push "$PULSE_MODEL_URI"       # once: training data -> <uri>/_data/
python pulse_pred/train.py --uri "$PULSE_MODEL_URI"                  # here (needs the embedding server for new texts)
PULSE_MODEL_URI=... bash deploy/training/run_training.sh             # or as a Kubernetes Job on EKS
python pulse_pred/model_store.py list "$PULSE_MODEL_URI"              # versions + metrics; * = latest
python pulse_pred/model_store.py promote "$PULSE_MODEL_URI" <version> # also: roll back
```

1. **Data:** pulls `<uri>/_data/` into `pulse_pred/data/`: tau-bench trajectories, tool schemas and the
   embedding cache.
2. **Split:** by task (seed 252) into train, val (temperature) and test. The test tasks are held out,
   so the model has never seen them.
3. **Train:** the tool graph, then the V3 model.
4. **Evaluate:** hit@1, hit@3 and warm@1..3 on the held-out hints, for both the new model and the
   current `latest`. Each uses its own tool graph, on the same hints.
5. **Release gate:** with `--promote auto`, the new model becomes `latest` only if its **warm@3**
   beats the current one's. warm@3 is the share of next-step calls that would find a warm pod if
   the top 3 tools were prewarmed. Otherwise it's published as a candidate. `always` / `never`
   override the gate.
6. **Publish:** uploads a new version with the metrics in its manifest. New embeddings go back to
   `_data/` for the next training.

- **Training Job** (`deploy/training/`): runs on the client node, with an embedding-server sidecar
  (llama.cpp + nomic-embed-text). Its service account `pulse-trainer` gets the
  `pulse-model-publisher` policy through IRSA: run `eksctl create iamserviceaccount` (the script
  prints it). With the Terraform cluster and GitOps, `terraform/eks/irsa.tf` creates the role and
  `deploy/gitops/` the service account.
- **In the UI, Models tab:**
  - **Versions:** every version with its held-out metrics, which one is `latest` and which one the
    predictor serves.
  - **Use:** serves that version.
  - **Promote:** makes it the latest, or rolls back.
  - **Retrain:** runs here or as the EKS Job. The log streams in Experiments → Run log, and the
    gate's verdict appears in the tab.
- **Rollout:** a predictor loads `latest` when it starts. To pin a version, point `PULSE_MODEL_URI`
  at `.../pulse-v3/<version>`; this is what the UI's **Use** button does.
- **Current limitation:** retraining uses the tau-bench trajectories (airline + retail). The team's
  own run traces aren't converted into training data yet.

## Layout

| Path | What |
|---|---|
| `train.py` | Retraining: data from S3, train, evaluate on held-out tasks, release gate, publish |
| `model_store.py` | The S3 model registry: `fetch` (cached, sha256-checked), `publish` (a new version, never overwritten), `list`, `promote`, `data push/pull` |
| `contract/` | `Hint` / `End` (session events) and `Prediction` (next-step probabilities per tool) |
| `predictors/base.py` | The session-predictor interface pulse-v3 implements |
| `predictors/pulse_v3/` | `predictor.py` (`PulseV3Predictor`, bundle loader), `models.py`, `graph.py`, `state.py`, `embed.py`, `online.py`, `features.py` (training / evaluation features and metrics) |
| `tools.py` | Tool catalog (domain-prefixed names, e.g. `airline.get_user_details`) and class space |
| `traces/tau.py` | `Episode` / `Step`, the session representation the features are built from |
| `data/` (gitignored) | Downloaded models and the embedding cache. The bundle is 19 MB and needs scikit-learn 1.9.1, as pinned in `EKS_deployment/cloud/requirements.txt`; each version's manifest records the scikit-learn version it was trained with. |

In the cluster, `PREDICTOR=pulse_v3 bash EKS_deployment/cloud/run_cloud.sh` switches the
`pulse-predictor` Deployment to pulse-v3. It takes `PULSE_MODEL_URI` from your shell (run_cloud.sh sets it
on the Deployment), else from the `pulse-config` ConfigMap (GitOps). The pod reads S3 through IRSA
(`terraform/eks/irsa.tf`), so this needs the Terraform-built cluster. Its embeddings come from the
`pulse-embed` Deployment (llama.cpp + nomic-embed-text, the same server as the training Job's sidecar), which
`cloud_up.sh` and GitOps deploy next to the predictor (`PULSE_EMBED_BASE=http://pulse-embed:8091/v1`).
