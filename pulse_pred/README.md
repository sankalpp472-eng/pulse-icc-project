# pulse-v3 predictor (`pulse_pred/`)

A next-tool predictor that reads the conversation: the customer's messages, the tool calls with their
arguments, and the tool results. Its features don't depend on how a particular LLM writes:
- a tool graph built from the tool schemas (which IDs are known, so which tools are ready);
- environment-state flags (user identified, order or reservation state, the user said yes or no, ...);
- a sentence embedding of the latest user text and results.

A frozen model gives the next-step probabilities. An online adapter, one per agent model, corrects
them from finished sessions. It was trained on tau-bench's historical airline and retail trajectories.
The training code and experiments are in the original pulse repository.

It plugs into this repository as one more predictor. `service.py` serves the same `POST /predict`
API as `predictor/service.py` and writes `PrewarmRequest`s through `predictor/crd_writer.py`, so
the prewarm controller, the replay runner and the UI work with it unchanged.

## Run

The trained model isn't kept in git. It lives in the S3 model registry (`terraform/model_store.tf`),
one folder per version, with a manifest and a `latest.json` pointer. The service downloads it once
into `pulse_pred/data/models/<version>/`:

```bash
pip install -r pulse_pred/requirements.txt
export PULSE_MODEL_URI=s3://<bucket>/pulse-v3                 # latest; or .../pulse-v3/<version>
CRD_MODE=k8s uvicorn service:app --app-dir pulse_pred --port 8000                  # from the repo root
python test/test_knative/w1_replay.py --predictor-url http://127.0.0.1:8000 --hint-context ...
```

- **Model versions:** a retrained model becomes a new version with
  `python pulse_pred/model_store.py publish <file> s3://<bucket>/pulse-v3 --version <v>`. Without
  `PULSE_MODEL_URI` the service loads a local file instead (`PULSE_MODEL`, default
  `pulse_pred/data/predictor.pkl`).
- **UI:** choose **pulse-v3** in the Predictor card. Hints then carry the conversation (`history`),
  as for the LLM speculator.
- **Policy:** the prewarm policy settings (`THETA`, `PREWARM_POLICY`, `TOP_K`, `P_MIN`, `TTL_S`) are
  the same as the main predictor's. All settings are listed at the top of `service.py`.
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

Hints without `history` (e.g. `agent/hints.py`) still work from the tool names alone, with lower
accuracy.

## Layout

| Path | What |
|---|---|
| `service.py` | The `/predict` + `/healthz` API (see above) |
| `model_store.py` | The S3 model registry: `fetch` (cached, sha256-checked) and `publish` (a new version, never overwritten) |
| `contract/` | `Hint` / `End` (session events) and `Prediction` (next-step probabilities per tool) |
| `predictors/base.py` | The session-predictor interface pulse-v3 implements |
| `predictors/pulse_v3/` | `predictor.py` (`PulseV3Predictor`, bundle loader), `models.py`, `graph.py`, `state.py`, `embed.py`, `online.py` |
| `tools.py` | Tool catalog (domain-prefixed names, e.g. `airline.get_user_details`) and class space |
| `traces/tau.py` | `Episode` / `Step`, the session representation the features are built from |
| `data/` (gitignored) | Downloaded models and the embedding cache. The bundle is 19 MB and needs scikit-learn 1.9.1, as pinned in `EKS_deployment/cloud/requirements.txt`; each version's manifest records the scikit-learn version it was trained with. |

Not done yet: an in-cluster deployment. `run_cloud.sh` and `cloud_up.sh` deploy only the gbt
predictor and the speculator.
