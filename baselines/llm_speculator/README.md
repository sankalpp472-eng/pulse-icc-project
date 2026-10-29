# Baseline: LLM speculator (following Speculative Actions)

A predictor baseline for Pulse: a fast LLM reads the conversation so far and guesses the agent's
next action (one of the 14 airline tools, or `RESPOND`). This reimplements the *speculator* role
of Speculative Actions (Ye et al., 2026) for next-tool prediction. It is not a reproduction of
their system: theirs guesses the customer's next reply on retail, using OpenAI/Gemini models.

It is scored like the v2 predictor's models: same τ-bench airline runs, same task split (seed 252,
10 unseen tasks, 1,578 steps), same `evaluate.accuracy` and `evaluate.theta_sweep`. It also gets
timing scores, since a correct guess only helps if it comes back before the tool call.

## Pipeline (from the repo root, with the agent's venv: it has `openai` and `python-dotenv`)

```bash
agent/.venv/Scripts/python.exe baselines/llm_speculator/steps.py                  # 1. held-out steps + 300-step sample
agent/.venv/Scripts/python.exe baselines/llm_speculator/speculate.py --dry-run    # see one prompt + token estimate
agent/.venv/Scripts/python.exe baselines/llm_speculator/speculate.py              # 2. Groq, gpt-oss-20b, 300 steps
agent/.venv/Scripts/python.exe baselines/llm_speculator/score.py results/<run>    # 3. vs n-gram (+ lr/gbdt if trained)
```

* `steps.py` uses the v2 predictor's `tau_data.py` (in `predictor/` once the v2 patch is applied,
  in `predictor_v2/pulse_predictor_v2_files/predictor/` until then; `PREDICTOR_DIR` overrides).
  It downloads τ-bench's trajectories into that predictor's `data/tau/` if they are missing.
* `speculate.py` reads `GROQ_API_KEY_1..N` from `agent/.env`. Requests go round-robin over the
  keys, and a key that hits its daily limit is dropped. Rows are appended as they arrive, so a run
  cut off by the quota resumes with `--out results/<same folder>`.
* `score.py` fits the n-gram here (on the training tasks). It adds `lr` / `gbdt` when the v2
  `data/predictor.pkl` exists and was not trained with `--final`. Several runs (models) can be
  scored together if they used the same sample.

## Prompt (`prompt.py`)

Trimmed to fit Groq's free tier, ~1.1K prompt tokens per step:
tool names with one-line descriptions, every tool called so far, the last 10 messages (each cut to
400 characters), no policy text and no parameter schemas. The answer is JSON with 4 actions and
probabilities. It is 4, not 3, so that "tool top-3" (RESPOND removed) still has 3 tools to rank.
Unparseable answers count as misses.

## Metrics (`score.py`)

| Column | Meaning |
|---|---|
| tool top1 / top3 | steps where the agent called a tool, ranking tools only (as in `evaluate.py`) |
| all top1 | all steps, tools + RESPOND |
| p50 / p95 s | prediction latency (LLM: the successful request only, rate-limit waits excluded) |
| timely top3 | tool in the top 3 **and** the guess back within `--lead-s` (1.0 s, hint → call) |
| cold hidden | share of the cold start (`--cold-s` 1.8 s) hidden: min(lead − latency, cold) / cold, 0 on a miss |

With a 1.0 s lead and a 1.8 s cold start, even an instant, perfect predictor hides at most 56%.

## Models on our Groq keys (25 Oct 2026)

`llama-3.1-8b-instant` is no longer served to these keys. The ones available are
`openai/gpt-oss-20b` (default), `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`.

Smoke test (20 sampled steps, gpt-oss-20b): 20/20 parsed, latency p50 0.61 s / p95 0.72 s,
about 1.1K prompt and 120 output tokens per step. That is far faster than the 2–17 s per guess in
the Speculative Actions logs (GPT-5 / Gemini), so on Groq the guess does arrive before the call.
It still uses up about 60% of the lead time. 20 steps is too few to compare accuracy.

## Files

| Path | What |
|---|---|
| `data/` (gitignored) | `airline_{train,test}_steps.jsonl`, `airline_test_runs.jsonl`, `airline_sample_300_seed252.json` |
| `results/<date>_<time>_<model>_<sample>/` | `predictions.jsonl` (one row per step), `run.json` (config, key log per session) |
| `results/score_<time>.json` | all methods on the same steps, with the θ sweep |
