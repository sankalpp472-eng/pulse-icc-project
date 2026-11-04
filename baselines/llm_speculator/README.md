# Baseline: LLM speculator (following Speculative Actions)

A predictor baseline for Pulse: a fast LLM reads the conversation so far and guesses the agent's
next action (one of the 14 airline tools, or `RESPOND`). This reimplements the *speculator* role
of Speculative Actions (Ye et al., 2026) for next-tool prediction. It is not a reproduction of
their system: theirs guesses the customer's next reply on retail, using OpenAI/Gemini models.

It is scored like the v2 predictor's models: same τ-bench airline runs, same task split (seed 252,
10 unseen tasks, 1,578 steps), same `evaluate.accuracy` and `evaluate.theta_sweep`. It also gets
timing scores, since a correct guess only helps if it comes back before the tool call.

## How it maps onto the paper (Algorithm 1)

| Paper | Here |
|---|---|
| Actor: slow, authoritative call that decides `a_t` | the agent's LLM step (tool + arguments, or RESPOND) |
| Speculator `g(s_t)`: fast model, k guesses | `speculate.py`: a fast Groq model, k guesses with confidences and arguments |
| Pre-launch the next call for each guess | prewarm the guessed tool (a ping: no side effects, so nothing to roll back) |
| Commit if the actor matches a guess | hit: the real call's tool was prewarmed |
| Breadth k, confidence-aware selection (Thm 4) | launch policies in `score_vs_action_model.py`: top-k, θ, greedy λ |
| Multi-model speculator, outputs pooled | `--models a,b`: models in parallel, distributions averaged |

## Pipeline (from the repo root)

```bash
python baselines/llm_speculator/steps.py --tau-bench ../tau-bench          # 1. held-out steps + 300-step sample
python baselines/llm_speculator/speculate.py --dry-run                    #    one prompt + the call budget
python baselines/llm_speculator/speculate.py                              # 2. list, k=3, gpt-oss-20b (300 calls)
python predictor/build_action_data.py --tau-bench ../tau-bench --split speculator   # 3. same steps as rows
python predictor/train_action.py --split speculator                       # 4. gbt / lr / ngram on that split
python baselines/llm_speculator/score_vs_action_model.py results/<run> [results/<run2> ...]   # 5. compare
```

`steps.py` uses `predictor/build_action_data.py`'s split (it used to need the uncommitted v2 `tau_data.py`);
it reproduces the 1,578 held-out steps and the committed 300-step sample exactly. `score.py` still needs v2.

## Speculation strategies (`speculate.py`)

| Strategy | LLM calls per step | Confidence | Paper |
|---|---|---|---|
| `--strategy list --k K` (default K=3) | 1 per model | stated by the model | chess: top-k predictions from one call |
| `--strategy sample --samples N` (default N=5, T=1.0) | N per model | vote share of N independent guesses | (self-consistency; measured instead of stated) |
| `--models a,b` (with either) | x number of models | average of the models | e-commerce: multi-model speculator |

The calls of a step run in parallel; the step's latency is the slowest call. Each guess includes the
tool's arguments, so the paper's strict match (tool AND arguments) is scored too.
Budget: 300 steps = 300 calls for `list`, 1,500 for `sample --samples 5`, 600 for `list` with 2 models.
Keys from the same Groq organization share one quota.

## Prompt (`prompt.py`)

~1.1K prompt tokens per step: tool names with one-line descriptions, every tool called so far, the last 10
messages (each cut to 400 characters), no policy text and no parameter schemas. The schema shows no example
numbers: with `0.55 / 0.25 / 0.1 / 0.05` in it, gpt-oss-20b copied them in 147 of 300 answers (run
`2026-10-25_1900`), so θ could not tell sure guesses from unsure ones. That run is still scored correctly
for top-k, but its θ / greedy rows mean little.

## Metrics (`score_vs_action_model.py`)

| Column | Meaning |
|---|---|
| tool top1 / top3 | steps where the agent called a tool, ranking tools only |
| all top1 | all steps, tools + RESPOND |
| p50 / p95 s | prediction latency (LLM: the slowest call of the step, rate-limit waits excluded) |
| timely top3 | tool in the top 3 **and** the guess back within `--lead-s` (1.0 s, hint → call) |
| cold hidden | share of the cold start (`--cold-s` 1.8 s) hidden: min(lead − latency, cold) / cold, 0 on a miss |
| launch policies | for every method: top-k (k = 1..4, with Proposition 1's independent p(k)), θ, greedy λ (Thm 4: add the next tool while p_j·Π(1 − p_i) ≥ λ); hit, pods and wasted per step, timely hit, cold hidden |
| args exact | LLM only: tool AND arguments right, top-1 / any kept guess (the paper's strict match) |
| cost | LLM only: calls, prompt and completion tokens per step |

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
