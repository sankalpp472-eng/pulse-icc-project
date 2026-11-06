"""
Step 2: the speculator. For every sampled held-out step, a fast LLM guesses the agent's next action.

How this maps onto Speculative Actions (Ye et al., Algorithm 1):
  Actor        the agent's LLM step: it decides the next call a_t (a tool + arguments, or RESPOND)
  Speculator   a fast LLM g(s_t) that returns k guesses {a_t^(1..k)}, each with a confidence
  Pre-launch   for each guessed TOOL, prewarm its function (a ping: no side effects, so the paper's
               Assumption 2 holds without any rollback; a guessed RESPOND launches nothing)
  Commit       the actor's real call hits if its tool was prewarmed
This script produces the k guesses; score_vs_action_model.py decides which of them to launch
(top-k, theta, or the paper's confidence-aware greedy rule) and scores time and cost.

Three ways to get the k guesses (--strategy):
  list     ONE call per model: "give k actions, most likely first, with p" (the paper's chess speculator:
           "top-k move predictions, ordered by confidence"). Cheapest; confidence = what the model states.
  sample   --samples N calls per model at --temperature (default 1.0), one guess each; confidence = vote
           share (self-consistency). N x the cost, but the confidence is measured, not stated.
  --models a,b   several models per step, in parallel, pooled by averaging their distributions (the paper's
           e-commerce "multi-model speculator": outputs "aggregated into a shared pool"). Works with both.
The calls of one step run in parallel; the step's latency is the slowest of them (all are needed to
aggregate). Each guess also predicts the tool's arguments, so the paper's strict match
(tool AND arguments) can be scored, not just the tool name that prewarming needs.

Keys: GROQ_API_KEY_1..N (or GROQ_API_KEY) from .env, round-robin; a key that hits its DAILY limit is
dropped. SPEC_BASE_URL + SPEC_API_KEY point it at any other OpenAI-compatible server (OpenRouter, Ollama).
Rows are appended as they come, so a run stopped by the quota resumes with --out <same folder>.

    python -m pulse.speculator.speculate --dry-run                         # one prompt + call budget
    python -m pulse.speculator.speculate                                   # list, k=3, gpt-oss-20b
    python -m pulse.speculator.speculate --k 4                             # k = 4 (tool top-3 even with RESPOND)
    python -m pulse.speculator.speculate --strategy sample --samples 5     # 5 votes per step
    python -m pulse.speculator.speculate --models openai/gpt-oss-20b,qwen/qwen3-32b   # pooled
    python -m pulse.speculator.speculate --out results/<folder>            # resume

Writes results/speculator/<date>_<time>_<models>_<strategy>_<sample>/run.json + predictions.jsonl
"""

import argparse
import json
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, NotFoundError,
                    OpenAI, PermissionDeniedError, RateLimitError)

from pulse import paths
from pulse.speculator.common import DATA, RESULTS, read_jsonl, step_key
from pulse.speculator.prompt import LAST_N, build_messages, domain_actions, parse_full

GROQ_URL = "https://api.groq.com/openai/v1"
MAX_RATE_WAIT_S = 120      # longer waits (or "per day") = that key's quota is used up


class QuotaExhausted(RuntimeError):
    pass


def api_keys():
    load_dotenv(paths.ENV_FILE)
    if os.getenv("SPEC_API_KEY"):
        return [os.environ["SPEC_API_KEY"]]
    numbered = sorted((k for k in os.environ if re.fullmatch(r"GROQ_API_KEY_\d+", k)),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
    keys = [os.environ[k] for k in numbered if os.environ[k].strip()]
    if not keys and os.getenv("GROQ_API_KEY"):
        keys = [os.environ["GROQ_API_KEY"]]
    if not keys:
        raise ValueError("No Groq key: set GROQ_API_KEY_1 (..._2, ...) or GROQ_API_KEY in .env")
    return keys


class KeyPool:
    """Round-robin over keys; per-minute 429s cool a key down, daily 429s drop it. Thread-safe."""

    def __init__(self, keys, timeout=60, base_url=None):
        base_url = base_url or os.getenv("SPEC_BASE_URL") or GROQ_URL
        self.clients = [OpenAI(base_url=base_url, api_key=k, max_retries=0, timeout=timeout) for k in keys]
        self.cool_until = [0.0] * len(keys)
        self.dead, self.next, self.log = set(), 0, []
        self.lock = threading.Lock()

    def pick(self):
        """-> (key index, seconds slept waiting for a key)."""
        slept = 0.0
        while True:
            with self.lock:
                live = [i for i in range(len(self.clients)) if i not in self.dead]
                if not live:
                    raise QuotaExhausted(f"all {len(self.clients)} key(s) hit their daily limit")
                now = time.time()
                for off in range(len(self.clients)):
                    i = (self.next + off) % len(self.clients)
                    if i in live and self.cool_until[i] <= now:
                        self.next = i + 1
                        return i, slept
                wait = max(0.05, min(self.cool_until[i] for i in live) - now)
            time.sleep(wait)                       # outside the lock: other threads keep going
            slept += wait

    def rate_limited(self, i, err):
        retry_after = err.response.headers.get("retry-after") if getattr(err, "response", None) else None
        wait = float(retry_after) if retry_after else 10.0
        msg = str(err).lower()
        with self.lock:
            if wait > MAX_RATE_WAIT_S or "per day" in msg or "per-day" in msg:
                self.dead.add(i)
                self.log.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "key": i + 1, "event": "daily limit",
                                 "error": str(err)[:300]})
                print(f"  [llm] key {i + 1} hit its daily limit; {len(self.clients) - len(self.dead)} key(s) left")
            else:
                self.cool_until[i] = time.time() + wait


def ask(pool, model, messages, max_tokens, json_mode, temperature=0.0, max_errors=3):
    """-> dict(text, latency_s, rate_wait_s, prompt_tokens, completion_tokens, key)."""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    if model.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"
    waited, errors = 0.0, 0
    while True:
        i, slept = pool.pick()
        waited += slept
        t0 = time.perf_counter()
        try:
            r = pool.clients[i].chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens, **extra)
        except RateLimitError as e:
            pool.rate_limited(i, e)
            continue
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code < 500:
                raise                                        # bad model name, bad request: retrying won't help
            errors += 1
            if errors > max_errors:
                raise
            print(f"  [llm] {type(e).__name__} ({errors}/{max_errors}): {str(e)[:120]}")
            time.sleep(2 ** errors)
            waited += 2 ** errors
            continue
        lat = time.perf_counter() - t0
        u = r.usage
        return {"text": r.choices[0].message.content, "latency_s": lat, "rate_wait_s": waited,
                "prompt_tokens": getattr(u, "prompt_tokens", None) or 0,
                "completion_tokens": getattr(u, "completion_tokens", None) or 0, "key": i + 1}


# ----------------------------------------------------------------- aggregation

def _args_id(args):
    return json.dumps(args or {}, sort_keys=True)


def model_distribution(answers, strategy):
    """Parsed answers of ONE model -> ({action: p}, {action: args}).
    list: the one answer's stated p (scaled down if they sum above 1).
    sample: vote share over the valid answers; args = the most common args among the votes."""
    answers = [a for a in answers if a[0]]
    if not answers:
        return None, None
    if strategy == "list":
        probs, args = answers[0]
        s = sum(probs.values())
        return ({a: p / s for a, p in probs.items()} if s > 1 else dict(probs)), args
    votes, arg_votes = Counter(), defaultdict(Counter)
    for probs, args in answers:
        top = max(probs, key=probs.get)
        votes[top] += 1
        arg_votes[top][_args_id(args.get(top))] += 1
    n = len(answers)
    return ({a: c / n for a, c in votes.items()},
            {a: json.loads(arg_votes[a].most_common(1)[0][0]) for a in votes})


def pool_models(per_model, k):
    """Average the models' distributions (models with no usable answer are left out), keep the top k."""
    usable = [(p, a) for p, a in per_model if p]
    if not usable:
        return None, None
    total, args_from = defaultdict(float), {}
    for probs, args in usable:
        for act, p in probs.items():
            total[act] += p / len(usable)
            if act not in args_from or p > args_from[act][0]:
                args_from[act] = (p, args.get(act, {}))
    top = sorted(total.items(), key=lambda kv: -kv[1])[:k]
    return dict(top), {a: args_from[a][1] for a, _ in top}


# ----------------------------------------------------------------- one step

def speculate_step(pool, ex, s, runs, cfg, actions, names):
    hist = runs[s["run_id"]]["messages"][:s["msg_index"]]
    single = cfg["strategy"] == "sample"
    msgs = build_messages(hist, s["trajectory"], actions, cfg["last_n"], k=cfg["k"], single=single)
    n_per_model = cfg["samples"] if single else 1
    jobs = [(m, j) for m in cfg["models"] for j in range(n_per_model)]

    def call(job):
        model, _ = job
        try:
            r = ask(pool, model, msgs, cfg["max_tokens"][model], cfg["json_mode"], cfg["temperature"])
            probs, args = parse_full(r["text"], names)
            return {"model": model, "ok": probs is not None, "probs": probs, "args": args, **r}
        except (QuotaExhausted, AuthenticationError, NotFoundError, PermissionDeniedError):
            raise
        except Exception as e:                       # noqa: BLE001  (record and move on)
            return {"model": model, "ok": False, "probs": None, "args": None,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    t0 = time.perf_counter()
    results = list(ex.map(call, jobs))                # the step's calls run in parallel
    wall = time.perf_counter() - t0
    per_model = [model_distribution([(r["probs"], r["args"]) for r in results if r["model"] == m],
                                    cfg["strategy"]) for m in cfg["models"]]
    pred, pred_args = pool_models(per_model, cfg["k"])
    lat = [r["latency_s"] for r in results if "latency_s" in r]
    return {"key": step_key(s), "action": s["action"], "true_args": s.get("args", {}),
            "pred": pred, "args": pred_args, "ok": pred is not None,
            "strategy": cfg["strategy"], "k": cfg["k"], "n_calls": len(jobs),
            "calls_ok": sum(r["ok"] for r in results),
            # latency of the speculation = the slowest successful call (rate-limit waits excluded)
            "latency_s": max(lat) if lat else None, "latency_first_s": min(lat) if lat else None,
            "wall_s": wall, "rate_wait_s": max((r.get("rate_wait_s", 0) for r in results), default=0),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in results),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in results),
            "calls": [{"model": r["model"], "ok": r["ok"], "latency_s": r.get("latency_s"),
                       "text": (r.get("text") or "")[:600], "error": r.get("error", "")} for r in results]}


def load_sample(sample_path):
    sample = json.loads(Path(sample_path).read_text())
    dom = sample["domain"]
    steps = {step_key(s): s for s in read_jsonl(DATA / f"{dom}_test_steps.jsonl")}
    runs = {r["run_id"]: r for r in read_jsonl(DATA / f"{dom}_test_runs.jsonl")}
    return dom, [steps[k] for k in sample["keys"]], runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default=str(DATA / "airline_sample_300_seed252.json"))
    p.add_argument("--models", "--model", default="openai/gpt-oss-20b",
                   help="one model, or several comma-separated (pooled). Groq ids on our keys (25 Oct): "
                        "openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b")
    p.add_argument("--strategy", choices=["list", "sample"], default="list")
    p.add_argument("--k", type=int, default=3, help="guesses kept per step (the paper's speculation breadth)")
    p.add_argument("--samples", type=int, default=5, help="sample strategy: calls per model per step")
    p.add_argument("--temperature", type=float, default=None, help="default 0 for list, 1.0 for sample")
    p.add_argument("--last-n", type=int, default=LAST_N, help="conversation messages shown")
    p.add_argument("--max-tokens", type=int, default=None, help="default 150, 1024 for gpt-oss (reasoning)")
    p.add_argument("--no-json-mode", action="store_true", help="don't send response_format=json_object")
    p.add_argument("--limit", type=int, default=0, help="only the first N sampled steps (0 = all)")
    p.add_argument("--out", default=None, help="existing results folder to resume")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and the call budget")
    a = p.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.strategy == "sample" and a.samples < 2:
        raise SystemExit("--strategy sample needs --samples >= 2 (vote shares)")
    dom, steps, runs = load_sample(a.sample)
    actions = domain_actions(dom)
    names = [n for n, _ in actions]
    if a.limit:
        steps = steps[:a.limit]
    cfg = {"models": models, "strategy": a.strategy, "k": a.k,
           "samples": a.samples if a.strategy == "sample" else 1,
           "temperature": a.temperature if a.temperature is not None else (1.0 if a.strategy == "sample" else 0.0),
           "last_n": a.last_n, "json_mode": not a.no_json_mode,
           "max_tokens": {m: a.max_tokens or (1024 if m.startswith("openai/gpt-oss") else 150) for m in models},
           "sample": str(a.sample), "domain": dom, "steps": len(steps)}
    calls_per_step = len(models) * cfg["samples"]

    if a.dry_run:
        s = steps[0]
        hist = runs[s["run_id"]]["messages"][:s["msg_index"]]
        msgs = build_messages(hist, s["trajectory"], actions, a.last_n, k=a.k, single=a.strategy == "sample")
        print(f"--- step {step_key(s)} (true action: {s['action']} {json.dumps(s.get('args', {}))})")
        for m in msgs:
            print(f"[{m['role']}]\n{m['content']}\n")
        est = statistics.mean(len(json.dumps(build_messages(runs[x["run_id"]]["messages"][:x["msg_index"]],
                                                            x["trajectory"], actions, a.last_n, k=a.k)))
                              for x in steps) / 3.5
        print(f"{a.strategy}, k={a.k}, models={models}: {calls_per_step} call(s) per step x {len(steps)} steps = "
              f"{calls_per_step * len(steps)} calls, ~{est * calls_per_step * len(steps) / 1000:.0f}K prompt tokens")
        return

    slug = "+".join(re.sub(r"[^A-Za-z0-9.-]+", "-", m.split("/")[-1]) for m in models)
    strat = f"{a.strategy}-k{a.k}" + (f"-n{a.samples}" if a.strategy == "sample" else "")
    out = Path(a.out) if a.out else RESULTS / f"{time.strftime('%Y-%m-%d_%H%M')}_{slug}_{strat}_{Path(a.sample).stem}"
    if not out.is_absolute() and not out.exists():
        out = paths.ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {r["key"] for r in read_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [s for s in steps if step_key(s) not in done]
    print(f"{strat} {models}: {len(todo)} steps to run ({len(done)} already done), "
          f"{calls_per_step} call(s) per step -> {out}")

    pool = KeyPool(api_keys())
    t_start, n_ok = time.time(), 0
    try:
        with open(pred_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=calls_per_step) as ex:
            for n, s in enumerate(todo, 1):
                row = speculate_step(pool, ex, s, runs, cfg, actions, names)
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_ok += row["ok"]
                if n % 10 == 0 or n == len(todo):
                    top = max(row["pred"], key=row["pred"].get) if row["pred"] else "-"
                    print(f"  {n}/{len(todo)}  ok {n_ok}  last: true {row['action']}, guess {top}, "
                          f"{row['latency_s'] or 0:.2f}s ({row['calls_ok']}/{row['n_calls']} calls ok)")
    except QuotaExhausted as e:
        print(f"stopped: {e}. Resume tomorrow with --out {out}")
    finally:
        run_path = out / "run.json"                          # one entry per session (resumes add one)
        sessions = json.loads(run_path.read_text()).get("sessions", []) if run_path.exists() else []
        sessions.append({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "steps_ok": n_ok,
                         "wall_s": round(time.time() - t_start, 1), "key_log": pool.log})
        run_path.write_text(json.dumps({**cfg, "model": "+".join(models), "sessions": sessions}, indent=1))


if __name__ == "__main__":
    main()
