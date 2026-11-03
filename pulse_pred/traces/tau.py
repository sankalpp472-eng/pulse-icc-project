"""
Load tau-bench historical trajectories into episodes of parallel tool-call steps.

Source: github.com/sierra-research/tau-bench/historical_trajectories (MIT), stored in
data/tau_bench/: GPT-4o and Claude 3.5 Sonnet (new) runs on airline (50 tasks) and retail
(115 tasks), 4 and 8 trials per task.

What a step is
  The traces issue exactly one tool call per assistant turn. To model agents that call tools
  concurrently, consecutive calls are merged into one PARALLEL STEP when all of these hold:
    - every call in the step is read-only (get_/find_/search_/list_/calculate/think),
    - no user message and no agent message to the user comes between them,
    - no call uses a value that appeared in the result of an earlier call in the same step
      (its arguments do not depend on that result).
  Writes are always their own step. All calls in a step start together; the agent waits for the
  slowest.

What the predictor may see at each hint (no oracle information)
  user_text   the latest user message
  agent_text  the agent's latest message to the user
  reasoning   text the agent wrote together with the step's tool calls (Sonnet writes it on every
              call; GPT-4o on ~8%), with the embedded <function_calls> markup removed
  result      the step's tool results (truncated), only once they have returned

Timing is not in the traces, so it is drawn from a seeded model: LLM think time before each step
(lognormal), a user-turn wait when a user message came before the step, and per-tool exec time.
"""

import json
import random
import re
from dataclasses import dataclass, field

from .. import paths
from ..tools import TOOL

DATA = paths.DATA / "tau_bench"
FILES = {  # file -> (domain, source model)
    "gpt-4o-airline.json": ("airline", "gpt-4o"),
    "gpt-4o-retail.json": ("retail", "gpt-4o"),
    "sonnet-35-new-airline.json": ("airline", "sonnet-35-new"),
    "sonnet-35-new-retail.json": ("retail", "sonnet-35-new"),
}
RESULT_CHARS = 400


@dataclass
class TimingConfig:
    think_median_s: float = 1.6
    think_sigma: float = 0.5
    user_wait_s: tuple = (2.0, 8.0)


@dataclass
class Step:
    tools: list          # tools called concurrently in this step
    reasoning: str
    user_text: str
    agent_text: str
    result: str
    user_turn_before: bool
    think_s: float = 0.0
    user_wait_s: float = 0.0
    exec_s: list = field(default_factory=list)
    args: list = field(default_factory=list)          # call arguments, one dict per tool (for the ready set)
    results_full: list = field(default_factory=list)  # untruncated results, one per tool (for the ready set)
    selfhint: list = None                             # live runs: the agent's forked self-hint for the NEXT step
                                                      # [(tool or END, p)], taken with the step's last call
    call_ids: list = field(default_factory=list)      # tool-call ids of the step's calls, in order


@dataclass
class Episode:
    domain: str
    model: str
    task_id: int
    trial: int
    reward: float
    first_user_text: str
    steps: list


_MARKUP = re.compile(r"<function_calls>.*?(</function_calls>|$)", re.S)


def _clean(text):
    return re.sub(r"\s+", " ", _MARKUP.sub(" ", text or "")).strip()


def _arg_values(arguments):
    """Argument values of a call. GPT-4o traces store arguments as a JSON string, Sonnet's as a dict."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return []
    else:
        args = arguments or {}
    out = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif v is not None and len(str(v)) >= 4:
            out.append(str(v))
    walk(args)
    return out


def parse(record, domain, model):
    """One tau-bench record -> Episode with parallel steps (timings filled in by add_timing)."""
    steps, cur = [], None
    user_text = agent_text = first_user = ""
    user_since_last_step = False
    pending = {}                      # tool_call_id -> index of the call inside cur
    boundary = True                   # a user/agent-to-user message closes the current step

    hint_by_call = {}
    for h in record.get("selfhints", []):
        if h.get("tool_call_id"):
            hint = [(f"{domain}.{n}", p) for n, p in h["hints"]]
            if hint and sum(p for _, p in hint) == 0:  # tools named but no probabilities: weight by rank
                hint = [(t, w) for (t, _), w in zip(hint, (0.5, 0.3, 0.2))]
            if h.get("end"):
                hint.append(("END", h["end"]))
            hint_by_call[h["tool_call_id"]] = hint

    def close():
        nonlocal cur
        if cur:
            steps.append(Step(cur["tools"], " ".join(t for t in cur["reasoning"] if t), cur["user_text"],
                              cur["agent_text"], " || ".join(r[:RESULT_CHARS] for r in cur["results"]),
                              cur["user_turn_before"], args=cur["args"], results_full=cur["results"],
                              selfhint=hint_by_call.get(cur["ids"][-1]), call_ids=cur["ids"]))
        cur = None

    for m in record["traj"]:
        role = m["role"]
        if role == "user":
            close()
            user_text = m.get("content") or ""
            first_user = first_user or user_text
            user_since_last_step = True
            boundary = True
        elif role == "assistant" and not m.get("tool_calls"):
            close()
            agent_text = _clean(m.get("content"))
            boundary = True
        elif role == "assistant":
            for call in m["tool_calls"]:
                name = f"{domain}.{call['function']['name']}"
                if name not in TOOL:
                    raise KeyError(f"unknown tool {name} in task {record['task_id']}")
                vals = _arg_values(call["function"].get("arguments"))
                joinable = (cur is not None and not boundary and TOOL[name].read_only
                            and all(TOOL[t].read_only for t in cur["tools"])
                            and not any(v in r for v in vals for r in cur["results"]))
                if not joinable:
                    close()
                    cur = {"tools": [], "reasoning": [], "results": [], "args": [], "ids": [], "user_text": user_text,
                           "agent_text": agent_text, "user_turn_before": user_since_last_step}
                    user_since_last_step = False
                args = call["function"].get("arguments")
                cur["args"].append(json.loads(args or "{}") if isinstance(args, str) else (args or {}))
                cur["tools"].append(name)
                cur["ids"].append(call["id"])
                cur["reasoning"].append(_clean(m.get("content")))
                pending[call["id"]] = len(cur["results"])
                cur["results"].append("")
                boundary = False
        elif role == "tool" and cur is not None and m.get("tool_call_id") in pending:
            cur["results"][pending.pop(m["tool_call_id"])] = m.get("content") or ""
    close()
    return Episode(domain, model, record["task_id"], record.get("trial", 0), record.get("reward", 0.0),
                   first_user, steps)


def add_timing(ep, rng, cfg=TimingConfig()):
    for s in ep.steps:
        s.think_s = rng.lognormvariate(0, cfg.think_sigma) * cfg.think_median_s
        s.user_wait_s = rng.uniform(*cfg.user_wait_s) if s.user_turn_before else 0.0
        s.exec_s = [rng.uniform(*TOOL[t].exec_s) for t in s.tools]
    return ep


def load(files=tuple(FILES), seed=0, cfg=TimingConfig()):
    """All episodes with at least one tool call, timings drawn from `seed`."""
    rng = random.Random(seed)
    eps = []
    for f in files:
        domain, model = FILES[f]
        for rec in json.loads((DATA / f).read_text()):
            ep = parse(rec, domain, model)
            if ep.steps:
                eps.append(add_timing(ep, rng, cfg))
    return eps


def split_by_task(eps, val_frac=0.1, test_frac=0.2, seed=252):
    """Train / val / test split BY TASK (all trials and both models of a task stay together),
    so the test set never contains a task the predictor has seen."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    tasks = sorted({(e.domain, e.task_id) for e in eps})
    rng.shuffle(tasks)
    n_test, n_val = round(len(tasks) * test_frac), round(len(tasks) * val_frac)
    which = {t: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, t in enumerate(tasks)}
    for e in eps:
        out[which[(e.domain, e.task_id)]].append(e)
    return out["train"], out["val"], out["test"]
