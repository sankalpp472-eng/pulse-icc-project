# Pulse: session notes (Claude Code, Oct 2026)

A readable record of the long Claude Code session that built and evaluated Pulse: what was decided,
what was built, every run and where its results are, what broke and how it was fixed, and what is open.
The raw transcript (JSON Lines, ~29 MB) is at
`C:\Users\Sankalp\.claude\projects\C--Users-Sankalp-OneDrive-Documents-iisc-ICC-Pulse-project1\b31e5b61-99ef-4b89-8fb9-2e96ff98bf58.jsonl`.

---

## 1. The project in one paragraph

Pulse (DS252, IISc) predicts which MCP tool an LLM agent will call next and wakes that tool's
serverless pod before the call arrives, so the agent does not wait for a cold start. Configurations:
**C0** scale-to-zero, no prewarming (cold-start baseline); **C1** always warm, one pod per tool
(latency floor, cost ceiling); **C3** predictor + CRD controller prewarming (the proposed method).

---

## 2. How the work went (chronological)

1. **Design discussion.** Workloads, benchmarks and how to build the agent. Decision: a custom
   ReAct agent on τ-bench's airline environment (14 tools), LLM via the Groq API (free tier).
2. **ReAct agent built** (`agent/`): a plain loop and a LangGraph version, a simulated customer
   (LLM role-plays the τ-bench customer), local tools for scoring.
3. **OpenFaaS CE tested and dropped.** Scale-to-zero is a paid (Pro) feature; faas-netes put
   replicas back to 1 even with `scale.min=0`. So C0 could not exist. Moved to **Knative Serving
   1.23 + Kourier** on a local kind cluster `pulse-kn` (see the "Why we are moving from OpenFaaS CE
   to Knative" doc in the repo root).
4. **Tools as MCP servers.** Each of the 14 tools is its own MCP server (MCP Python SDK, Streamable
   HTTP) and its own Knative service; one image `docker.io/sankalps2003/pulse-mcp-tools:0.2`;
   shared Redis database, reset to a seed copy per run.
5. **C0 / C1 baselines on the laptop** with replayed τ-bench traces (W1 = 23 change/cancel tasks).
6. **Live Groq runs.** Hit Groq's limits (8K tokens per request, 200K tokens/day per key); added
   context trimming, 4-key rotation and retries. Recorded a clean live run of task 9 (gpt-oss-120b).
7. **Predictor integrated** (teammate's scikit-learn service in `predictor/`); agent sends a hint
   before every LLM step.
8. **CRD + controller, "Option A"**: the controller sends a warm-up MCP `ping`; Knative's 60 s idle
   window is the TTL; the controller never evicts.
9. **Task-9 comparison C0 / C3 / C1** on the laptop (first C3 result).
10. **C0 and C1 redefined** as the 5 W1 tasks 4, 8, 10, 23, 27 (chosen so all 14 tools are used).
11. **AWS EKS** (ap-south-1): C0 and C1 on the 5 tasks, then full teardown.
12. **Slides, architecture diagram, explanations** (agent flow, replay, episodes, seed, predictor).
13. **Held-out tasks 6, 20, 22, 26, 30** (not in the predictor's training data): live Groq agent +
    predictor + CRD on EKS, then C0 / C3 / C1 replays of those traces, **two trials**, then teardown.

---

## 3. Architecture

Diagram: `docs/architecture/pulse_architecture.png` (and `.svg`; regenerate with
`py docs/architecture/draw_architecture.py`).

| Tier | Component | Where |
|---|---|---|
| Client (laptop) | ReAct agent + simulated customer (Groq) | `agent/agent.py`, `agent/user.py`, `agent/llm.py` |
| Client | Trace replayer (no LLM) | `test/test_knative/w1_replay.py` |
| Client | Prediction service (FastAPI + scikit-learn) | `predictor/service.py`, `predictor/models.py` |
| Client | Prewarm controller (Python prototype, Option A) | `controller/prewarm_controller.py` |
| Cloud (kind or EKS) | Kourier ingress → Knative → 14 MCP tool services | `deploy/knative/` |
| Cloud | PrewarmRequest CRD (`pwr`) | `deploy/prewarm/prewarmrequest-crd.yaml` |
| Cloud | Redis (shared DB) | `deploy/knative/redis.yaml` |

Numbered flow: (1) hint → predictor, (2) tool call over MCP/HTTP via Kourier, (3) predictor writes a
PrewarmRequest for every tool with P ≥ θ, (4) controller watches it, (5) controller pings the tool
through the same ingress, which wakes the pod. TTL = Knative idle window (60 s).

---

## 4. Key concepts (as explained during the session)

- **Agent loop** (`agent/agent.py` `run_episode`): the task's `instruction` goes only to the
  simulated customer; the agent learns it by conversation. Each step: send hint → LLM call → either
  a tool call (run it, append result) or text to the customer (customer replies; `###STOP###` ends).
  `transfer_to_human_agents` also ends the episode. Max 30 steps.
- **Trace**: a recording of one past agent run (tool calls + arguments, in order).
  **Episode**: one replay of a trace against the live tools. **Seed 252**: fixes episode order,
  start times and every wait, so all configurations get identical calls at identical times
  (252 = the course number).
- **Replay waits**: think time (lognormal, median 1.5 s) + one customer turn (median 8 s) per
  recorded agent→customer message; our own live traces use their real recorded gaps. Episodes start
  as a Poisson process, 2 per minute, so several overlap. The waits decide cold vs warm (60 s
  window) and are C3's lead time.
- **Calls per run vary** (e.g. task 10: 2–14) because each task has 12 recorded runs by different
  LLMs taking different paths (extra searches, retries, early transfers).
- **Cold detection**: a new pod appeared during the call, or latency > 1 s. "Real cold start" = the
  pod signal.
- **Predictor** (`predictor/models.py`): P = 0.2·P_graph + 0.0·P_history + 0.8·P_trajectory
  (`p_ensemble`, line ~103); `p_graph` = uniform over allowed next tools; `p_history` = counts after
  the last tool; `p_trajectory` = counts after the last two tools, backing off to history;
  `prewarm_set` keeps P ≥ θ = 0.3 (never END). Weights chosen in `predictor/train.py`, saved in
  `predictor/data/model.pkl`. Trained on 480 τ-bench runs (40 tasks); 10 tasks held out by a seeded
  split in `predictor/export_traces.py` (held-out W1 tasks include 6, 20, 21, 22, 26, 30, 38, 45, 46).
  Tasks 4, 8, 9, 10, 23, 27 are in the **training** split.
- **Why EKS**: the PRD separates client and cloud tiers; the laptop distorted results (cold starts
  3–6 s, a 36.9 s outlier under memory pressure); EKS gives real cloud cold starts (~1.8 s) and a
  real request path. Downside: laptop→Mumbai network noise.

---

## 5. Workloads used

**Main set (EKS C0/C1): tasks 4, 8, 10, 23, 27** — all 14 tools used.
60 τ-bench traces (12 per task, Claude 3.5 Sonnet ×8, GPT-4o ×4) → 56 episodes, 484 calls
(4 traces have no calls). File: `test/test_knative/data/w1_eks5_traces.jsonl`.

| Task | Customer goal | Calls/run (avg) |
|---|---|---|
| 4 | Change passenger, upgrade to economy, 3 bags, gift card | 0–10 (7.6) |
| 8 | Sum balances, cheapest business round trip, cancel+rebook if needed | 0–16 (8.0) |
| 10 | Remove passenger (else cancel), book cheapest direct NY→West Coast | 2–14 (9.3) |
| 23 | Delay both legs, business, 2 bags, fee waiver, economy fallback | 2–13 (9.4) |
| 27 | Cancel two reservations, change a third to nonstop | 4–9 (6.0) |

**Held-out set (predictor never trained on them): tasks 6, 20, 22, 26, 30** — 10 of 14 tools
(missing list_all_airports, send_certificate, update_reservation_passengers,
update_reservation_baggages). τ-bench: 60 traces → 60 episodes, 413 calls. Task 26 is the same
customer as training task 27, so it is the least "unseen".

---

## 6. Every run and where its results are

All under `test/test_knative/results/` unless noted.

### Local kind cluster

| Run | Folder | Result |
|---|---|---|
| W1 τ-trace replay C0 (23 tasks) | `w1_replay_20261019-001516` | 46 eps, 386 calls, 84 cold |
| W1 τ-trace replay C1 | `w1_replay_20261021-161136` | 46 eps, 386 calls, 4 cold |
| Comparison | `compare_W1_C0_vs_C1` | |
| Live Groq, task 9 (the replayed trace) | `w1_live_20261021-152753` | 8 calls, gpt-oss-120b |
| Live agent + predictor (dry-run CRD, local tools), task 9 | `agent_predictor_20261022-121009` | top-1 25%, top-3 62.5%, prewarm hit 37.5% |
| Task 9 C0 (×3) | `w1_replay_20261022-132650` | 18/24 cold, mean 2.71 s, p95 5.1 s, 1,617 pod-s |
| Task 9 C3 (×3), real CRD + controller | `w1_replay_20261022-133636` (+ `controller/results/controller_task9_c3.jsonl`) | 12/24 cold, mean 1.73 s, p95 3.99 s, 1,969 pod-s; first two calls 4.31→0.20 s and 3.72→0.05 s |
| Task 9 C1 (×3) | `w1_replay_20261022-134839` | 0 real cold; one 36.9 s outlier (laptop out of memory); 5,025 pod-s |
| Your own Groq runs (no predictor) | `w1_live_20261022-223104`, `w1_live_20261022-223953` | tasks 4, 8, 10, 27 ok; 23 failed (customer-LLM error) |
| Earlier tests | `w1_live_20261018-*`, `w1_live_20261021-122140`, `w1_live_20261021-130356`, `w1_live_20261021-151157_task_9_w1`, `discarded/` | partial / rate-limited |

### AWS EKS, tasks 4, 8, 10, 23, 27 (τ-bench traces), 22 Oct

| | C0 `w1_replay_20261022-163742` | C1 `w1_replay_20261022-170618` |
|---|---|---|
| p50 / p95 / p99 | 317 ms / 1.89 s / 2.23 s | 307 ms / 541 ms / 1.30 s |
| Real cold starts | 86 (of 484) | 0 |
| Pod-seconds | 12,419 | 21,706 (+75%) |

Comparison + plots: `compare_EKS_W1_C0_vs_C1`. Cold starts on EKS 1.7–2.4 s; warm ~300 ms (mostly
laptop→Mumbai round trip). C1's 7 slow calls (incl. 20 s and 17.5 s) were network stalls.

### AWS EKS, held-out tasks 6, 20, 22, 26, 30, 23 Oct

**Live Groq agent + predictor + real CRD on EKS** (traces for the replays):
`w1_live_20261023-135307` (tasks 6, 26, 30: 5, 9, 9 calls) and `w1_live_20261023-145447`
(tasks 20, 22: 4, 7 calls). `w1_live_20261023-140954` = failed attempt (no calls). Hints:
`heldout5_groq_c3_hints.jsonl`; controller log: `controller/results/controller_heldout5_groq_c3.jsonl`.
Predictor accuracy (`heldout5_groq_c3_hint_scores.json`): **top-1 44%, top-3 76%, prewarm hit 50%**,
hint round trip 15 ms, lead time (hint → call) ~1.0 s. Offline guide numbers for comparison:
47% / 82% / 53%.

**Replays** (each trace ×3 = 15 episodes, 102 calls, seed 252, real recorded gaps):

| | C0 t1 | C0 t2 | C3 t1 | C3 t2 | C1 t1 | C1 t2 |
|---|---|---|---|---|---|---|
| Folder (`w1_replay_20261023-…`) | `150403` | `154506` | `151518` | `155612` | `152853` | `160734` |
| Real cold starts | 17 | 17 | 13 | 11 | 0 | 0 |
| Slow (>1 s), no pod start | 16 | 8 | 24 | 23 | 22 | 5 |
| p50 | 584 ms | 402 ms | 686 ms | 568 ms | 412 ms | 362 ms |
| p95 | 2.37 s | 2.11 s | 3.10 s | 2.58 s | 1.90 s | 1.00 s |
| p99 | 3.67 s | 3.02 s | 5.41 s | 3.32 s | 3.20 s | 1.61 s |
| Pod-seconds | 2,728 | 3,368 | 3,140 | 3,106 | 7,908 | 8,097 |

Comparisons: `compare_EKS_heldout5_groq_C0_C3_C1` (trial 1) and
`compare_EKS_heldout5_groq_C0_C3_C1_trial2`.

**What holds in both trials:** C3 cuts real cold starts 24–35% vs C0 and uses ~61% fewer
pod-seconds than C1 (PRD target ≥30%). **But** C3's latency percentiles are worse than C0, driven by
~23 slow calls to already-warm pods in each C3 trial. Ruled out: new pods, overlapping warm-up
pings, first call on a ping-woken pod. Leading hypothesis (unconfirmed): the laptop, since C3 is the
only setup where predictor + controller run beside the replay. The controller sent ~135 pings per
trial, only ~5 of which actually started a pod.

---

## 7. Problems hit and fixes

| Problem | Fix |
|---|---|
| OpenFaaS CE cannot scale to zero | Moved to Knative + Kourier |
| Knative readiness via `/_/health` → 500s | Image 0.2 with a fast `/healthz` |
| Pod watcher saw no pods | `pulse/role`, `pulse/tool` labels on the pod template |
| Groq 413 (8K tokens/request) | Context trimming calibrated on provider token counts; budget 6000 |
| Groq `tool_use_failed` from the agent | Retry (×3) in `_chat_within_limit` |
| Groq `tool_use_failed` from the **simulated customer** (tasks 20, 22, your 23) | Fix in `agent/user.py`: retry ×3, then use the text inside Groq's error (`_failed_text`) |
| Groq daily quota | 4 keys (`GROQ_API_KEY_1..4`, different orgs) with rotation; replay instead of live |
| Redis wiped on Docker restart / DNS not ready | `deploy.sh` reseeds and waits for coredns |
| Restarting 14 pods at once overloaded Redis | Restart one at a time |
| Excel resaved CSVs | Restored with git |
| C1 replay waited for zero pods | Waits for all-ready instead |
| Laptop overload (C1 36.9 s outlier) | Flagged; moved to EKS |
| EKS Free plan refused t3.xlarge | Nodegroup `workers-m7i`, 2 × m7i-flex.large (`EKS_deployment/cluster.yaml`) |
| Laptop public IP changed after a crash | Patched `loadBalancerSourceRanges` on the Kourier service |
| `eksctl delete cluster` failed twice (VPC stuck) | Deleted leftover worker network interfaces + cluster security group by hand, then the stack; verified nothing left |
| Laptop crash killed predictor/controller | Restarted; later stopped all leftover copies |

---

## 8. Code and files changed or added in this session

- `agent/`: agent loop, LangGraph agent, LLM client (Groq/OpenRouter/Ollama, key rotation,
  trimming), simulated customer (incl. the `tool_use_failed` fix), tool backends (local / MCP),
  MCP server, Dockerfile, task runner.
- `deploy/knative/`: `gen_services.py`, `deploy.sh` (`SKIP_BUILD`, `NO_REDIS`, `WINDOW`,
  `KUBE_CONTEXT`, arg `1` = C1), `redis.yaml`, `check_tools.py`.
- `deploy/prewarm/`: CRD (Option A status fields) and README.
- `controller/prewarm_controller.py`: Option A controller (ping, status, sweeper → Evicted).
- `test/test_knative/`: `w1_live.py`, `w1_replay.py`, `c0_common.py`, `plot_c0.py`,
  `plot_compare.py`, `score_hints.py`, `data/` traces, `C0_BASELINE_README.md`.
- `EKS_deployment/`: `cluster.yaml`, `setup_knative.sh` (`ALLOW_CIDR`), `use_eks.sh`,
  `use_local.sh`, README.
- `docs/architecture/`: diagram script and outputs.
- Teammate's: `predictor/*`, `agent/hints.py`.

---

## 9. Slide deck

https://claude.ai/artifact/UmxCo78WMi2aihYP6ZmBEJ (download as .pptx or PDF from the deck; private
until shared). 17 slides: cover, problem, architecture, workload (4, 8, 10, 23, 27), tool list,
tool usage, what is built, predictor, why Knative, EKS setup, C0 vs C1 results, CDF/cost plots,
first C3 (task 9), lessons, next steps, held-out workload (6, 20, 22, 26, 30), evaluation status.
Cover still has a `[Team ID / member names]` placeholder. The held-out C0/C3/C1 results are not on a
slide yet.

---

## 10. Evaluation status (PRD metrics)

| # | Metric | Exp | Status | Left |
|---|---|---|---|---|
| 1 | Tool-call latency p50/p95/p99 | E1, E2 | Partly | Startup breakdown (scheduling, readiness, init); more trials |
| 3 | Resources (warm pods, pod-seconds) | E4 | Partly | Warm-pods-vs-time plot; idle vs busy pod-seconds |
| 7 | Efficiency E | E2, E4 | Partly | Needs idle pod-seconds (computable from existing pods.csv/calls.csv) |
| 2 | Prediction API latency vs hint rate | E3 | To do | Load test (`predictor/loadtest.py`) |
| 4 | Scalability, p99 vs clients | E5 | To do | Replay C3 at higher rates |
| 5 | θ sensitivity | E6 | To do | Live θ sweep (offline curve exists) |
| 6 | TTL sensitivity | E7 | To do | Re-run with `WINDOW=30s/60s/120s` |

---

## 11. Open issues and next steps

1. **C3 slow warm calls**: test the laptop hypothesis (run C0 with predictor + controller busy but
   no prewarming), or move the client tier (replay, predictor, controller) to EC2 next to the cluster.
2. **Lead time** (~1.0 s hint→call vs ~1.8 s cold start): warm the top 2 tools, hint during
   customer turns, or predict two steps ahead.
3. **Pings mostly hit warm tools** (~130 of ~135): skip the ping when the tool already has a pod
   and was used recently.
4. **Controller in Go / in-cluster** (PRD), replacing the Python prototype.
5. **Full protocol**: 3 trials, 2-min warm-up + 10-min steady state, median with min/max.
6. **Teardown script** in `EKS_deployment/` that also deletes leftover network interfaces and the
   cluster security group (the delete failed twice without it).
7. Retrain the predictor on our own Groq runs once there are enough (predictor guide's first limit).
8. `EKS_deployment/README.md` still mentions t3.xlarge; update to m7i-flex.large.

---

## 12. How to rerun things (from the project root, Git Bash)

```bash
# Local cluster: tools at C0 (min-scale 0) or C1 (min-scale 1)
SKIP_BUILD=1 bash deploy/knative/deploy.sh
SKIP_BUILD=1 bash deploy/knative/deploy.sh 1

# EKS
eksctl create cluster -f EKS_deployment/cluster.yaml
ALLOW_CIDR="$(curl -s https://checkip.amazonaws.com)/32" bash EKS_deployment/setup_knative.sh
KUBE_CONTEXT="$(kubectl config current-context)" SKIP_BUILD=1 bash deploy/knative/deploy.sh
source EKS_deployment/use_eks.sh
kubectl apply -f deploy/prewarm/prewarmrequest-crd.yaml

# C3 helpers (two terminals)
CONTROLLER_LOG=controller/results/<name>.jsonl agent/.venv/Scripts/python controller/prewarm_controller.py
CRD_MODE=k8s agent/.venv/Scripts/python -m uvicorn service:app --app-dir predictor --port 8000

# Live Groq agent (+ predictor if PREDICTOR_URL is set)
PREDICTOR_URL=http://127.0.0.1:8000 WORKFLOW_ID=W1 agent/.venv/Scripts/python test/test_knative/w1_live.py --tasks 6 20 22 26 30 --backend groq --agent-model openai/gpt-oss-120b

# Replays (add --predictor-url http://127.0.0.1:8000 for C3)
agent/.venv/Scripts/python test/test_knative/w1_replay.py --traces test/test_knative/data/w1_eks5_traces.jsonl
agent/.venv/Scripts/python test/test_knative/w1_replay.py --traces test/test_knative/results/w1_live_20261023-135307 test/test_knative/results/w1_live_20261023-145447 --repeat 3

# Plots
py test/test_knative/plot_compare.py <C0 dir> <C3 dir> <C1 dir> --out test/test_knative/results/<name>

# EKS teardown (then check EC2, ELB, NAT, EIP, EBS, VPC, stacks in ap-south-1)
kubectl -n kourier-system delete service kourier
eksctl delete cluster -f EKS_deployment/cluster.yaml --wait --disable-nodegroup-eviction
source EKS_deployment/use_local.sh
```

---

## 13. Housekeeping notes

- Never commit or print `agent/.env` (Groq/OpenRouter keys). The git `origin` URL contains a GitHub
  token: revoke it and use a credential helper instead.
- AWS account 294056656243, region ap-south-1, Free plan (8 vCPU quota). As of 23 Oct the EKS
  cluster and every related resource are deleted (verified).
- Two old CloudWatch log groups from week-6 clusters (`/aws/eks/ds252-w6-eks-*`) were left alone.
- kubectl is on the local `kind-pulse-kn` context; the kind cluster needs Docker Desktop running.
