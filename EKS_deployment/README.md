# Running C0 and C1 with the tools on AWS EKS

The same C0 / C1 baselines as `test/test_knative/C0_BASELINE_README.md`, with **one change: the 14
MCP tools (and their Redis database) run on an EKS cluster instead of the local kind cluster.**
The agent, the runners (`w1_live.py`, `w1_replay.py`), the traces and the plots stay on your laptop
and are unchanged. They just call the tools at a different address.

```
                 LOCAL (before)                                  EKS (this guide)
laptop: agent / w1_replay.py                     laptop: agent / w1_replay.py
   │ http://127.0.0.1:8081/mcp                      │ http://<AWS load balancer>/mcp
   ▼                                                ▼
kind cluster pulse-kn (Docker on the laptop)     EKS cluster pulse-eks (ap-south-1, 2 x t3.xlarge)
   Knative + Kourier, 14 tools, Redis               Knative + Kourier, 14 tools, Redis   (identical)
```

| Step | What | Time |
|---|---|---|
| 1 | Install `aws`, `eksctl`, `kubectl`; log in to AWS | one time |
| 2 | Create the EKS cluster | 15-20 min |
| 3 | Install Knative + Kourier (gets an AWS load balancer) | ~5 min |
| 4 | Deploy Redis + the 14 tools | ~3 min |
| 5 | Point your terminal at EKS, check the tools | 3 min |
| 6 | C0 baseline (replay; optionally live) | ~30 min |
| 7 | C1 baseline (same seed) | ~30 min |
| 8 | Plots | 1 min |
| 9 | **Delete the cluster** | ~10 min |

> **Cost:** EKS charges per hour for the control plane, plus the 2 EC2 nodes and the load balancer,
> for as long as the cluster exists. Check current prices on the AWS pricing pages, use your AWS
> credits if you have them, and **delete the cluster (step 9) when you're done for the day.**

> **Status:** written and syntax-checked, but **not yet run on AWS** (it creates paid resources).
> The Knative, tool and runner steps are the same ones tested on the local cluster.

All commands: **Git Bash, from the project root.**

---

## 1. Tools and AWS login (one time)

| Tool | Check | Install |
|---|---|---|
| AWS CLI v2 | `aws --version` | already installed on this laptop |
| eksctl | `eksctl version` | `arkade get eksctl --os mingw` (or `choco install eksctl`) |
| kubectl | `kubectl version --client` | comes with Docker Desktop |

```bash
export PATH="$HOME/.arkade/bin:$PATH"
```

Log in with an IAM user or role that may create EKS clusters, EC2 instances, VPCs and load
balancers, and set the region to `ap-south-1`:

```bash
aws configure
```
```bash
aws sts get-caller-identity
```

The second command must print your account. If it fails, nothing below will work.

## 2. Create the cluster (15-20 minutes)

```bash
eksctl create cluster -f EKS_deployment/cluster.yaml
```

`cluster.yaml` follows the PRD's cloud tier: **2 worker nodes, t3.xlarge (4 vCPU, 16 GiB), 50 GB
disk, region ap-south-1 (Mumbai)**. The node count is fixed (min = max = 2), so a cold start never
includes booting a new node. eksctl also creates the VPC and adds the cluster to your kubeconfig:

```bash
kubectl config current-context
```

It should print something like `you@pulse-eks.ap-south-1.eksctl.io`. Then:
```bash
kubectl get nodes
```
That should show 2 nodes, `Ready`.

## 3. Install Knative + Kourier (about 5 minutes)

```bash
ALLOW_CIDR="$(curl -s https://checkip.amazonaws.com)/32" bash EKS_deployment/setup_knative.sh
```

Same Knative Serving v1.23 + Kourier as the local setup. The difference: Kourier gets an **AWS
load balancer** with a public hostname instead of `localhost:8081`. `ALLOW_CIDR` restricts that load
balancer to **your current IP**, so the tools aren't open to the internet. If your IP changes
(different Wi-Fi), rerun this step. The script ends by printing `Ingress: http://<...>.elb.amazonaws.com`.

A brand-new load balancer's hostname can take a few minutes to resolve in DNS.

## 4. Deploy Redis + the 14 tools (about 3 minutes)

The same `deploy.sh` as locally, aimed at EKS with `KUBE_CONTEXT`:

```bash
KUBE_CONTEXT="$(kubectl config current-context)" SKIP_BUILD=1 bash deploy/knative/deploy.sh
```

It deploys Redis, loads the airline data into it, and deploys the 14 tools as Knative services with
**C0** settings (minimum scale 0, 60 s idle window) from the public image
`sankalps2003/pulse-mcp-tools:0.2`. The EKS nodes pull the image from Docker Hub the first time; after
that it's cached on the node (as on kind).

## 5. Point your terminal at EKS, then check (about 3 minutes)

**In every new terminal** where you run the agent or a runner:

```bash
source EKS_deployment/use_eks.sh
```

It switches `kubectl` to the EKS cluster and sets `MCP_URL_TEMPLATE` to the load balancer, so the
agent, `w1_live.py`, `w1_replay.py` and `check_tools.py` call the tools on EKS. Nothing else
changes. Check every tool, cold and warm:

```bash
agent/.venv/Scripts/python deploy/knative/check_tools.py
```

Expect 14/14 OK. **Warm calls now include the network trip from your laptop to Mumbai and back**
(tens of ms more than locally); cold calls include pod start-up on EKS.

## 6. C0 baseline

Replay of the recorded τ-bench traces, **same seed as the local runs**, so the request sequence is
identical and the results are directly comparable:

```bash
agent/.venv/Scripts/python test/test_knative/w1_replay.py
```

Replay of **our own agent's** live traces (e.g. task 9 with gpt-oss-120b):

```bash
agent/.venv/Scripts/python test/test_knative/w1_replay.py --traces test/test_knative/results/w1_live_20261021-152753
```

Live agent against EKS (Groq keys from `agent/.env`):

```bash
agent/.venv/Scripts/python test/test_knative/w1_live.py --backend groq --tasks 9 --agent-model openai/gpt-oss-120b --user-model openai/gpt-oss-20b
```

Results land in `test/test_knative/results/<run>/` as before. `run_config.json` records the
**cluster context** and **ingress address**, so EKS runs are easy to tell apart from local ones.

## 7. C1 baseline (same seed)

```bash
KUBE_CONTEXT="$KUBE_CONTEXT" SKIP_BUILD=1 bash deploy/knative/deploy.sh 1
```
```bash
agent/.venv/Scripts/python test/test_knative/w1_replay.py
```
```bash
KUBE_CONTEXT="$KUBE_CONTEXT" SKIP_BUILD=1 bash deploy/knative/deploy.sh
```

1. Switch the tools to C1 (minimum scale 1, always warm).
2. Run the same replay: it waits until all 14 tools are warm, then starts.
3. Switch back to C0.

## 8. Plots

```bash
python test/test_knative/plot_c0.py test/test_knative/results/<run folder>
```

## 9. Delete everything (do this when you're done)

```bash
source EKS_deployment/use_local.sh
```
```bash
eksctl delete cluster -f EKS_deployment/cluster.yaml
```

`use_local.sh` switches the terminal back to the local kind cluster. The delete removes the nodes,
the load balancer, the VPC and the cluster itself (about 10 minutes). Afterwards check the AWS
console (EC2 → Load Balancers, EKS) that nothing is left running.

---

## Things to know

- **Laptop-to-AWS latency is now in every call.** Warm calls get slower by your network's round
  trip to Mumbai, and Wi-Fi jitter can widen p95/p99. For the cleanest numbers, run the replay from a
  small EC2 instance in `ap-south-1` (clone the repo there and follow steps 5-8). That's the PRD's
  "device tier", with a stable network.
- **Cold starts on EKS differ from kind:** real nodes and network, but also real image pulls the
  first time each node runs a tool. Run `check_tools.py` once before a baseline so every node has
  the image cached, or record that the first calls included pulls.
- **Redis keeps its data in memory.** If its pod restarts, rerun step 4 (it reseeds).
- **Switching back and forth:** `source EKS_deployment/use_local.sh` / `source EKS_deployment/use_eks.sh`.
  Check which cluster a terminal uses with `kubectl config current-context`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `aws sts get-caller-identity` fails | `aws configure` again: wrong keys or region |
| eksctl: not authorized / quota exceeded | Your IAM user needs EKS, EC2, VPC, CloudFormation and IAM permissions; or the region's vCPU quota is too low |
| `setup_knative.sh` waits forever for a hostname | `kubectl describe svc kourier -n kourier-system` shows the AWS error |
| Tool calls time out right after setup | The new load balancer's DNS isn't live yet: wait 2-5 minutes |
| Calls refused / time out after changing network | Your IP changed: rerun step 3 with the new `ALLOW_CIDR` |
| `curl` / calls return 404 | Wrong `Host` header: use `use_eks.sh`, which sets it |
| Tool pods crash: "Redis ... has no airline data" | Redis restarted: rerun step 4 |
| `This looks like the LOCAL kind cluster` | Your kubectl context is still kind: `aws eks update-kubeconfig --region ap-south-1 --name pulse-eks` |

## Files

| File | Purpose |
|---|---|
| `cluster.yaml` | eksctl cluster definition (2 x t3.xlarge, ap-south-1, fixed size) |
| `setup_knative.sh` | Knative + Kourier on EKS, load balancer restricted to your IP |
| `use_eks.sh` / `use_local.sh` | Point a terminal at the EKS tools / back to the local kind cluster |
| `../deploy/knative/deploy.sh` | Same deploy as locally; `KUBE_CONTEXT` picks the cluster |
