# Shared steps of the team experiment scripts (experiments/team/script*.sh). Sourced, not run.
#
# Every script does, in order (each step is skipped when it is already done, so the same command resumes):
#   preflight   tools installed, logged in to AWS (your own account), Docker running, confirmation
#   python      .venv with the project's packages, tau-bench data
#   train       lightgbm (+ lightgbm_adapt, which is lightgbm with an online adapter) and markov, airline and retail,
#               on the FIXED held-out tasks below; the held-out lists and the models' checksums are checked / recorded
#   cluster     EKS (infra/eks/eks_up.sh), the runner image (build_image.sh), Pulse's services (cloud_up.sh),
#               the retail tools (deploy_retail.sh) and Prometheus + Grafana (infra/monitoring/install.sh)
#   run         the script's experiments: every run in results/team/<script>/manifest.tsv (resumable)
#   telemetry   per experiment: Prometheus export (CPU, memory, pods, nodes) + summary tables and plots
#   publish     experiments/results/<script>/ committed on exp/<script>-<you>, pushed, pull request opened
#   down        delete the EKS cluster (asks; AUTO_DOWN=1 deletes without asking)
#
# Settings (environment): TEAM_MEMBER (required: your name, for the branch), REGION (ap-south-1), TRIALS (3),
#   YES=1 (no questions), AUTO_DOWN=1, KEEP_CLUSTER=1 (never delete the cluster), PUBLISH=0 (do not push), ONLY="exp1 exp2" (run only these experiments),
#   SKIP_TRAIN=1 / SKIP_CLUSTER=1 (reuse what is there), ARMS="armA armB" (run only these arms),
#   PYTHON=<an existing venv's python> (skip creating .venv)
# Local test on kind instead of EKS (no AWS): CLUSTER=kind (the current kubectl context must be a kind-* cluster;
#   results go to results/team/<script>-kind, nothing is pushed, the cluster is kept), QUICK=1 (minutes-long runs:
#   6 episodes, 10 s per hint rate, 1 cold-start round), MONITORING=0 (no Prometheus: small laptops)

set -uo pipefail
export MSYS_NO_PATHCONV=1              # Git Bash on Windows: keep pod paths as they are
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never   # git never waits for a password prompt overnight   # Windows: Python output to a log file is cp1252 otherwise (θ, ≤ ...)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ---- the held-out tasks (no predictor is trained on them) and the tasks every replay uses
AIRLINE_HELDOUT="6 11 20 21 22 26 30 38 45 46"            # data/predictor/steps.csv, split=test
AIRLINE_REPLAY="6 20 22 26 30"                              # run_cloud.sh WORKLOAD=heldout (46 episodes)
RETAIL_HELDOUT="3 14 19 21 23 26 28 29 38 42 44 48 50 51 57 67 68 72 73 82 94 98 106"   # retail_split.json
RETAIL_REPLAY="$RETAIL_HELDOUT"                             # all 23, 46 episodes like airline
PREDICTORS_USED="lightgbm_adapt markov"
# what the trained models predict (pulse/experiments/model_fingerprint.py): the same on every laptop, so every
# teammate's runs use identical models. A different value stops the script (ALLOW_MODEL_MISMATCH=1 to go on anyway)
FINGERPRINT_AIRLINE=a697978ccce9727138c72f46dc3cca8d1f2ff06a4fabfa71afb1bf7056383006
FINGERPRINT_RETAIL=ac60a495efc549715d9c3f3db9bbd9bb06564846a29a9b9c9cd69fcb59302f62

SCRIPT=${SCRIPT:?set by the experiment script}
TEAM_MEMBER=${TEAM_MEMBER:-}
REGION=${REGION:-ap-south-1}
export REGION
TRIALS=${TRIALS:-3}
CLUSTER=${CLUSTER:-eks}               # eks | kind (local test)
QUICK=${QUICK:-0}
ARMS=${ARMS:-}
MONITORING=${MONITORING:-1}
[ "$CLUSTER" = kind ] && PUBLISH=${PUBLISH:-0}
[ "$QUICK" = 1 ] && PUBLISH=${PUBLISH:-0}      # a smoke test is never published
RUN_CLOUD=${RUN_CLOUD:-infra/cloud/run_cloud.sh}       # (tests replace it)
RETRY_WAIT_S=${RETRY_WAIT_S:-120}
# EKS nodes are x86: an Apple Silicon Mac would otherwise build arm64 images they cannot run
[ "$CLUSTER" = kind ] || export DOCKER_DEFAULT_PLATFORM=linux/amd64
# quick (smoke-test) runs get their own folder: the real runs must not skip them as done
OUT=results/team/$SCRIPT$([ "$CLUSTER" = kind ] && echo -kind)$([ "$QUICK" = 1 ] && [ "$CLUSTER" != kind ] && echo -quick)
# EKS set up by any script (the same cluster serves every script of this account): one marker for all of them
CLUSTER_MARK=$([ "$CLUSTER" = kind ] && echo "$OUT/.cluster_up" || echo "results/team/.cluster_up-eks-$REGION")
MANIFEST=$OUT/manifest.tsv
PUBLISH_DIR=experiments/results/$SCRIPT
LOGDIR=$OUT/logs
mkdir -p "$OUT" "$LOGDIR" infra/eks/logs
touch "$MANIFEST"

say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
die()  { say "ERROR: $*"; exit 1; }
ask()  {   # ask "question" -> 0 for yes
  [ "${YES:-0}" = 1 ] && return 0
  local a; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]
}
native() { command -v cygpath >/dev/null && cygpath -m "$1" || echo "$1"; }   # a path kubectl.exe / python.exe can open

# ------------------------------------------------------------------ preflight
step_preflight() {
  [ -n "$TEAM_MEMBER" ] || die "set TEAM_MEMBER=<your name> (used for your branch: exp/$SCRIPT-<name>)"
  [[ "$TEAM_MEMBER" =~ ^[A-Za-z0-9._-]+$ ]] || die "TEAM_MEMBER may only have letters, digits, '.', '_' and '-' (it names a git branch)"
  local missing="" tools="git aws eksctl kubectl helm docker curl"
  export PATH="$HOME/.arkade/bin:$PATH"
  [ "$CLUSTER" = kind ] && tools="git kubectl helm docker curl kind"
  for t in $tools; do command -v "$t" >/dev/null || missing="$missing $t"; done
  [ -z "$missing" ] || die "not installed:$missing (experiments/team/README.md, Prerequisites)"
  command -v gh >/dev/null || say "gh (GitHub CLI) not installed: results will be committed and pushed, open the PR by hand"
  docker info >/dev/null 2>&1 || die "Docker is not running (needed to build the runner and retail tool images)"
  if [ "$CLUSTER" = kind ]; then
    KUBE_CONTEXT=$(kubectl config current-context 2>/dev/null)
    [[ "$KUBE_CONTEXT" == kind-* ]] || die "CLUSTER=kind: the current kubectl context is '$KUBE_CONTEXT', not a kind cluster"
    kubectl get nodes --request-timeout=15s >/dev/null 2>&1 || die "the kind cluster $KUBE_CONTEXT does not answer"
    export KUBE_CONTEXT
    say "LOCAL TEST on $KUBE_CONTEXT | member $TEAM_MEMBER | script $SCRIPT | trials $TRIALS | quick $QUICK"
    return 0
  fi
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || die "not logged in to AWS: aws configure"
  say "AWS account $ACCOUNT | region $REGION | member $TEAM_MEMBER | script $SCRIPT | trials $TRIALS"
  if [ ! -f "$OUT/.confirmed" ]; then
    echo "This creates (or reuses) the EKS cluster 'pulse-eks' in AWS account $ACCOUNT, region $REGION:"
    echo "  4 x m7i-flex.large + the EKS control plane, billed while it runs (delete it at the end: step 'down')."
    ask "Use YOUR OWN AWS account $ACCOUNT for this?" || die "stopped"
    date > "$OUT/.confirmed"
  fi
}

# ------------------------------------------------------------------ python + data
PY=""
step_python() {
  if [ -n "${PYTHON:-}" ] && "$PYTHON" -c "import pulse, lightgbm, matplotlib" >/dev/null 2>&1; then
    PY=$PYTHON                                      # an existing environment with the project installed
    "$PY" -m pulse.setup.download_data >/dev/null || die "download_data failed"
    return 0
  fi
  local base="" b
  # the first one that really runs and is 3.11+ (Windows: "python3" may be the Microsoft Store placeholder)
  for b in python3.11 python3 python "py -3.11" py; do
    $b -c "import sys; sys.exit(sys.version_info < (3, 11))" >/dev/null 2>&1 && { base=$b; break; }
  done
  [ -n "$base" ] || [ -d .venv ] || die "no Python 3.11 or newer found (install Python 3.11, then rerun)"
  if [ ! -d .venv ]; then
    say "creating .venv ($base) and installing the project (a few minutes)"
    $base -m venv .venv || die "could not create .venv"
  fi
  PY=$( [ -x .venv/Scripts/python.exe ] && echo .venv/Scripts/python.exe || echo .venv/bin/python )
  if [ ! -f .venv/.pulse-installed ] || [ requirements.txt -nt .venv/.pulse-installed ]; then
    "$PY" -m pip install -q --upgrade pip && "$PY" -m pip install -q -r requirements.txt -e . || die "pip install failed"
    touch .venv/.pulse-installed
  fi
  export PYTHON="$PY"
  "$PY" -m pulse.setup.download_data >/dev/null || die "download_data failed"
}

# ------------------------------------------------------------------ training
step_train() {
  if [ "${SKIP_TRAIN:-0}" = 1 ] && [ -f models/action_model.pkl ] && [ -f models/action_model_retail.pkl ]; then
    say "training skipped (SKIP_TRAIN=1)"
  else
    say "training lightgbm + markov: airline (held out: $AIRLINE_HELDOUT)"
    "$PY" -m pulse.setup.train_models action > "$LOGDIR/train_airline.log" 2>&1 || die "airline training failed: $LOGDIR/train_airline.log"
    say "training lightgbm + markov: retail (held out: $RETAIL_HELDOUT)"
    "$PY" -m pulse.setup.train_models action_retail > "$LOGDIR/train_retail.log" 2>&1 || die "retail training failed: $LOGDIR/train_retail.log"
  fi
  # the held-out tasks must be exactly the agreed ones, or the four scripts' results are not comparable
  "$PY" - "$AIRLINE_HELDOUT" "$RETAIL_HELDOUT" "$OUT/models.json" "$FINGERPRINT_AIRLINE" "$FINGERPRINT_RETAIL" \
    "${ALLOW_MODEL_MISMATCH:-0}" <<'PY' | tee -a "$OUT/progress.log"
import csv, json, sys
from pathlib import Path
from pulse.experiments.model_fingerprint import fingerprint
want_air, want_ret, out = set(map(int, sys.argv[1].split())), set(map(int, sys.argv[2].split())), Path(sys.argv[3])
want_fp, allow = {"airline": sys.argv[4], "retail": sys.argv[5]}, sys.argv[6] == "1"
air = {int(r["task_id"].rsplit("-", 1)[1]) for r in csv.DictReader(open("data/predictor/steps.csv", encoding="utf-8"))
       if r["split"] == "test" and r["task_id"].startswith("airline-")}
ret = set(json.load(open("data/predictor/retail_split.json", encoding="utf-8"))["test_tasks"])
ok = True
for name, got, want in (("airline", air, want_air), ("retail", ret, want_ret)):
    print(f"  {name} held-out tasks: {sorted(got)}" + ("" if got == want else f"  <-- EXPECTED {sorted(want)}"))
    ok &= got == want
fps = {d: fingerprint(d) for d in ("airline", "retail")}
for d, fp in fps.items():
    same = fp == want_fp[d]
    print(f"  {d} model fingerprint {fp[:16]}... " + ("(as expected)" if same else f"<-- EXPECTED {want_fp[d][:16]}..."))
    ok &= same or allow
out.write_text(json.dumps({"airline_heldout": sorted(air), "retail_heldout": sorted(ret), "fingerprints": fps,
                           "fingerprints_expected": want_fp}, indent=2))
open(str(out) + ".ok", "w").write("1" if ok else "0")
PY
  [ "$(cat "$OUT/models.json.ok" 2>/dev/null)" = 1 ] || die "held-out tasks or models differ from the agreed ones (above): results would not be comparable"
  cp data/predictor/retail_split.json "$OUT/retail_split.json"
}

# ------------------------------------------------------------------ cluster
KCFG=""
pin_kubeconfig() {   # this script's own kubeconfig: a later `kubectl config use-context` elsewhere cannot redirect runs
  KCFG=$(native "$ROOT/$OUT/kubeconfig")
  aws eks update-kubeconfig --name pulse-eks --region "$REGION" --kubeconfig "$KCFG" >/dev/null || return 1
  export KUBECONFIG="$KCFG"
  export KUBE_CONTEXT=$(kubectl config current-context)
}
cluster_ready() {
  pin_kubeconfig 2>/dev/null || return 1
  kubectl get deploy pulse-predictor pulse-controller --request-timeout=20s >/dev/null 2>&1 &&
    [ -n "$(kubectl get ksvc -l pulse/role=mcp-tool -o name 2>/dev/null)" ] &&
    [ -n "$(kubectl get ksvc -l pulse/role=mcp-tool-retail -o name 2>/dev/null)" ] &&
    kubectl get ns monitoring >/dev/null 2>&1 && [ -f infra/cloud/.image-ecr ] && [ -f "$CLUSTER_MARK" ]
}
step_cluster_kind() {   # the same services on the local kind cluster (no EKS, no ECR)
  if kubectl get deploy pulse-predictor pulse-controller >/dev/null 2>&1 && [ -f infra/cloud/.image-kind ] &&
     [ -n "$(kubectl get ksvc -l pulse/role=mcp-tool-retail -o name 2>/dev/null)" ] &&
     { [ -f "$OUT/.cluster_up" ] || [ "${SKIP_CLUSTER:-0}" = 1 ]; }; then
    say "kind cluster ready (image $(cat infra/cloud/.image-kind))"; return
  fi
  [ -n "$(kubectl get ksvc -l pulse/role=mcp-tool -o name 2>/dev/null)" ] || {
    say "airline tools on kind"
    KUBE_CONTEXT="$KUBE_CONTEXT" SKIP_BUILD=1 PYTHON="$PY" bash infra/knative/deploy.sh 0 > "$LOGDIR/deploy_airline.log" 2>&1 ||
      die "deploy.sh failed: $LOGDIR/deploy_airline.log"; }
  kubectl apply -f infra/prewarm/prewarmrequest-crd.yaml >/dev/null
  say "runner image -> kind (a few minutes)"
  TARGET=kind bash infra/cloud/build_image.sh > "$LOGDIR/build_image.log" 2>&1 || die "build_image failed: $LOGDIR/build_image.log"
  say "Pulse services on kind"
  TARGET=kind EMBED=0 PREDICTOR=lightgbm_adapt bash infra/cloud/cloud_up.sh > "$LOGDIR/cloud_up.log" 2>&1 ||
    die "cloud_up failed: $LOGDIR/cloud_up.log"
  say "retail tools on kind"
  KUBE_CONTEXT="$KUBE_CONTEXT" PYTHON="$PY" bash infra/knative/deploy_retail.sh > "$LOGDIR/deploy_retail.log" 2>&1 ||
    die "deploy_retail failed: $LOGDIR/deploy_retail.log"
  if [ "$MONITORING" = 1 ]; then
    say "Prometheus + Grafana on kind"
    bash infra/monitoring/install.sh > "$LOGDIR/monitoring.log" 2>&1 || say "monitoring failed (runs go on): $LOGDIR/monitoring.log"
  fi
  date > "$OUT/.cluster_up"
  record_environment
}
step_cluster() {
  [ "$CLUSTER" = kind ] && { step_cluster_kind; return; }
  if cluster_ready; then
    say "cluster ready (context $KUBE_CONTEXT, image $(cat infra/cloud/.image-ecr))"
    if [ "${REBUILD:-0}" = 1 ]; then          # code pulled since the cluster was set up: into the image and services
      say "REBUILD=1: runner image from the current code -> your ECR, then the Pulse services"
      TARGET=ecr bash infra/cloud/build_image.sh > "$LOGDIR/build_image.log" 2>&1 || die "build_image failed: $LOGDIR/build_image.log"
      EMBED=0 PREDICTOR=lightgbm_adapt bash infra/cloud/cloud_up.sh > "$LOGDIR/cloud_up.log" 2>&1 || die "cloud_up failed: $LOGDIR/cloud_up.log"
      record_environment
    fi
    [ -f "$OUT/environment.txt" ] || record_environment
    return
  fi
  [ "${SKIP_CLUSTER:-0}" = 1 ] && die "SKIP_CLUSTER=1 but the cluster is not ready"
  say "EKS cluster + Knative + airline tools (about 25 min the first time)"
  PYTHON="$PY" bash infra/eks/eks_up.sh > "$LOGDIR/eks_up.log" 2>&1 || die "eks_up failed: $LOGDIR/eks_up.log"
  pin_kubeconfig || die "no kubeconfig for pulse-eks"
  say "runner image (with the models just trained) -> your ECR"
  TARGET=ecr bash infra/cloud/build_image.sh > "$LOGDIR/build_image.log" 2>&1 || die "build_image failed: $LOGDIR/build_image.log"
  say "Pulse services (predictor, controller, embeddings) on the system node"
  EMBED=0 PREDICTOR=lightgbm_adapt bash infra/cloud/cloud_up.sh > "$LOGDIR/cloud_up.log" 2>&1 || die "cloud_up failed: $LOGDIR/cloud_up.log"
  say "retail tools (image -> your ECR)"
  bash infra/knative/deploy_retail.sh > "$LOGDIR/deploy_retail.log" 2>&1 || die "deploy_retail failed: $LOGDIR/deploy_retail.log"
  say "Prometheus + Grafana (telemetry)"
  bash infra/monitoring/install.sh > "$LOGDIR/monitoring.log" 2>&1 || die "monitoring failed: $LOGDIR/monitoring.log"
  date > "$CLUSTER_MARK"
  cluster_ready || die "cluster set-up finished but something is missing (see $LOGDIR)"
  record_environment
}
record_environment() {
  { echo "script: $SCRIPT | member: $TEAM_MEMBER | account: ${ACCOUNT:-?} | region: $REGION | trials: $TRIALS"
    echo "git: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))$(git diff --quiet || echo ' + local changes')"
    echo "runner image: $(cat infra/cloud/.image-ecr 2>/dev/null)"
    echo; kubectl get nodes -L node.kubernetes.io/instance-type,pulse/role,topology.kubernetes.io/zone
    echo; kubectl get ksvc -A -o custom-columns='NAME:.metadata.name,IMAGE:.spec.template.spec.containers[0].image,WINDOW:.spec.template.metadata.annotations.autoscaling\.knative\.dev/window'
    echo; kubectl get deploy -o wide
  } > "$OUT/environment.txt" 2>&1
}

# ------------------------------------------------------------------ runs (manifest: exp domain arm trial status dir start end)
wanted() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
is_done() { awk -F'\t' -v e="$1" -v d="$2" -v a="$3" -v t="$4" '$1==e && $2==d && $3==a && $4==t && $5=="done" {f=1} END {exit !f}' "$MANIFEST"; }
# run_one EXP DOMAIN ARM TRIAL [VAR=VALUE ...] -- [run_cloud.sh arguments ...]
run_one() {
  local exp=$1 domain=$2 arm=$3 trial=$4; shift 4
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift
  if [ -n "$ARMS" ] && [[ " $ARMS " != *" $arm "* ]]; then return 0; fi
  if is_done "$exp" "$domain" "$arm" "$trial"; then say "  skip (done): $exp $domain $arm t$trial"; return 0; fi
  if [ "$QUICK" = 1 ]; then                    # smoke test: the same run, minutes long (the last value of an option wins)
    case " ${envs[*]} " in
      *" MODE=load "*)      set -- "$@" --rates 1 5 --duration 10 ;;
      *" MODE=coldstart "*) set -- "$@" --reps 1 ;;
      *)                    set -- "$@" --episodes 6 --rate 6 ;;
    esac
  fi
  local log="$LOGDIR/${exp}_${domain}_${arm}_t${trial}.log" start end rc pod result
  say "  run: $exp | $domain | $arm | trial $trial   (log: $log)"
  start=$(date +%s)
  env "${envs[@]}" DOMAIN="$domain" LABEL="$SCRIPT-$exp-$arm-t$trial" bash "$RUN_CLOUD" "$@" > "$log" 2>&1
  rc=$?
  pod=$(grep -o 'pulse-run-[0-9-]*' "$log" | head -1)
  local try
  for try in 1 2 3 4 5 6 7 8 9 10; do          # rc 2: the laptop lost the cluster; the pod keeps its results 1 h
    [ "$rc" = 2 ] || break
    say "    lost the cluster during $pod; fetching it again in 2 min (try $try)"
    sleep "$RETRY_WAIT_S"
    env "${envs[@]}" DOMAIN="$domain" COLLECT="$pod" bash "$RUN_CLOUD" >> "$log" 2>&1
    rc=$?
  done
  end=$(date +%s)
  result=$(grep -o 'Result: [^ ]*' "$log" | tail -1 | cut -d' ' -f2)
  if [ "$rc" = 0 ] && [ -n "$result" ] && [ -d "$result" ]; then
    printf '%s\t%s\t%s\t%s\tdone\t%s\t%s\t%s\n' "$exp" "$domain" "$arm" "$trial" "$result" "$start" "$end" >> "$MANIFEST"
    say "    done in $(( (end - start) / 60 )) min: $result"
    export_telemetry "$exp" "$domain/$arm/t$trial" || true   # now, while Prometheus still has it
  else
    printf '%s\t%s\t%s\t%s\tfailed(rc=%s)\t%s\t%s\t%s\n' "$exp" "$domain" "$arm" "$trial" "$rc" "${pod:-}" "$start" "$end" >> "$MANIFEST"
    say "    FAILED (rc=$rc); see $log (rerun the script to retry it)"
  fi
}
replay_tasks() {   # the replay arguments of a domain's held-out tasks
  if [ "$1" = retail ]; then echo "--tasks $RETAIL_REPLAY --episodes 46"; else echo ""; fi
}

# ------------------------------------------------------------------ telemetry
export_telemetry() {   # export_telemetry EXP [RUN]: Prometheus -> $OUT/EXP/prometheus (one run, or all done runs)
  local exp=$1 run=${2:-} pf ok=1 i
  kubectl get svc prometheus-operated -n monitoring >/dev/null 2>&1 || return 1
  mkdir -p "$OUT/$exp"
  kubectl -n monitoring port-forward svc/prometheus-operated 19090:9090 > "$LOGDIR/port-forward.log" 2>&1 &
  pf=$!
  for i in $(seq 1 30); do                       # ready, not just started (Windows can take a while)
    curl -sf --max-time 3 http://127.0.0.1:19090/-/ready >/dev/null 2>&1 && break
    sleep 2
  done
  "$PY" -m pulse.experiments.prom_export --url http://127.0.0.1:19090 --manifest "$MANIFEST" --exp "$exp" \
    --out "$OUT/$exp/prometheus" ${run:+--run "$run"} >> "$LOGDIR/telemetry.log" 2>&1 || ok=0
  kill "$pf" 2>/dev/null; wait "$pf" 2>/dev/null
  [ "$ok" = 1 ] || say "  Prometheus export failed${run:+ for $run} (see $LOGDIR/telemetry.log)"
  return 0
}
step_telemetry() {   # step_telemetry EXP: Prometheus export (every done run again; earlier rows are kept) + summary
  local exp=$1
  wanted "$exp" || return 0
  say "telemetry: $exp"
  mkdir -p "$OUT/$exp"
  export_telemetry "$exp" ||
    say "  no Prometheus in the cluster: CPU / memory not exported (pod-seconds and latencies are in the runs)"
  "$PY" -m pulse.experiments.team_summary --manifest "$MANIFEST" --exp "$exp" --out "$OUT/$exp" \
    > "$OUT/$exp/summary.txt" 2>&1 || say "  summary failed (see $OUT/$exp/summary.txt)"
  say "  $OUT/$exp/$exp.md"
}

# ------------------------------------------------------------------ publish
step_publish() {   # never fatal: a failed push must not keep the cluster (and its bill) running
  [ "${PUBLISH:-1}" = 1 ] || { say "not published (PUBLISH=0$([ "$CLUSTER" = kind ] && echo ', local test')): results stay in $OUT"; return 0; }
  local branch="exp/$SCRIPT-$TEAM_MEMBER" orig
  orig=$(git rev-parse --abbrev-ref HEAD)
  say "packaging $PUBLISH_DIR on $branch"
  rm -rf "$PUBLISH_DIR"                          # before switching: untracked copies would block the switch
  if ! { git switch -c "$branch" 2>/dev/null || git switch "$branch"; }; then
    say "PUBLISH FAILED: cannot switch to $branch (results are safe in $OUT; publish later with: PUBLISH=1 SKIP_TRAIN=1 SKIP_CLUSTER=1 and the same command)"
    return 1
  fi
  mkdir -p "$PUBLISH_DIR/runs"
  cp -r "$OUT"/. "$PUBLISH_DIR"/
  rm -f "$PUBLISH_DIR/kubeconfig" "$PUBLISH_DIR/.confirmed" "$PUBLISH_DIR/models.json.ok"
  awk -F'\t' '$5=="done" {print $6}' "$MANIFEST" | while read -r d; do [ -d "$d" ] && cp -r "$d" "$PUBLISH_DIR/runs/"; done
  # the manifest as committed points at runs/<name>
  awk -F'\t' 'BEGIN{OFS="\t"} {sub(/^results\/(cloud|local)_runs\//, "runs/", $6); print}' "$MANIFEST" > "$PUBLISH_DIR/manifest.tsv"
  git add -f "$PUBLISH_DIR"
  git commit -q -m "results: $SCRIPT ($TEAM_MEMBER)" -m "$(head -3 "$OUT/environment.txt" 2>/dev/null)" || say "nothing new to commit"
  local pushed=1
  git push -u origin "$branch" || { pushed=0; say "PUSH FAILED: the results are committed on $branch; push it later with: git push -u origin $branch"; }
  if [ "$pushed" = 1 ] && command -v gh >/dev/null; then
    gh pr view "$branch" >/dev/null 2>&1 || gh pr create --base exp/team-scripts --head "$branch" \
      --title "Results: $SCRIPT ($TEAM_MEMBER)" --body "$(cat "$PUBLISH_DIR"/*/*.md 2>/dev/null | head -150)" || \
      say "open the pull request by hand: $branch -> exp/team-scripts"
  fi
  git switch "$orig" >/dev/null 2>&1 || say "still on $branch (git switch $orig to get back)"
  [ "$pushed" = 1 ] && say "published on $branch"
  return 0
}

# ------------------------------------------------------------------ down
step_down() {
  [ "$CLUSTER" = kind ] && { say "local kind cluster kept"; return 0; }
  [ "${KEEP_CLUSTER:-0}" = 1 ] && { say "cluster kept (KEEP_CLUSTER=1): delete it when done with  REGION=$REGION bash infra/eks/eks_down.sh"; return 0; }
  if [ "${AUTO_DOWN:-0}" = 1 ] || ask "Delete the EKS cluster now (about 10 min; results are already saved)?"; then
    say "deleting the cluster"
    REGION="$REGION" bash infra/eks/eks_down.sh > "$LOGDIR/eks_down.log" 2>&1 && rm -f "$CLUSTER_MARK" || say "eks_down failed: $LOGDIR/eks_down.log"
  else
    say "cluster kept: delete it when done with  REGION=$REGION bash infra/eks/eks_down.sh"
  fi
}

# ------------------------------------------------------------------ the whole thing
run_script() {   # run_script <function that runs the experiments> <experiment ids ...>
  local body=$1; shift
  step_preflight
  step_python
  step_train
  step_cluster
  say "experiments: $*"
  "$body"
  local e; for e in "$@"; do step_telemetry "$e"; done
  local failed; failed=$(awk -F'\t' '$5 ~ /^failed/' "$MANIFEST" | wc -l)
  local missing; missing=$(missing_runs)
  if [ -n "$missing" ]; then
    say "some runs did not finish:"; echo "$missing"
    say "rerun the same command to retry them (finished runs are skipped); publishing what is there"
  fi
  step_publish || true
  step_down
  say "finished ($failed failed attempt(s) recorded in $MANIFEST)"
}
missing_runs() {   # runs whose last attempt failed
  awk -F'\t' '{k=$1"|"$2"|"$3"|t"$4; s[k]=$5} END {for (k in s) if (s[k] != "done") print "  " k " " s[k]}' "$MANIFEST"
}
