"""
The Setup tab: everything the scripts in scripts/ and infra/ do, as buttons. Each action becomes a few steps that run
in the UI's job runner (backend.BatchRun), so their output streams into the log like an experiment run, one job at
a time. Actions that cost money or delete things are marked `confirm` (the page asks first).

  status()           what is there: tools on PATH, data, models, LLM keys (names only), clusters, image, AWS account
  ACTIONS            name -> (title, confirm text or None, where it may run)
  build(name, p)     the steps for one action
  save_env(values)   write LLM keys / settings into .env (never read back to the page)
"""

import json
import re
import shutil

from pulse import paths
from pulse.setup import download_data, train_models
from pulse.ui import backend

ENV_KEYS = ("LLM_BACKEND", "GROQ_API_KEY", "GROQ_MODEL", "OPEN_ROUTER_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
            "USER_MODEL", "SPEC_BASE_URL", "SPEC_API_KEY", "SPEC_MODEL", *backend.REGISTRY_KEYS)
S3_URI = re.compile(r"s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9](/[\w./-]+)?")   # the model registry URIs
GROQ_NUMBERED = re.compile(r"GROQ_API_KEY_(\d+)")   # any number of extra Groq keys: GROQ_API_KEY_1, _2, ...
MAX_GROQ_KEYS = 20
SECRET = re.compile(r"KEY")               # values of these names are never shown, only "set"
PREDICTORS = ("lightgbm", "markov", "speculator", "pulse_v3")
CLOUD = paths.INFRA / "cloud"
IMAGE_FILE = {"ecr": CLOUD / ".image-ecr", "kind": CLOUD / ".image-kind"}

ACTIONS = {   # name: (what the button does, confirm text or None)
    "download_data": ("Download the tau-bench data", None),
    "train_action": ("Train lightgbm + markov (~1 min)", None),
    "train_action_retail": ("Train the retail lightgbm + markov (~1 min)", None),
    "embed_start": ("Start the embedding server (Docker)", None),
    "embed_stop": ("Stop the embedding server", None),
    "train_pulse_v3": ("Train pulse-v3 (~45 min, needs the embedding server)", None),
    "kind_create": ("Create the local kind cluster with Knative and the 14 tools (~5 min)", None),
    "kind_delete": ("Delete the local kind cluster", "Delete the kind cluster pulse-kn and everything in it?"),
    "retail_tools": ("Deploy the 16 retail tools into the kind cluster (~5 min)", None),
    "image_build": ("Build the runner image", None),
    "eks_create": ("Create the EKS cluster, Knative and the tools (~25 min)",
                   "Create an EKS cluster in your AWS account? It costs money by the hour (about $0.10 for the "
                   "control plane plus the EC2 nodes) until you delete it."),
    "cloud_up": ("Deploy Pulse's services into the current cluster", None),
    "set_predictor": ("Switch the cluster's predictor", None),
    "eks_delete": ("Delete the EKS cluster (~10 min)",
                   "Delete the EKS cluster and everything in it (nodes, load balancer, VPC)? Runs in the cluster "
                   "are lost; result folders on this machine are kept."),
    "eks_leftovers": ("Check AWS for leftovers (clusters, instances, load balancers)", None),
    "registry_access": ("Give the EKS cluster access to the model registry (IRSA roles)", None),
    "monitoring_install": ("Install Prometheus + Grafana into the current cluster", None),
    "gitops_configure": ("Write your AWS account into the GitOps manifests", None),
    "gitops_install": ("Install Argo CD and the GitOps applications", None),
    "tf_bootstrap": ("Terraform state bucket (pulse-tfstate-<account>, once per AWS account)",
                     "Create the S3 bucket pulse-tfstate-<account> for Terraform's state in your AWS account? "
                     "Needed once, before Terraform plan / apply. It costs cents a month."),
    "tf_plan": ("Terraform plan: the S3 model registry, ECR, CI roles", None),
    "tf_apply": ("Terraform apply: the S3 model registry, ECR, CI roles",
                 "Apply the Terraform plan to your AWS account (S3 bucket, IAM policies, ECR repository)? "
                 "Review the plan in the log first."),
    "registry_from_tf": ("Use the model registry Terraform created (fills in both URIs)", None),
    "registry_push_data": ("Upload the training data to the S3 registry", None),
    "registry_publish": ("Publish the local model as a registry version",
                         "Publish the local model file as a new registry version (latest only if that box is ticked)?"),
}


def _which(*names):
    return {n: bool(shutil.which(n)) for n in names}


def _env_values() -> dict:
    vals = {}
    if paths.ENV_FILE.exists():
        for line in paths.ENV_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                vals[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return vals


def _real(v: str) -> bool:
    return bool(v) and "your_" not in v                 # .env.example's placeholders are not keys


def _groq_values(vals: dict) -> dict:
    """The Groq keys .env has: GROQ_API_KEY and every GROQ_API_KEY_<n>, name -> value (real keys only)."""
    return {k: v for k, v in vals.items() if (k == "GROQ_API_KEY" or GROQ_NUMBERED.fullmatch(k)) and _real(v)}


def env_status() -> dict:
    """Which settings .env has: secret ones only as set / not set; placeholders count as not set. Groq keys only
    as a count (groq_keys: distinct keys the agent and the speculator rotate through)."""
    vals = _env_values()
    out = {}
    for k in ENV_KEYS:
        v = vals.get(k, "")
        ok = _real(v)
        out[k] = ("set" if ok else "") if SECRET.search(k) else (v if ok else "")
    groq = len(set(_groq_values(vals).values()))
    llm_ready = bool(groq) or bool(out.get("SPEC_BASE_URL")) or bool(out.get("OPEN_ROUTER_API_KEY")) \
        or out.get("LLM_BACKEND") == "ollama"
    return {"exists": paths.ENV_FILE.exists(), "values": out, "groq_keys": groq, "llm_ready": llm_ready}


def save_env(values: dict, groq_keys: list | None = None) -> dict:
    """Set the given names in .env (empty value = leave it); creates .env from .env.example if needed.
    groq_keys: new Groq keys to add, each under the next free name (GROQ_API_KEY first, then GROQ_API_KEY_1, _2,
    ...); a key .env already has is skipped."""
    unknown = [k for k in values if k not in ENV_KEYS]
    if unknown:
        raise ValueError(f"not a known setting: {unknown}")
    values = {k: str(v or "").strip() for k, v in values.items()}
    bad = [k for k in backend.REGISTRY_KEYS if values.get(k) and not S3_URI.fullmatch(values[k])]
    if bad:
        raise ValueError(f"{', '.join(bad)}: an S3 URI is needed, like s3://pulse-models-<account>/pulse-v3")
    if not paths.ENV_FILE.exists():
        example = paths.ROOT / ".env.example"
        paths.ENV_FILE.write_text(example.read_text(encoding="utf-8") if example.exists() else "", encoding="utf-8")
    values = dict(values)
    have = _groq_values(_env_values())
    taken, known = set(have), set(have.values())
    for key in groq_keys or []:
        key = str(key or "").strip()
        if not key or "\n" in key or key in known:
            continue
        if len(known) >= MAX_GROQ_KEYS:
            raise ValueError(f"at most {MAX_GROQ_KEYS} Groq keys")
        name = "GROQ_API_KEY" if "GROQ_API_KEY" not in taken else \
            f"GROQ_API_KEY_{next(n for n in range(1, 1000) if f'GROQ_API_KEY_{n}' not in taken)}"
        values[name] = key
        taken.add(name)
        known.add(key)
    lines = paths.ENV_FILE.read_text(encoding="utf-8").splitlines()
    for k, v in values.items():
        v = str(v or "").strip()
        if not v or "\n" in v:
            continue
        for i, line in enumerate(lines):
            if re.match(rf"\s*#?\s*{k}\s*=", line):
                lines[i] = f"{k}={v}"
                break
        else:
            lines.append(f"{k}={v}")
    paths.ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    backend.set_registry({k: values[k] for k in backend.REGISTRY_KEYS if values.get(k)})   # live: no UI restart
    return env_status()


def _tf_init(env: dict) -> tuple:
    """The step that points Terraform at its state bucket (pulse-tfstate-<account>, infra/terraform/bootstrap)."""
    acct = _aws_account()
    if not acct:
        raise ValueError("not logged in to AWS (aws configure)")
    return ("terraform init (state bucket pulse-tfstate-<account>)",
            ["terraform", "-chdir=infra/terraform", "init", "-input=false", f"-backend-config=bucket=pulse-tfstate-{acct}"],
            env)


def registry_from_terraform(say) -> None:
    """Save the registry URIs Terraform created (outputs model_uri, action_model_uri) and use them right away."""
    tf = paths.INFRA / "terraform"
    vals, missing = {}, []
    for key, output in (("PULSE_MODEL_URI", "model_uri"), ("ACTION_MODEL_URI", "action_model_uri")):
        code, out = backend._run(["terraform", f"-chdir={tf}", "output", "-no-color", "-raw", output], 60)
        if code == 0 and S3_URI.fullmatch(out.strip()):
            vals[key] = out.strip()
        else:                                   # e.g. an output added to the code after the last apply
            missing.append(output)
    if not vals:
        raise RuntimeError(f"Terraform has no registry outputs ({', '.join(missing)}): run Terraform apply first")
    save_env(vals)
    for k, v in vals.items():
        say(f"  {k}={v}")
    say("  saved to .env and in use now (Models tab)")
    if missing:
        say(f"  not in Terraform's state yet: {', '.join(missing)}. Run Terraform apply to add it, or type it above")


def _aws_account():
    code, out = backend._run(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"], 15)
    return out.strip() if code == 0 and out.strip().isdigit() else None


def _kind_exists():
    code, out = backend._run(["kind", "get", "clusters"], 10)
    return code == 0 and "pulse-kn" in out.split()


def _retail_tools() -> int:
    """How many retail tool services the kind cluster has (Deploy the retail tools)."""
    code, out = backend._run(["kubectl", "--context", "kind-pulse-kn", "get", "ksvc", "-l", backend.DOMAINS["retail"]["role"],
                              "-o", "name", "--request-timeout=6s"], 10)
    return len(out.split()) if code == 0 else 0


def _eks_context():
    code, out = backend._run(["kubectl", "config", "get-contexts", "-o", "name"], 10)
    return next((c for c in out.split() if "pulse-eks" in c), None) if code == 0 else None


def _cluster_predictor():
    code, out = backend._run(["kubectl", "get", "deploy", "pulse-predictor", "-o",
                              "jsonpath={.spec.template.spec.containers[0].env}", "--request-timeout=6s"], 10)
    if code != 0 or not out:
        return None
    try:
        return next((e.get("value") for e in json.loads(out) if e.get("name") == "PREDICTOR"), None)
    except ValueError:
        return None


def status(cluster) -> dict:
    """Everything the Setup tab shows. The slow checks (AWS, kubectl) take a few seconds."""
    tools = _which("docker", "kubectl", "kind", "helm", "aws", "eksctl", "terraform")
    tools["bash"] = bool(backend.bash())
    images = {t: (f.read_text().strip() if f.exists() else None) for t, f in IMAGE_FILE.items()}
    knative = cluster.mode == "knative" and cluster.reachable
    kind = _kind_exists() if tools["kind"] else False
    return {
        "python": backend.PY, "tools": tools,
        "data": download_data.status(), "models": train_models.status(), "env": env_status(),
        "kind": {"exists": kind, "retail_tools": _retail_tools() if kind else 0},
        "eks": {"context": _eks_context() if tools["kubectl"] else None,
                "account": _aws_account() if tools["aws"] else None},
        "images": images,
        "cluster": {"context": cluster.context, "knative": knative,
                    "predictor": _cluster_predictor() if knative else None},
        "registry": {"pulse_v3": backend.MODEL_URI or None, "action": backend.ACTION_URI or None},
        "predictors": PREDICTORS,
    }


def _need_bash():
    b = backend.bash()
    if not b:
        raise ValueError("bash not found: install Git for Windows (Git Bash), or run the UI on Linux / macOS")
    return b


def build(name: str, p: dict, cluster) -> list[tuple]:
    """The steps (title, command, env) for one Setup action. Raises ValueError with what is missing."""
    if name not in ACTIONS:
        raise ValueError(f"unknown action {name!r}")
    py, title = backend.PY, ACTIONS[name][0]
    region = str(p.get("region") or "ap-south-1")
    if not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", region):
        raise ValueError(f"not an AWS region: {region}")
    env = {"PYTHON": py, "REGION": region, "PYTHONPATH": str(paths.ROOT)}

    if name == "download_data":
        return [(title, [py, "-m", "pulse.setup.download_data"], env)]
    if name == "train_action":
        return [(title, [py, "-m", "pulse.setup.train_models", "action"], env)]
    if name == "train_action_retail":
        return [(title, [py, "-m", "pulse.setup.train_models", "action_retail"], env)]
    if name == "train_pulse_v3":
        if not train_models.embed_server_up():
            raise ValueError("start the embedding server first (Setup -> Embedding server -> Start)")
        return [(title, [py, "-m", "pulse.setup.train_models", "pulse_v3"], env)]
    if name in ("embed_start", "embed_stop"):
        return [(title, [_need_bash(), "scripts/embed_server.sh"] + (["stop"] if name == "embed_stop" else []), env)]
    if name == "kind_create":
        b = _need_bash()
        return [("kind cluster + Knative + Kourier (infra/kind/setup.sh)", [b, "infra/kind/setup.sh"], env),
                ("Redis + the 14 tools at scale-to-zero (infra/knative/deploy.sh)", [b, "infra/knative/deploy.sh", "0"],
                 {**env, "SKIP_BUILD": "1", "KUBE_CONTEXT": "kind-pulse-kn"}),
                ("the PrewarmRequest CRD", ["kubectl", "--context", "kind-pulse-kn", "apply", "-f",
                                            "infra/prewarm/prewarmrequest-crd.yaml"], env)]
    if name == "kind_delete":
        return [(title, [_need_bash(), "infra/kind/teardown.sh"], env)]
    if name == "retail_tools":
        if not _kind_exists():
            raise ValueError("create the kind cluster first (Local cluster -> Create cluster)")
        # The published tools image predates the retail tools: build it here and load it into kind. kind.local/ is a
        # registry Knative never looks up (registries-skipping-tag-resolving), so the nodes' copy is used.
        b, image = _need_bash(), "kind.local/pulse-mcp-tools:retail"
        return [("build the tools image (infra/knative/Dockerfile.mcp)",
                 ["docker", "build", "-f", "infra/knative/Dockerfile.mcp", "-t", image, "."], env),
                ("load it into the kind nodes", ["kind", "load", "docker-image", image, "--name", "pulse-kn"], env),
                ("Redis DB 1 + the 16 retail tools at scale-to-zero (DOMAIN=retail infra/knative/deploy.sh)",
                 [b, "infra/knative/deploy.sh", "0"],
                 {**env, "DOMAIN": "retail", "SKIP_BUILD": "1", "IMAGE": image, "KUBE_CONTEXT": "kind-pulse-kn"})]
    if name == "image_build":
        target = p.get("target") if p.get("target") in ("ecr", "kind") else "ecr"
        return [(f"{title} for {target} (infra/cloud/build_image.sh)", [_need_bash(), "infra/cloud/build_image.sh"],
                 {**env, "TARGET": target})]
    if name == "eks_create":
        return [(title + " (infra/eks/eks_up.sh)", [_need_bash(), "infra/eks/eks_up.sh"],
                 {**env, "USE_TERRAFORM": "1" if p.get("terraform") else "0"})]
    if name == "eks_delete":
        return [(title + " (infra/eks/eks_down.sh)", [_need_bash(), "infra/eks/eks_down.sh"],
                 {**env, "USE_TERRAFORM": "1" if p.get("terraform") else "0"}),
                ("check for leftovers", lambda say: leftovers(region, say), {})]
    if name == "registry_access":
        if not _eks_context():
            raise ValueError("no EKS cluster yet (AWS EKS -> Create cluster); a new eksctl cluster gets this by itself")
        return [(title + " (infra/eks/registry_access.sh)", [_need_bash(), "infra/eks/registry_access.sh"], env)]
    if name == "eks_leftovers":
        return [(f"{title} in {region}", lambda say: leftovers(region, say), {})]
    if name == "cloud_up":
        pred = p.get("predictor") if p.get("predictor") in PREDICTORS else "lightgbm"
        target = "kind" if (cluster.context or "").startswith("kind-") else "ecr"
        return [(f"{title}: predictor {pred} (infra/cloud/cloud_up.sh)", [_need_bash(), "infra/cloud/cloud_up.sh"],
                 {**env, "TARGET": target, "PREDICTOR": pred, "THETA": str(float(p.get("theta") or 0.3))})]
    if name == "set_predictor":
        pred = p.get("predictor")
        if pred not in PREDICTORS:
            raise ValueError(f"predictor must be one of {PREDICTORS}")
        return [(f"pulse-predictor -> {pred}", ["kubectl", "set", "env", "deploy/pulse-predictor", f"PREDICTOR={pred}"], env),
                ("wait for the new predictor", ["kubectl", "rollout", "status", "deploy/pulse-predictor",
                                                "--timeout=300s"], env)]
    if name == "monitoring_install":
        return [(title + " (infra/monitoring/install.sh)", [_need_bash(), "infra/monitoring/install.sh"], env)]
    if name == "gitops_configure":
        acct = str(p.get("account") or _aws_account() or "")
        if not re.fullmatch(r"\d{12}", acct):
            raise ValueError("an AWS account id (12 digits) is needed: log in with aws configure, or type it")
        return [(f"{title}: {acct} (infra/gitops/configure.sh; commit and push it afterwards)",
                 [_need_bash(), "infra/gitops/configure.sh", acct], env)]
    if name == "gitops_install":
        return [(title + " (infra/gitops/install.sh)", [_need_bash(), "infra/gitops/install.sh"],
                 {**env, **({"GITOPS_REVISION": p["revision"]} if p.get("revision") else {})})]
    if name == "tf_bootstrap":                  # its own state stays local (gitignored): applied once per account
        boot = ["terraform", f"-chdir={paths.INFRA / 'terraform' / 'bootstrap'}"]
        if not _aws_account():
            raise ValueError("not logged in to AWS (aws configure)")
        def apply(say):                         # skip if the bucket is there (made once, perhaps on another machine)
            bucket = f"pulse-tfstate-{_aws_account()}"
            if backend._run(["aws", "s3api", "head-bucket", "--bucket", bucket], 30)[0] == 0:
                say(f"  {bucket} already exists: nothing to do")
                return
            code, out = backend._run([*boot, "apply", "-input=false", "-auto-approve", "-no-color",
                                      "-var", f"region={region}"], 600)
            for line in out.splitlines()[-12:]:
                say("  " + line)
            if code != 0:
                raise RuntimeError("terraform apply failed (log above)")
        return [("terraform init (infra/terraform/bootstrap)", boot + ["init", "-input=false", "-no-color"], env),
                (title, apply, {})]
    if name in ("tf_plan", "tf_apply"):
        tf = ["terraform", "-chdir=infra/terraform"]
        repo = str(p.get("github_repo") or "").strip()
        var = ["-var", f"github_repo={repo}"] if repo else []
        steps = [_tf_init(env)]
        if name == "tf_plan":
            return steps + [(title, tf + ["plan", "-input=false", "-no-color", *var], env)]
        return steps + [(title, tf + ["apply", "-input=false", "-auto-approve", "-no-color", *var], env),
                        ("use the model registry it created", registry_from_terraform, {})]
    if name == "registry_from_tf":
        return [_tf_init(env), (title, registry_from_terraform, {})]
    if name in ("registry_push_data", "registry_publish"):
        fam = p.get("family") if p.get("family") in backend.FAMILIES else "pulse_v3"
        uri = backend.FAMILIES[fam]["uri"]()
        if not uri:
            raise ValueError("No model registry set: fill in Model registry above (or Fill in from Terraform)")
        if name == "registry_push_data":
            return [(f"{title} ({fam})", [py, "-m", "pulse.registry", "data", "push", uri, "--family", fam], env)]
        model = paths.MODELS / ("action_model.pkl" if fam == "action" else "pulse_v3/predictor.pkl")
        if not model.exists():
            raise ValueError(f"no {model.relative_to(paths.ROOT)} yet: train it first")
        version = str(p.get("version") or "").strip()
        if not re.fullmatch(r"[\w.-]{1,40}", version):
            raise ValueError("a version name is needed (letters, digits, - _ .)")
        latest = p.get("latest", True) is not False           # unticked: a candidate (Models tab -> Promote later)
        return [(f"{title}: {fam} {version}" + ("" if latest else " (candidate)"),
                 [py, "-m", "pulse.registry", "publish", str(model), uri, "--version", version,
                  "--notes", "published from the UI", *([] if latest else ["--no-latest"])], env)]
    raise ValueError(f"no steps for {name}")


def leftovers(region: str, say) -> None:
    """After a teardown: anything still running (and billing) in the region."""
    checks = [("EKS clusters", ["aws", "eks", "list-clusters", "--region", region, "--query", "clusters", "--output", "text"]),
              ("EC2 instances", ["aws", "ec2", "describe-instances", "--region", region, "--filters",
                                 "Name=instance-state-name,Values=pending,running", "--query",
                                 "Reservations[].Instances[].InstanceId", "--output", "text"]),
              ("load balancers (classic)", ["aws", "elb", "describe-load-balancers", "--region", region, "--query",
                                            "LoadBalancerDescriptions[].LoadBalancerName", "--output", "text"]),
              ("load balancers", ["aws", "elbv2", "describe-load-balancers", "--region", region, "--query",
                                  "LoadBalancers[].LoadBalancerName", "--output", "text"])]
    found = 0
    for what, cmd in checks:
        code, out = backend._run(cmd, 30)
        if code != 0:
            raise RuntimeError(f"{what}: {out[-200:]}")
        names = out.split()
        found += len(names)
        say(f"  {what}: {', '.join(names) if names else 'none'}")
    say("  nothing left running" if not found else f"  {found} resource(s) still there: delete them (AWS console)")
    if found:
        raise RuntimeError("leftovers found")
