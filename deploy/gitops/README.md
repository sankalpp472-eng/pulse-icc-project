# CI/CD and GitOps

How a code change reaches the cluster:

```
pull request ──▶ ci.yml: tests, terraform fmt/validate, kustomize build      terraform.yml: terraform plan (read-only)
merge to main ─▶ image.yml: build the runner image → ECR (OIDC role) → write its tag into EKS_deployment/cloud/kustomization.yaml (commit)
                                                                    │
Argo CD (in the cluster) ◀──── watches Git ─────────────────────────┘  → applies EKS_deployment/cloud, the CRD, monitoring
```

## Argo CD applications (`apps.yaml`)

| Application | Source | Sync |
|---|---|---|
| `pulse-system` | `EKS_deployment/cloud` (kustomize): predictor, speculator, controller, UI, RBAC, `pulse-config`, the trainer's service account | automatic, **no self-heal** (the UI changes things at runtime), no prune |
| `pulse-crds` | `deploy/prewarm`: the PrewarmRequest CRD | automatic, self-heal |
| `monitoring` | kube-prometheus-stack chart + `deploy/monitoring/values*.yaml` | automatic, self-heal, prune |
| `monitoring-config` | `deploy/monitoring` (kustomize): scrape configs and dashboards | automatic, self-heal, prune |

- **Not managed by Argo CD:** the Knative tools (`deploy/knative`). Experiments switch their
  min-scale at runtime (C0 ↔ C1), which Argo CD would keep undoing.
- **How the manifests are reused:** `EKS_deployment/cloud/kustomization.yaml` takes the same
  `pulse-cloud.yaml` and `pulse-ui.yaml` that `cloud_up.sh` and `ui_cloud.sh` apply. Kustomize fills
  their placeholders (image, θ, UI placement, IRSA role) instead of sed. See
  `kubectl kustomize EKS_deployment/cloud`.

## Setup

```bash
bash deploy/gitops/configure.sh          # once per AWS account: fills in ACCOUNT_ID, then commit + push
USE_GITOPS=1 bash EKS_deployment/eks_up.sh    # or, on a running cluster: bash deploy/gitops/install.sh
bash EKS_deployment/cloud/cloud_up.sh         # still needed once per cluster for the Secret pulse-llm-keys (Groq keys)
```

- **Repository and branch:** `install.sh` installs Argo CD (pinned version, `ARGOCD_VERSION`) and
  points it at `git remote get-url origin` and `main`. Use `GITOPS_REVISION=<branch>` to follow
  another branch.
- **Private repository:** Argo CD needs read access to it (a repository Secret; see the Argo CD docs).
- **IRSA roles:** the service-account annotations name the roles from `terraform/eks/irsa.tf`
  (`pulse-eks-pulse`, `pulse-eks-pulse-trainer`). They only work on the Terraform-made cluster. On
  an eksctl cluster, use `eksctl create iamserviceaccount` instead.

## In the UI

- **Top bar:** the code the UI runs (the git commit, or in the cluster the CI-built image tag), and
  **GitOps n/m synced**. Hovering shows each application's sync and health status and the Git
  revision. **Argo CD ↗** opens Argo CD's own UI: the laptop UI keeps a port-forward on :18080, and
  the login is `admin`.
- **Models tab:** **Pin in Git** writes the version into `PULSE_MODEL_VERSION` of
  `EKS_deployment/cloud/gitops/config.yaml` in your checkout. It doesn't commit; commit and open a
  pull request, and after the merge Argo CD rolls it out. The UI pod's predictor serves the pinned
  version.

## CI repository variables

Set these under GitHub → Settings → Secrets and variables → Actions → Variables. The values come
from `terraform/README.md`:

| Variable | Used by |
|---|---|
| `AWS_IMAGE_ROLE_ARN` | `image.yml`: push to ECR |
| `AWS_PLAN_ROLE_ARN` | `terraform.yml`: read-only plan |
| `TF_STATE_BUCKET` | `terraform.yml`: the shared state |

**Branch protection:** `image.yml` pushes a commit to `main` (the new image tag). If `main` is
protected, allow GitHub Actions to push, or change the workflow to open a pull request instead.
