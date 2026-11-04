# Terraform

Infrastructure as code for Pulse, in three roots with separate state. Each has its own lifecycle:

| Root | State | What | Lifecycle |
|---|---|---|---|
| `bootstrap/` | local (tiny) | `pulse-tfstate-<account>`: the S3 bucket holding the other two states (versioned, encrypted, private) | once per AWS account |
| `.` (this folder) | S3, `pulse/registry.tfstate` | the model registry, ECR and the CI roles | long-lived |
| `eks/` | S3, `pulse/eks.tfstate` | the EKS cluster (an alternative to `eksctl`) | created and destroyed per working session |

Because they're separate, destroying the cluster never touches the models, the images or CI.

## The registry root (this folder)

| File | What |
|---|---|
| `model_store.tf` | The `pulse-models-<account>` S3 bucket: private, versioned, encrypted, TLS only, with old object versions expiring after 90 days. Also two IAM policies: `pulse-model-reader` (predictors) and `pulse-model-publisher` (training, CI). |
| `ecr.tf` | The `pulse-runner` image repository: immutable tags, scanned on push, newest 20 images kept. If `build_image.sh` already created it, adopt it once with `terraform import aws_ecr_repository.runner pulse-runner`. |
| `ci.tf` | GitHub Actions → AWS through OIDC, so no keys are stored in GitHub. `pulse-ci-image` may push images (main branch only), and `pulse-ci-plan` is read-only for `terraform plan` on PRs. Created only with `-var github_repo=owner/name`. |
| `versions.tf` | The AWS provider and the S3 backend. |
| `variables.tf`, `outputs.tf` | Settings and outputs: `model_uri`, policy ARNs, `runner_repository_url`, `ci_role_arns`. |

## The EKS root (`eks/`)

This root builds the same cluster as `EKS_deployment/cluster.yaml`:
- 2 worker nodes and 1 client node (tainted `pulse/role=client`), all m7i-flex.large;
- a VPC with two public subnets and no NAT gateway (cheaper than eksctl's default);
- the cluster's OIDC provider and IRSA roles (`irsa.tf`): `default/pulse` reads models, and
  `default/pulse-trainer` (the retraining Job) publishes them.

It needs the registry root applied first, because the IRSA roles attach its policies.

```bash
USE_TERRAFORM=1 bash EKS_deployment/eks_up.sh      # terraform apply here + the usual Knative / tools setup
USE_TERRAFORM=1 bash EKS_deployment/eks_down.sh    # removes Kourier's load balancer, then terraform destroy
```

Without `USE_TERRAFORM=1` the scripts use eksctl exactly as before.

## Setup, in order

```bash
terraform -chdir=terraform/bootstrap init && terraform -chdir=terraform/bootstrap apply      # 1. state bucket
B=pulse-tfstate-$(aws sts get-caller-identity --query Account --output text)
terraform -chdir=terraform init -migrate-state -backend-config="bucket=$B"                  # 2. move local state to S3
terraform -chdir=terraform plan  -var github_repo=sankalpp472-eng/pulse-icc-project          # 3. review, then apply
terraform -chdir=terraform apply -var github_repo=sankalpp472-eng/pulse-icc-project
```

After step 3, set the repository variables `AWS_IMAGE_ROLE_ARN` and `AWS_PLAN_ROLE_ARN` (from
`terraform output ci_role_arns`) and `TF_STATE_BUCKET`. They're under GitHub → Settings → Secrets
and variables → Actions → Variables. The workflows in `.github/workflows/` use them.

Publishing a model and serving it:

```bash
python pulse_pred/model_store.py publish predictor.pkl "$(terraform -chdir=terraform output -raw model_uri)" --version 2026-11-03 --notes "..."
PULSE_MODEL_URI="$(terraform -chdir=terraform output -raw model_uri)" uvicorn service:app --app-dir pulse_pred --port 8000
```

CI never applies anything: `terraform plan` runs on pull requests that change `terraform/`, and a
person runs `apply`.
