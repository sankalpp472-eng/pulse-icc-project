# Terraform

Infrastructure as code for Pulse. It starts with the model registry. The EKS cluster is still created
with `eksctl` (`EKS_deployment/`).

| File | What |
|---|---|
| `model_store.tf` | The `pulse-models-<account id>` S3 bucket: private, versioned, encrypted, TLS only, with old object versions expiring after 90 days. Also two IAM policies: `pulse-model-reader` (the predictor) and `pulse-model-publisher` (training / CI). |
| `versions.tf` | The AWS provider (region `ap-south-1`, default tags) and a commented S3 backend for shared state. |
| `variables.tf`, `outputs.tf` | Settings, plus the bucket name, `model_uri` and the policy ARNs. |

## Use

```bash
cd terraform
terraform init
terraform plan                 # review: 1 bucket (+ its settings) and 2 IAM policies
terraform apply
```

Then publish the model and point the predictor at it (the publishing identity needs
`pulse-model-publisher`, or admin rights):

```bash
python pulse_pred/model_store.py publish predictor.pkl "$(terraform -chdir=terraform output -raw model_uri)" --version 2026-11-03 --notes "tau-bench airline+retail"
PULSE_MODEL_URI="$(terraform -chdir=terraform output -raw model_uri)" uvicorn service:app --app-dir pulse_pred --port 8000
```

## Next pieces

- **Remote state:** an S3 backend (see `versions.tf`) so the team and CI share one state.
- **EKS:** the cluster, node groups and ECR repositories in Terraform instead of `eksctl`. Then an
  IRSA / Pod Identity role per component; the predictor's role gets `pulse-model-reader`.
- **CI and GitOps:** GitHub Actions authenticating to AWS through OIDC (`terraform plan` on PRs,
  image builds to ECR), and Argo CD syncing the Kubernetes manifests from the repository.
