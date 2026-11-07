#!/usr/bin/env bash
# Put the AWS account into the GitOps manifests (once per account; then commit and push):
# ECR image name, the IRSA role ARNs of the pulse and pulse-trainer service accounts, the model registry.
#
#   bash infra/gitops/configure.sh                 # the account aws is logged in to
#   bash infra/gitops/configure.sh 123456789012
set -euo pipefail
cd "$(dirname "$0")/../.."
ACCOUNT=${1:-$(aws sts get-caller-identity --query Account --output text)}
[[ "$ACCOUNT" =~ ^[0-9]{12}$ ]] || { echo "not an AWS account id: $ACCOUNT"; exit 1; }
FILES="infra/cloud/kustomization.yaml infra/cloud/gitops/settings.yaml
       infra/cloud/gitops/config.yaml infra/cloud/gitops/trainer-sa.yaml"
# replaces the ACCOUNT_ID placeholder or whatever account is there now (a fork starts with the team's account)
sed -i -E "s/ACCOUNT_ID|\b[0-9]{12}\b/$ACCOUNT/g" $FILES
grep -n "$ACCOUNT" $FILES
echo "Now: git commit -am 'gitops: AWS account $ACCOUNT' && git push"
