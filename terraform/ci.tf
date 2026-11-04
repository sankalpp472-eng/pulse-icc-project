# GitHub Actions -> AWS without stored keys: GitHub's OIDC tokens are exchanged for short-lived role credentials.
# Two roles, both only for workflows of var.github_repo:
#   pulse-ci-image  push the runner image to ECR      (.github/workflows/image.yml; main branch only)
#   pulse-ci-plan   read-only, for `terraform plan`   (.github/workflows/terraform.yml; any branch / PR)
# Created only when github_repo is set. The repository then needs the variables AWS_IMAGE_ROLE_ARN and
# AWS_PLAN_ROLE_ARN (Settings -> Secrets and variables -> Actions -> Variables) = the outputs below.

variable "github_repo" {
  description = "owner/name of the GitHub repository whose workflows may assume the CI roles (empty = none)"
  type        = string
  default     = ""
}

locals {
  ci = var.github_repo == "" ? {} : {
    pulse-ci-image = "repo:${var.github_repo}:ref:refs/heads/main"
    pulse-ci-plan  = "repo:${var.github_repo}:*"
  }
}

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.github_repo == "" ? 0 : 1
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_policy_document" "ci_trust" {
  for_each = local.ci
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github[0].arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [each.value]
    }
  }
}

resource "aws_iam_role" "ci" {
  for_each           = local.ci
  name               = each.key
  assume_role_policy = data.aws_iam_policy_document.ci_trust[each.key].json
}

data "aws_iam_policy_document" "ci_image" {
  statement {
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    actions = ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
    "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:DescribeRepositories"]
    resources = [aws_ecr_repository.runner.arn]
  }
}

resource "aws_iam_role_policy" "ci_image" {
  count  = var.github_repo == "" ? 0 : 1
  name   = "push-runner-image"
  role   = aws_iam_role.ci["pulse-ci-image"].id
  policy = data.aws_iam_policy_document.ci_image.json
}

resource "aws_iam_role_policy_attachment" "ci_plan" {
  count      = var.github_repo == "" ? 0 : 1
  role       = aws_iam_role.ci["pulse-ci-plan"].name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess" # plan reads everything it manages, changes nothing
}

output "ci_role_arns" {
  value = { for k, r in aws_iam_role.ci : k => r.arn }
}
