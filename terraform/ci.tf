# GitHub Actions -> AWS without stored keys: GitHub's OIDC tokens are exchanged for short-lived role credentials.
# Two roles, both only for workflows of var.github_repo:
#   pulse-ci-image  push the runner image to ECR      (.github/workflows/image.yml; main branch only)
#   pulse-ci-plan   read-only, for `terraform plan`   (.github/workflows/terraform.yml; any branch / PR)
# Created for github_repo (empty = none). The repository then needs the variables AWS_IMAGE_ROLE_ARN and
# AWS_PLAN_ROLE_ARN (Settings -> Secrets and variables -> Actions -> Variables) = the outputs below.
# The default is the project's repository, so a plain `terraform plan` (and CI's) keeps the applied roles.

variable "github_repo" {
  description = "owner/name of the GitHub repository whose workflows may assume the CI roles (empty = none)"
  type        = string
  default     = "sankalpp472-eng/pulse-icc-project"
}

variable "github_repo_ids" {
  description = <<-EOT
    owner_id/repo_id of github_repo. GitHub's OIDC subject names the repository with its immutable ids
    (repo:<owner>@<owner_id>/<name>@<repo_id>:...), which a renamed or re-created repository can't reuse.
    Find them in a failed attempt in CloudTrail, or: curl -s https://api.github.com/repos/<owner>/<name> (owner.id, id)
  EOT
  type        = string
  default     = "328253652/1404634644"
}

locals {
  gh     = split("/", var.github_repo)
  gh_ids = split("/", var.github_repo_ids)
  subjects = var.github_repo == "" ? [] : compact([
    "repo:${var.github_repo}", # older subject form
    var.github_repo_ids == "" ? "" : "repo:${local.gh[0]}@${local.gh_ids[0]}/${local.gh[1]}@${local.gh_ids[1]}",
  ])
  ci = var.github_repo == "" ? {} : {
    pulse-ci-image = [for s in local.subjects : "${s}:ref:refs/heads/main"]
    pulse-ci-plan  = [for s in local.subjects : "${s}:*"]
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
      values   = each.value
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
