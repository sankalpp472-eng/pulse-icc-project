# IAM roles for service accounts (IRSA): pods get short-lived AWS credentials for exactly what they need, with no
# keys in the cluster. The model-registry policies come from the registry root (terraform/), applied before this.
#   default/pulse          Pulse's pods (predictors, UI): read models          pulse-model-reader
#   default/pulse-trainer  the retraining Job (deploy/training/): publish them  pulse-model-publisher
# deploy/gitops/ annotates the two service accounts with these role ARNs.

resource "aws_iam_openid_connect_provider" "eks" {
  url            = aws_eks_cluster.main.identity[0].oidc[0].issuer
  client_id_list = ["sts.amazonaws.com"]
}

locals {
  oidc_host = replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")
  irsa = {
    pulse         = { service_account = "pulse", policy = "pulse-model-reader" }
    pulse-trainer = { service_account = "pulse-trainer", policy = "pulse-model-publisher" }
  }
}

data "aws_iam_policy" "registry" {
  for_each = local.irsa
  name     = each.value.policy
}

data "aws_iam_policy_document" "irsa_trust" {
  for_each = local.irsa
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.eks.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = ["system:serviceaccount:default:${each.value.service_account}"]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "irsa" {
  for_each           = local.irsa
  name               = "${var.cluster_name}-${each.key}"
  assume_role_policy = data.aws_iam_policy_document.irsa_trust[each.key].json
}

resource "aws_iam_role_policy_attachment" "irsa" {
  for_each   = local.irsa
  role       = aws_iam_role.irsa[each.key].name
  policy_arn = data.aws_iam_policy.registry[each.key].arn
}
