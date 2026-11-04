# IRSA role for the training Job (deploy/training/train-job.yaml): the service account default/pulse-trainer may
# publish models (pulse-model-publisher). Created only when the cluster's OIDC provider is given, because the EKS
# cluster is still made by eksctl (its provider ARN changes with every new cluster):
#   aws iam list-open-id-connect-providers     ->  terraform apply -var eks_oidc_provider_arn=<arn>
# Until EKS is in Terraform, `eksctl create iamserviceaccount` (run_training.sh prints it) does the same.

variable "eks_oidc_provider_arn" {
  description = "IAM OIDC provider of the EKS cluster (empty = no training role)"
  type        = string
  default     = ""
}

locals {
  oidc_host = replace(var.eks_oidc_provider_arn, "/^.*oidc-provider\\//", "")
}

data "aws_iam_policy_document" "trainer_trust" {
  count = var.eks_oidc_provider_arn == "" ? 0 : 1
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = ["system:serviceaccount:default:pulse-trainer"]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "trainer" {
  count              = var.eks_oidc_provider_arn == "" ? 0 : 1
  name               = "${var.project}-trainer"
  assume_role_policy = data.aws_iam_policy_document.trainer_trust[0].json
}

resource "aws_iam_role_policy_attachment" "trainer_publish" {
  count      = var.eks_oidc_provider_arn == "" ? 0 : 1
  role       = aws_iam_role.trainer[0].name
  policy_arn = aws_iam_policy.model_publisher.arn
}

output "trainer_role_arn" {
  description = "annotate the service account: eks.amazonaws.com/role-arn=<this>"
  value       = var.eks_oidc_provider_arn == "" ? null : aws_iam_role.trainer[0].arn
}
