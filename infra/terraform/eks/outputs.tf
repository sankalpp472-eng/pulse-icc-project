output "cluster_name" {
  value = aws_eks_cluster.main.name
}

output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.region} --name ${aws_eks_cluster.main.name}"
}

output "irsa_role_arns" {
  description = "service account -> IAM role (annotation eks.amazonaws.com/role-arn; infra/gitops/ sets them)"
  value       = { for k, r in aws_iam_role.irsa : local.irsa[k].service_account => r.arn }
}
