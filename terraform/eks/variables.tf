variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "cluster_name" {
  type    = string
  default = "pulse-eks"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version; null = EKS's current default"
  type        = string
  default     = null
}

variable "instance_type" {
  description = "m7i-flex.large: 2 vCPU, 8 GiB, Free Tier eligible (see EKS_deployment/cluster.yaml)"
  type        = string
  default     = "m7i-flex.large"
}

variable "workers" {
  description = "Tool / Knative / Pulse nodes. Fixed size: a cold start never includes booting a node"
  type        = number
  default     = 2
}

variable "public_access_cidrs" {
  description = "Who may reach the Kubernetes API (default: anyone with credentials, like eksctl's default)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
