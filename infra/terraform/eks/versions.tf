# The EKS cluster as code: the same cluster as infra/eks/cluster.yaml (eksctl), created and destroyed per
# working session. Its own root and state, so `terraform destroy` here never touches the model registry
# (infra/terraform/).
#
#   terraform -chdir=infra/terraform/eks init -backend-config="bucket=pulse-tfstate-<account>"
#   USE_TERRAFORM=1 bash infra/eks/eks_up.sh        # apply + the usual Knative / tools setup
#   USE_TERRAFORM=1 bash infra/eks/eks_down.sh      # destroy

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
  backend "s3" { # bucket from infra/terraform/bootstrap (-backend-config="bucket=...")
    key          = "pulse/eks.tfstate"
    region       = "ap-south-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "pulse", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}
