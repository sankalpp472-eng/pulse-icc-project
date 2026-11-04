terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }

  # Shared state in S3 (bucket from terraform/bootstrap), so the team and CI plan against the same state.
  # First time, with an existing local state:  terraform init -migrate-state -backend-config="bucket=pulse-tfstate-<account>"
  backend "s3" {
    key          = "pulse/registry.tfstate"
    region       = "ap-south-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
