variable "region" {
  description = "AWS region (the EKS cluster's)"
  type        = string
  default     = "ap-south-1"
}

variable "project" {
  description = "Name prefix and Project tag"
  type        = string
  default     = "pulse"
}

variable "model_bucket_name" {
  description = "S3 bucket for trained models; empty = <project>-models-<account id>"
  type        = string
  default     = ""
}

variable "model_prefix" {
  description = "Key prefix of the pulse-v3 models in the bucket"
  type        = string
  default     = "pulse-v3"
}

variable "action_model_prefix" {
  description = "Key prefix of the lightgbm + markov models (action_model.pkl) in the bucket"
  type        = string
  default     = "action-model"
}

variable "noncurrent_version_days" {
  description = "Days an overwritten or deleted model object is kept before it expires"
  type        = number
  default     = 90
}
