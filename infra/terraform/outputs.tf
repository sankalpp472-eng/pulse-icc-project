output "model_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "model_uri" {
  description = "PULSE_MODEL_URI for the pulse_v3 predictor (loads latest.json)"
  value       = "s3://${aws_s3_bucket.models.bucket}/${var.model_prefix}"
}

output "action_model_uri" {
  description = "ACTION_MODEL_URI for the lightgbm and markov predictors (loads latest.json)"
  value       = "s3://${aws_s3_bucket.models.bucket}/${var.action_model_prefix}"
}

output "model_reader_policy_arn" {
  value = aws_iam_policy.model_reader.arn
}

output "model_publisher_policy_arn" {
  value = aws_iam_policy.model_publisher.arn
}
