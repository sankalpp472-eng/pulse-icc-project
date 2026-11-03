output "model_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "model_uri" {
  description = "PULSE_MODEL_URI for pulse_pred/service.py (loads latest.json)"
  value       = "s3://${aws_s3_bucket.models.bucket}/${var.model_prefix}"
}

output "model_reader_policy_arn" {
  value = aws_iam_policy.model_reader.arn
}

output "model_publisher_policy_arn" {
  value = aws_iam_policy.model_publisher.arn
}
