# Model registry: a private, versioned, encrypted bucket (layout in pulse_pred/model_store.py),
# a read-only policy for the predictor and a write policy for whoever publishes models.

locals {
  model_bucket = var.model_bucket_name != "" ? var.model_bucket_name : "${var.project}-models-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "models" {
  bucket = local.model_bucket
}

resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    id     = "expire-old-object-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_days
    }
  }
  depends_on = [aws_s3_bucket_versioning.models]
}

data "aws_iam_policy_document" "models_tls_only" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "models" {
  bucket     = aws_s3_bucket.models.id
  policy     = data.aws_iam_policy_document.models_tls_only.json
  depends_on = [aws_s3_bucket_public_access_block.models]
}

# The model families in the bucket: pulse-v3, and lightgbm + markov (one action_model.pkl)
locals {
  model_prefixes = [var.model_prefix, var.action_model_prefix]
}

# Predictor: read the models under the prefixes (attach to its pod's IAM role once EKS is in Terraform)
data "aws_iam_policy_document" "model_reader" {
  statement {
    actions   = ["s3:GetObject"]
    resources = [for p in local.model_prefixes : "${aws_s3_bucket.models.arn}/${p}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.models.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = [for p in local.model_prefixes : "${p}/*"]
    }
  }
}

resource "aws_iam_policy" "model_reader" {
  name        = "${var.project}-model-reader"
  description = "Read ${var.model_prefix} models from ${local.model_bucket}"
  policy      = data.aws_iam_policy_document.model_reader.json
}

# Training / CI: publish new versions (model_store.py publish)
data "aws_iam_policy_document" "model_publisher" {
  source_policy_documents = [data.aws_iam_policy_document.model_reader.json]
  statement {
    actions   = ["s3:PutObject"]
    resources = [for p in local.model_prefixes : "${aws_s3_bucket.models.arn}/${p}/*"]
  }
}

resource "aws_iam_policy" "model_publisher" {
  name        = "${var.project}-model-publisher"
  description = "Publish ${var.model_prefix} models to ${local.model_bucket}"
  policy      = data.aws_iam_policy_document.model_publisher.json
}
