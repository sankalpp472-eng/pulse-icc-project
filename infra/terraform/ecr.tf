# The runner image repository (infra/cloud/build_image.sh, CI's image workflow). It outlives every
# cluster, so it lives in this long-lived root next to the model registry. Images are scanned on push; only the
# newest 20 are kept.
# If the repository already exists (build_image.sh creates it on first use), adopt it once instead of creating it:
#   terraform -chdir=infra/terraform import aws_ecr_repository.runner pulse-runner

resource "aws_ecr_repository" "runner" {
  name                 = "pulse-runner"
  image_tag_mutability = "IMMUTABLE" # a tag (date + commit) always means the same image
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "runner" {
  repository = aws_ecr_repository.runner.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the newest 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

output "runner_repository_url" {
  value = aws_ecr_repository.runner.repository_url
}
