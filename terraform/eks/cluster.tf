# The cluster and its two managed node groups, as in EKS_deployment/cluster.yaml:
#   workers  2 x m7i-flex.large, 50 GB   label pulse/role=worker   (tools, Knative, Pulse services)
#   client   1 x m7i-flex.large, 30 GB   label + taint pulse/role=client  (run pods, UI, monitoring, training)

data "aws_iam_policy_document" "eks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cluster" {
  name               = "${var.cluster_name}-cluster"
  assume_role_policy = data.aws_iam_policy_document.eks_assume.json
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  version  = var.kubernetes_version
  role_arn = aws_iam_role.cluster.arn
  vpc_config {
    subnet_ids              = aws_subnet.public[*].id
    endpoint_public_access  = true
    endpoint_private_access = true
    public_access_cidrs     = var.public_access_cidrs
  }
  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true # whoever applies this can use kubectl
  }
  depends_on = [aws_iam_role_policy_attachment.cluster]
}

data "aws_iam_policy_document" "node_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "${var.cluster_name}-node"
  assume_role_policy = data.aws_iam_policy_document.node_assume.json
}

resource "aws_iam_role_policy_attachment" "node" {
  for_each   = toset(["AmazonEKSWorkerNodePolicy", "AmazonEKS_CNI_Policy", "AmazonEC2ContainerRegistryReadOnly"])
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/${each.value}"
}

resource "aws_eks_node_group" "workers" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "workers"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.public[*].id
  instance_types  = [var.instance_type]
  disk_size       = 50
  labels          = { "pulse/role" = "worker" }
  scaling_config {
    desired_size = var.workers
    min_size     = var.workers
    max_size     = var.workers
  }
  depends_on = [aws_iam_role_policy_attachment.node]
}

resource "aws_eks_node_group" "client" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "client"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.public[*].id
  instance_types  = [var.instance_type]
  disk_size       = 30
  labels          = { "pulse/role" = "client" }
  taint {
    key    = "pulse/role"
    value  = "client"
    effect = "NO_SCHEDULE"
  }
  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 1
  }
  depends_on = [aws_iam_role_policy_attachment.node]
}
