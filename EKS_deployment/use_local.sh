# Switch this terminal back to the LOCAL kind cluster (undoes use_eks.sh).
#   source EKS_deployment/use_local.sh
kubectl config use-context kind-pulse-kn >/dev/null && echo "Using local kind cluster (kind-pulse-kn)"
unset KUBE_CONTEXT
export MCP_URL_TEMPLATE="http://127.0.0.1:8081/mcp"
export MCP_HOST_TEMPLATE="{svc}.default.example.com"
echo "           tools at $MCP_URL_TEMPLATE"
