# Point the agent and the runners at the tools on EKS (instead of the local kind cluster).
# SOURCE it (don't run it with bash), from the project root, in every new terminal:
#
#   source EKS_deployment/use_eks.sh
#
# Sets for this terminal:
#   kubectl context      -> the EKS cluster (pod watcher, Redis reset, deploys all use it)
#   KUBE_CONTEXT         -> same, for deploy/knative/deploy.sh
#   MCP_URL_TEMPLATE     -> http://<Kourier load balancer>/mcp   (was http://127.0.0.1:8081/mcp)
#   MCP_HOST_TEMPLATE    -> {svc}.default.example.com           (unchanged)
# Undo with: source EKS_deployment/use_local.sh

_ctx=$(kubectl config get-contexts -o name | grep -m1 "pulse-eks" || true)
if [ -z "$_ctx" ]; then
  echo "No kubectl context for pulse-eks. Run: aws eks update-kubeconfig --region ap-south-1 --name pulse-eks"
  return 1 2>/dev/null || exit 1
fi
kubectl config use-context "$_ctx" >/dev/null
_lb=$(kubectl get svc kourier -n kourier-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
if [ -z "$_lb" ]; then
  echo "Kourier has no load balancer yet. Run: bash EKS_deployment/setup_knative.sh"
  return 1 2>/dev/null || exit 1
fi
export KUBE_CONTEXT="$_ctx"
export TOOL_BACKEND=mcp
export MCP_URL_TEMPLATE="http://$_lb/mcp"
export MCP_HOST_TEMPLATE="{svc}.default.example.com"
echo "Using EKS: context $KUBE_CONTEXT"
echo "           tools at $MCP_URL_TEMPLATE (Host: $MCP_HOST_TEMPLATE)"
unset _ctx _lb
