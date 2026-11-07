#!/usr/bin/env bash
# Prometheus + Grafana (kube-prometheus-stack) for Pulse, in the namespace `monitoring` of the current kubectl
# context (kind or EKS). Safe to rerun: it upgrades in place and refreshes the scrape configs and dashboards.
#
#   bash infra/monitoring/install.sh              # install / update
#   bash infra/monitoring/install.sh uninstall    # remove it all (Prometheus data goes with it)
#
# Needs helm and kubectl. On EKS run it after eks_up.sh (it puts the stack on the client node); the UI and
# `kubectl port-forward -n monitoring svc/pulse-mon-grafana 3000:80` then show Grafana. CHART_VERSION pins the
# chart version (default: the newest).
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=monitoring
REL=pulse-mon
DIR=infra/monitoring

command -v helm >/dev/null || { echo "helm not found: https://helm.sh/docs/intro/install/"; exit 1; }
CTX=$(kubectl config current-context)

if [ "${1:-}" = "uninstall" ]; then
  kubectl delete -k "$DIR" --ignore-not-found
  helm uninstall "$REL" -n "$NS" || true
  kubectl delete namespace "$NS" --ignore-not-found
  echo "monitoring removed from $CTX (the chart's CRDs stay: kubectl get crd | grep monitoring.coreos.com)"
  exit 0
fi

FILES=(-f "$DIR/values.yaml")
if [[ "$CTX" == kind-* ]]; then
  # The laptop's processes are scraped at host.docker.internal (Docker Desktop). Native Docker (Linux, WSL)
  # has no such name: use the kind network's gateway, which is the host.
  KIND_VALUES=$DIR/values-kind.yaml
  if ! docker exec "${CTX#kind-}-control-plane" getent hosts host.docker.internal >/dev/null 2>&1; then
    GW=$(docker network inspect kind -f '{{range .IPAM.Config}}{{.Gateway}} {{end}}' | tr ' ' '\n' | grep -m1 '\.')
    KIND_VALUES=$(mktemp)
    sed "s/host\.docker\.internal/$GW/g" "$DIR/values-kind.yaml" > "$KIND_VALUES"
    echo "no host.docker.internal: scraping the laptop at $GW"
  fi
  FILES+=(-f "$KIND_VALUES")
elif [ -n "$(kubectl get nodes -l pulse/role=client -o name)" ]; then
  FILES+=(-f "$DIR/values-eks.yaml")
fi
echo "context $CTX | values: ${FILES[*]}"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update >/dev/null
helm upgrade --install "$REL" prometheus-community/kube-prometheus-stack -n "$NS" --create-namespace \
  "${FILES[@]}" ${CHART_VERSION:+--version "$CHART_VERSION"} --wait --timeout 10m

kubectl apply -k "$DIR"          # scrape configs + dashboards as ConfigMaps (kustomization.yaml; Argo CD applies the same)

# Knative (1.18+) exports no metrics unless asked: turn on its Prometheus endpoint (:9090, the knative-serving
# PodMonitor). The activator and autoscaler read it only at start, so restart them the first time.
if kubectl get ns knative-serving >/dev/null 2>&1; then
  if [ "$(kubectl get cm config-observability -n knative-serving -o jsonpath='{.data.metrics-protocol}')" != prometheus ]; then
    kubectl patch cm config-observability -n knative-serving --type merge -p '{"data":{"metrics-protocol":"prometheus"}}'
    kubectl rollout restart deploy/activator deploy/autoscaler -n knative-serving
    kubectl rollout status deploy/activator -n knative-serving --timeout=180s
    kubectl rollout status deploy/autoscaler -n knative-serving --timeout=180s
  fi
fi

helm list -n "$NS" --filter "^$REL\$"
echo
echo "Grafana: in the UI (Knative mode -> Monitoring tab), or"
echo "  kubectl port-forward -n $NS svc/pulse-mon-grafana 3000:80   ->  http://127.0.0.1:3000 (read-only without login)"
echo "  admin password: kubectl get secret -n $NS pulse-mon-grafana -o jsonpath='{.data.admin-password}' | base64 -d"
