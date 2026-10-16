#!/bin/bash

FUNC_NAME="cold-tool"
GATEWAY="http://127.0.0.1:8080"
NAMESPACE="openfaas-fn"

echo "1. Scaling deployment down to 0 replicas..."
kubectl scale deployment -n $NAMESPACE $FUNC_NAME --replicas=0

echo "2. Waiting for pod termination..."
kubectl wait --for=delete pod -l faas_function=$FUNC_NAME -n $NAMESPACE --timeout=60s

echo "3. Measuring COLD START latency..."
curl -w "Cold Start Latency: %{time_total}s\n" -o /dev/null -s "$GATEWAY/function/$FUNC_NAME"

echo "4. Measuring WARM START latency..."
curl -w "Warm Start Latency: %{time_total}s\n" -o /dev/null -s "$GATEWAY/function/$FUNC_NAME"