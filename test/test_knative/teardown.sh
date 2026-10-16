#!/usr/bin/env bash
# Delete the local Knative kind cluster.
export PATH="$HOME/.arkade/bin:$PATH"
kind delete cluster --name pulse-kn
