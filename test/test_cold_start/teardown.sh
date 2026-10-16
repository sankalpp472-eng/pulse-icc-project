#!/usr/bin/env bash
# Delete the local kind cluster (OpenFaaS and functions go with it).
export PATH="$HOME/.arkade/bin:$PATH"
kind delete cluster --name pulse
