#!/usr/bin/env bash
# One-time setup after cloning: a Python virtual environment with every requirement, the public tau-bench data,
# and agent/.env from the example. Run from anywhere, in Git Bash on Windows or a Linux / macOS shell.
#
#   bash scripts/setup.sh                 # PYTHON=python3.11 bash scripts/setup.sh to pick the interpreter
#
# Afterwards, in every new terminal:  export PY=<the path printed at the end>
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ needed"' \
  || { echo "Python 3.11+ needed (PYTHON=... bash scripts/setup.sh)"; exit 1; }

[ -d agent/.venv ] || "$PYTHON" -m venv agent/.venv
if [ -x agent/.venv/Scripts/python.exe ]; then PY="$PWD/agent/.venv/Scripts/python"; else PY="$PWD/agent/.venv/bin/python"; fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r agent/requirements.txt -r predictor/requirements.txt -r pulse_pred/requirements.txt \
  -r loadtest/requirements.txt
echo "packages installed"

"$PY" scripts/download_data.py

if [ ! -f agent/.env ]; then
  cp agent/.env.example agent/.env
  echo "created agent/.env from agent/.env.example: add your LLM keys there for live runs and the speculator"
fi

echo
echo "Setup done. In every new terminal:"
echo "  export PY=\"$PY\""
echo "Then: \"\$PY\" ui/server.py   (docs/RUNNING.md). Optional: \"\$PY\" scripts/train_models.py all"
