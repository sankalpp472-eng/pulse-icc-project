#!/usr/bin/env bash
# One-time setup after cloning: a Python virtual environment (.venv) with every requirement and the pulse package,
# the public tau-bench data, and .env from the example. Run from anywhere, in Git Bash on Windows or a Linux / macOS
# shell.
#
#   bash scripts/setup.sh                 # PYTHON=python3.11 bash scripts/setup.sh to pick the interpreter
#
# Afterwards, in every new terminal:  export PY=<the path printed at the end>, then "$PY" -m pulse.ui.server
set -euo pipefail
cd "$(dirname "$0")/.."
# The first Python 3.11+ that actually runs (on Windows `python3` is often the Microsoft Store placeholder)
if [ -z "${PYTHON:-}" ]; then
  for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1; then
      PYTHON=$c; break
    fi
  done
fi
[ -n "${PYTHON:-}" ] && "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1 \
  || { echo "Python 3.11+ needed (PYTHON=... bash scripts/setup.sh)"; exit 1; }

[ -d .venv ] || "$PYTHON" -m venv .venv
if [ -x .venv/Scripts/python.exe ]; then PY="$PWD/.venv/Scripts/python"; else PY="$PWD/.venv/bin/python"; fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt -e .
echo "packages installed"

"$PY" -m pulse.setup.download_data

if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example: add your LLM keys there for live runs and the speculator"
fi

echo
echo "Setup done. In every new terminal:"
echo "  export PY=\"$PY\""
echo "Then start the UI:  \"\$PY\" -m pulse.ui.server   -> http://127.0.0.1:8765   (README.md)"
