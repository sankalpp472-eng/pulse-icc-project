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
# The first Python 3.11+ that runs and can make a virtual environment (on Windows `python3` is often the Microsoft
# Store placeholder; Debian / Ubuntu ship venv's ensurepip separately, in python3-venv)
USABLE='import sys, ensurepip; assert sys.version_info >= (3, 11)'
if [ -z "${PYTHON:-}" ]; then
  for c in python3 python py python3.13 python3.12 python3.11; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "$USABLE" >/dev/null 2>&1; then PYTHON=$c; break; fi
  done
fi
if [ -z "${PYTHON:-}" ] || ! "$PYTHON" -c "$USABLE" >/dev/null 2>&1; then
  echo "No Python 3.11+ that can create a virtual environment was found."
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
    echo "  python3 is $(python3 --version 2>&1) but lacks venv: on Debian / Ubuntu  sudo apt install python3-venv"
  fi
  echo "  or point to another Python:  PYTHON=/path/to/python3.11 bash scripts/setup.sh"
  exit 1
fi
echo "using $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# (re)create .venv unless a working one is there (a failed attempt can leave a folder without pip)
VPY=.venv/bin/python; [ -x .venv/Scripts/python.exe ] && VPY=.venv/Scripts/python.exe
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  "$PYTHON" -m venv .venv
fi
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
