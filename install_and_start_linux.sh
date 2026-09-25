#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-portable.txt
fi
python3 -m webbrowser http://127.0.0.1:4173 >/dev/null 2>&1 || true
exec .venv/bin/python local_server.py
