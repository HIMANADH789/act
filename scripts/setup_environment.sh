#!/bin/sh
set -eu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-models.txt
python -m pip install -e .
