#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
python -m pip install huggingface_hub[cli]
printf 'Activate: source .venv/bin/activate\nLogin: huggingface-cli login\nTest: python generate.py --theme vintage_winter_cabin --count 1 --device mps --test-one\n'
