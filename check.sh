#!/bin/bash
set -e

source .venv/bin/activate
pyright
pytest
black .
python check.py