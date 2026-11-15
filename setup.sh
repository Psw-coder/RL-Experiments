#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_DIR"
if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    python3 -m venv "$VIRTUAL_ENV"
fi
# CUDA 12.8 wheels support Blackwell (sm_120) and the host driver.
# requirements.txt adds the PyTorch cu128 index, so pip selects the
# wheel matching this venv's Python version (3.13 or 3.14).
# Set PIP_INDEX_URL to a faster mirror when pypi.org is slow.
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
"$VIRTUAL_ENV/bin/python" -m pip install --index-url "$PIP_INDEX_URL" \
    --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
"$VIRTUAL_ENV/bin/python" -m pip freeze > requirements.lock.txt
