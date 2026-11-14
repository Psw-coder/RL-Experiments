#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_DIR"
if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    python3 -m venv "$VIRTUAL_ENV"
fi
# CUDA 12.4 build matches the host NVIDIA 550 driver.
"$VIRTUAL_ENV/bin/python" -m pip install --index-url https://pypi.org/simple 'https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp313-cp313-linux_x86_64.whl'
"$VIRTUAL_ENV/bin/python" -m pip install --index-url https://pypi.org/simple -r requirements.txt
"$VIRTUAL_ENV/bin/python" -m pip freeze > requirements.lock.txt
