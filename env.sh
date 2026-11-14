#!/usr/bin/env bash
# Source this before installing dependencies or running Python.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR
export VIRTUAL_ENV="$PROJECT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.python"
export UV_TOOL_DIR="$PROJECT_DIR/.tools"
export UV_TOOL_BIN_DIR="$PROJECT_DIR/.tools/bin"
export TMPDIR="$PROJECT_DIR/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export CUDA_CACHE_PATH="$PROJECT_DIR/.cache/cuda"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export SDL_VIDEODRIVER=dummy
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# Prevent inherited environment paths from importing unrelated projects.
unset PYTHONPATH PYTHONHOME
