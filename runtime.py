"""Configure project-local storage before importing third-party packages."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for key, value in {
    'XDG_CACHE_HOME': '.cache', 'PIP_CACHE_DIR': '.cache/pip',
    'TORCH_HOME': '.cache/torch', 'TORCHINDUCTOR_CACHE_DIR': '.cache/torchinductor',
    'TRITON_CACHE_DIR': '.cache/triton', 'CUDA_CACHE_PATH': '.cache/cuda',
    'MPLCONFIGDIR': '.cache/matplotlib', 'HF_HOME': '.cache/huggingface',
    'TMPDIR': 'tmp', 'TEMP': 'tmp', 'TMP': 'tmp',
}.items():
    path = ROOT / value
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Output and input paths must remain inside the project')
    return path
