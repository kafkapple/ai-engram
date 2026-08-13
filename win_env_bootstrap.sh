#!/usr/bin/env bash
# Windows personal server (WSL2 Ubuntu, RTX 3060 12GB / sm_86) env bootstrap for ai-engram.
# Successor to gpu03_env_bootstrap.sh — gpu03 access ends 2026-08-17 (offboarding).
# Version pins mirror the gpu03 env that produced _repro_out/ (environment.gpu03.yml).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/ai-engram}"
REPRO_ROOT="${REPRO_ROOT:-$HOME/dev/ai-engram-repro}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME/checkpoints}"
# WSL2 ships the GPU userspace under /usr/lib/wsl/lib and does NOT put it on PATH.
export PATH="/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# `pip` must be named explicitly: a conda-forge python=3.11 env ships WITHOUT pip, and
# a bare `pip install` then silently falls through to /usr/bin/pip (observed 2026-08-13).
conda env list | grep -qE '^ai-engram\s' || conda create -n ai-engram python=3.11 pip -y
conda activate ai-engram

# Never rely on PATH here. Under `set -euo pipefail` in a non-interactive shell,
# `conda activate` returned 0 but left `python` pointing at /usr/bin/python3, and the
# whole torch stack silently landed in ~/.local/lib/python3.10 (observed 2026-08-13).
# Bind the interpreter explicitly and refuse to continue if it is the wrong one.
ENV_PREFIX="$HOME/miniconda3/envs/ai-engram"
PY="$ENV_PREFIX/bin/python"
[ -x "$PY" ] || { echo "FATAL: $PY missing — conda env creation failed"; exit 1; }
export PYTHONNOUSERSITE=1   # keep any stray ~/.local packages out of the env
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$PY" -m pip --version >/dev/null 2>&1 || conda install -n ai-engram -y pip

# sm_86 (Ampere) runs on any cu12x wheel — unlike gpu03's Blackwell sm_120 (cu128+ only).
"$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install "torch==2.12.1"

[ -d "$REPO_ROOT/.git" ] || git clone https://github.com/kafkapple/ai-engram "$REPO_ROOT"
cd "$REPO_ROOT"
# rouge_score/scikit-learn are importorskip'd by tests/test_tofu_evaluate.py but absent
# from pyproject deps — without them T2b silently skips. Same pin as setup_and_repro.sh.
"$PY" -m pip install -e ".[dev]" jupyter nbconvert ipykernel datasets rouge_score scikit-learn

mkdir -p "$CHECKPOINT_ROOT/ai_engram" "$REPO_ROOT/_repro_out"

"$PY" - <<'EOF'
import torch, transformers, engram
print("engram", engram.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "| dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"vram free/total: {free/2**30:.1f}/{total/2**30:.1f} GiB")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
EOF

echo "=== T0: CPU unit tests ==="
"$PY" -m pytest tests/test_extraction.py tests/test_compat.py -q 2>&1 | tee "$REPO_ROOT/_repro_out/T0_pytest_win.log"
echo "BOOTSTRAP_DONE"
