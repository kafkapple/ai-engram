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
conda env list | grep -qE '^ai-engram\s' || conda create -n ai-engram python=3.11 -y
conda activate ai-engram

# sm_86 (Ampere) runs on any cu12x wheel — unlike gpu03's Blackwell sm_120 (cu128+ only).
python -c "import torch" 2>/dev/null || pip install "torch==2.12.1"

[ -d "$REPO_ROOT/.git" ] || git clone https://github.com/kafkapple/ai-engram "$REPO_ROOT"
cd "$REPO_ROOT"
# rouge_score/scikit-learn are importorskip'd by tests/test_tofu_evaluate.py but absent
# from pyproject deps — without them T2b silently skips. Same pin as setup_and_repro.sh.
pip install -e ".[dev]" jupyter nbconvert ipykernel datasets rouge_score scikit-learn

mkdir -p "$CHECKPOINT_ROOT/ai_engram" "$REPO_ROOT/_repro_out"

python - <<'EOF'
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
python -m pytest tests/test_extraction.py tests/test_compat.py -q 2>&1 | tee "$REPO_ROOT/_repro_out/T0_pytest_win.log"
echo "BOOTSTRAP_DONE"
