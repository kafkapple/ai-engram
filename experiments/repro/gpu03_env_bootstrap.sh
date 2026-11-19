#!/usr/bin/env bash
# gpu03 Blackwell (sm_120) env bootstrap for ai-engram. Run on gpu03.
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
if ! conda env list | grep -qE '^ai-engram\s'; then
  conda create -n ai-engram python=3.11 -y
fi
conda activate ai-engram
# Blackwell needs cu128+ wheels (sm_120). Install torch first, pinned channel.
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu128
cd ~/dev/ai-engram
pip install -e ".[dev]" jupyter nbconvert ipykernel datasets
python - <<'EOF'
import engram, torch
print("OK engram", engram.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), "| dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import transformers; print("transformers", transformers.__version__)
EOF
echo "=== T0: CPU unit tests ==="
mkdir -p ~/dev/ai-engram/_repro_out
python -m pytest tests/test_extraction.py tests/test_compat.py -q 2>&1 | tee ~/dev/ai-engram/_repro_out/T0_pytest.log
echo "BOOTSTRAP_DONE"
