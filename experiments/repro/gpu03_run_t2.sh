#!/usr/bin/env bash
# T2: TOFU/Llama-3.2-1B headline reproduction on gpu03 (Blackwell). Run on gpu03.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai-engram
export HF_HOME=/node_data_2/joon/hf_cache CUDA_VISIBLE_DEVICES=0
cd ~/dev/ai-engram
mkdir -p _repro_out
echo "=== T2a: TOFU NLL proxy (ENGRAM_RUN_TOFU) ==="
ENGRAM_RUN_TOFU=1 python -m pytest -s tests/test_tofu_unlearn.py 2>&1 | tee _repro_out/T2a_tofu_nll.log
echo "T2A_EXIT=${PIPESTATUS[0]}"
echo "=== T2b: TOFU official Overall 14-metric (ENGRAM_RUN_TOFU_EVALUATE) ==="
ENGRAM_RUN_TOFU_EVALUATE=1 python -m pytest -s tests/test_tofu_evaluate.py 2>&1 | tee _repro_out/T2b_tofu_overall.log
echo "T2B_EXIT=${PIPESTATUS[0]}"
echo T2_DONE
