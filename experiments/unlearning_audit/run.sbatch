#!/bin/bash
# Run audit experiments in order on one GPU (Slurm). Partition/time go on the sbatch line.
#
#   AUDIT_RESULTS=<dir> CHECKPOINT_ROOT=<dir> sbatch -p <partition> -t HH:MM:SS -o <logdir>/%x_%j.out \
#       experiments/unlearning_audit/run.sbatch sham_scr icl generation length_match
#
# Arguments are module names under experiments.unlearning_audit (quote to pass flags:
# "strength_match --model 3b --grid 1 2"), or `pytest:<test file>`.
# Assets must be prefetched (prefetch.py) and .venv synced on a node with network;
# compute nodes run offline. A failing step is logged and the next one still runs.
#SBATCH -J engram-audit
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
set -u
: "${AUDIT_RESULTS:?set AUDIT_RESULTS so committed results/ are not overwritten}"
: "${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$AUDIT_RESULTS" "$CHECKPOINT_ROOT/ai_engram"
PY=.venv/bin/python
head=$(cat .git/HEAD); [[ $head == ref:* ]] && head=$(cat ".git/${head#ref: }")  # compute nodes have no git
echo "host=$(hostname) commit=$head results=$AUDIT_RESULTS"
"$PY" -c "import torch, transformers, datasets; print('torch', torch.__version__, torch.cuda.get_device_name(0), '| transformers', transformers.__version__, '| datasets', datasets.__version__)"
status=0
for step in "$@"; do
  log="$AUDIT_RESULTS/$(tr -c 'A-Za-z0-9._-' '_' <<< "$step" | sed 's/_*$//').log"
  echo "=== $step $(date -Is)"
  start=$SECONDS
  case "$step" in
    pytest:*) ENGRAM_RUN_TOFU=1 ENGRAM_RUN_TOFU_EVALUATE=1 "$PY" -m pytest -s "${step#pytest:}" ;;
    *) read -r mod flags <<< "$step"
       "$PY" -m "experiments.unlearning_audit.$mod" $flags ;;
  esac > "$log" 2>&1
  rc=$?
  echo "=== $step rc=$rc elapsed=$((SECONDS - start))s log=$log"
  [ $rc -eq 0 ] || status=1
done
exit $status
