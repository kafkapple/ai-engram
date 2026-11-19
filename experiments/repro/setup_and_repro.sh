#!/usr/bin/env bash
# ai-engram reproduction harness — grounded in the repo's own docs/tests/notebooks.
# Design: do NOT reinvent experiments. The repo ships example notebooks + gated
# pytest tests that reproduce the paper. This script just orders them cheap->costly
# and gates the heavy GPU tiers behind flags (ponytail: reuse, don't rebuild).
#
# Usage:
#   ./setup_and_repro.sh setup          # clone + env + install + CPU verify (T0)
#   ./setup_and_repro.sh t1             # Qwen3 quickstart (small GPU, qualitative)
#   ./setup_and_repro.sh t2             # TOFU/Llama headline (GPU; needs cached models)
#   ./setup_and_repro.sh t3             # vision notebooks best-effort (research code)
#   ./setup_and_repro.sh report         # aggregate outputs -> one HTML report
#
# Env (override as needed):
#   REPRO_ROOT   default: $HOME/dev/ai-engram        (clone target on the run machine)
#   REPO_URL     default: https://github.com/kafkapple/ai-engram   (the fork)
#   PY           default: python                     (activate your conda/uv env first)
set -euo pipefail

REPRO_ROOT="${REPRO_ROOT:-$HOME/dev/ai-engram}"
REPO_URL="${REPO_URL:-https://github.com/kafkapple/ai-engram}"
PY="${PY:-python}"
OUT="${REPRO_ROOT}/_repro_out"
NB="${REPRO_ROOT}/examples"

log() { printf '\033[1;36m[repro]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[repro:ERR]\033[0m %s\n' "$*" >&2; exit 1; }

cmd_setup() {
  command -v git >/dev/null || die "git not found"
  if [ ! -d "$REPRO_ROOT/.git" ]; then
    log "cloning $REPO_URL -> $REPRO_ROOT"
    git clone "$REPO_URL" "$REPRO_ROOT"
  else
    log "repo exists; git pull"
    git -C "$REPRO_ROOT" pull --ff-only || log "pull skipped (local changes)"
  fi
  mkdir -p "$OUT"
  cd "$REPRO_ROOT"
  log "python: $($PY --version 2>&1)  torch/cuda check:"
  $PY - <<'EOF' || true
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF
  # editable install + pytest + jupyter (nbconvert to run notebooks). ai-engram
  # already pulls torch+transformers; we add the notebook/test tooling.
  log "pip install -e .[dev] + jupyter"
  $PY -m pip install -e ".[dev]" jupyter nbconvert ipykernel datasets
  log "import + version"
  $PY -c "import engram; print('engram', engram.__version__)"
  # T0: fast CPU unit tests — the install/extraction sanity gate.
  log "T0: pytest CPU unit tests (extraction/compat)"
  $PY -m pytest tests/test_extraction.py tests/test_compat.py -q | tee "$OUT/T0_pytest.log"
  log "T0 done. Next: ./setup_and_repro.sh t1"
}

_run_nb() {  # _run_nb <notebook.ipynb>
  local nb="$1"; local base; base="$(basename "$nb" .ipynb)"
  [ -f "$nb" ] || { log "SKIP (missing): $nb"; return 0; }
  log "executing $base (timeout 3600s/cell) ..."
  $PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
      --output "${base}.executed.ipynb" --output-dir "$OUT" "$nb" \
      && log "OK -> $OUT/${base}.executed.ipynb" \
      || log "FAIL (kept partial log): $base  — see note red-team (vision uses research code)"
}

cmd_t1() {  # qualitative, current API, small GPU
  cd "$REPRO_ROOT"; mkdir -p "$OUT"
  _run_nb "$NB/quick_ai_engram_qwen3.ipynb"
}

cmd_t2() {  # headline: TOFU forget10 / Llama-3.2-1B (paper's primary claim)
  cd "$REPRO_ROOT"; mkdir -p "$OUT"
  # test_tofu_evaluate.py importorskip's these but they are NOT in pyproject deps —
  # without them T2b silently SKIPs (rouge_score) or FAILs mid-run (sklearn). Pin here.
  $PY -m pip install -q rouge_score scikit-learn
  # Models to pre-cache (gpu03 is offline-by-policy; WSL can download live):
  #   open-unlearning/tofu_Llama-3.2-1B-Instruct_full   (memorised base)
  #   open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 (gold)
  #   locuslab/TOFU  (dataset)
  # On an offline node export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 first.
  log "T2a: NLL proxy (~6min)"; ENGRAM_RUN_TOFU=1 $PY -m pytest -s tests/test_tofu_unlearn.py | tee "$OUT/T2a_tofu_nll.log"
  log "T2b: Official Overall (~24-45min, needs both models cached)"
  ENGRAM_RUN_TOFU_EVALUATE=1 $PY -m pytest -s tests/test_tofu_evaluate.py | tee "$OUT/T2b_tofu_overall.log"
  log "T2 done. Expected: gold~0.998, plain(a=0.6)~0.706, adaptive(a=1,p=1)~0.817"
}

cmd_t3() {  # vision — BEST-EFFORT. Package covers only Linear+Conv1D; these
            # notebooks pull the paper's broader research code (Conv2d etc.) and
            # may use an earlier API. MLP/MNIST is Linear-only -> most likely to run.
  cd "$REPRO_ROOT"; mkdir -p "$OUT"
  for nb in fig_mlp_mnist fig_resnet18_cifar10 fig_resnet18_cifar100 fig_wae_celeba fig_vit_imagenet1k; do
    _run_nb "$NB/${nb}.ipynb"
  done
}

cmd_report() { cd "$REPRO_ROOT"; $PY "$(dirname "$0")/aggregate_report.py" "$OUT"; }

case "${1:-}" in
  setup)  cmd_setup ;;
  t1)     cmd_t1 ;;
  t2)     cmd_t2 ;;
  t3)     cmd_t3 ;;
  report) cmd_report ;;
  all)    cmd_setup; cmd_t1; cmd_t2; cmd_t3; cmd_report ;;
  *) die "usage: $0 {setup|t1|t2|t3|report|all}" ;;
esac
