# TOFU unlearning audit

This directory is the single source of truth for the July 2026 audit formerly
stored in the separate `ai-engram-repro` repository and as untracked
`gpu03_*.py` files.

## Status

- Reproduction and target ablations: complete.
- Relearning, novel-author, and entity-familiarity controls: support
  suppression rather than erasure.
- Alpha sweep: stronger edits trade utility for damage; they do not establish
  deeper erasure.
- Length-matched C3 control: promoted.
- Cross-validation: dissociation direction repeats on forget05 and 3B, while
  collapse does not; alpha is not transferable across scale.
- Remaining limitation: conclusions are behavioral and TOFU/model-family
  specific. See `results/final_report_260711.html`.

## Layout

- `common.py`: model loading, engram collection/editing, embeddings, paths.
- Other Python files: one experiment each, named by purpose rather than host.
- `results/`: compact final JSON/figures/report. Logs, checkpoints, executed
  notebooks, caches, and environment bootstrap scripts are intentionally not
  versioned.

Install and run from the repository root:

```bash
pip install -e '.[audit]'
python -m experiments.unlearning_audit.entity_control
python -m experiments.unlearning_audit.alpha_sweep
```

GPU placement is a runtime concern; no file or module is named after `gpu03`.
