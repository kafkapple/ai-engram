"""engram.benchmarks.tofu — the parts that must be right without a GPU or the Hub.

The end-to-end path (collect → edit → score → search) is exercised on the real TOFU model by
tests/test_tofu_unlearn.py under ENGRAM_RUN_TOFU=1. These cover the pure pieces: preprocessing
and masking, batching, the Report container, the search objective, and argument validation.

Run offline; CPU-only; deterministic.
"""
import math

import pytest
import torch

from engram.benchmarks import tofu
from engram.benchmarks.tofu import IGNORE, DEFAULT_ALPHA, SCALES, Report, _collate, _objective


class _Tok:
    """A stand-in tokenizer with the two methods preprocessing relies on."""

    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, chat, tokenize=True, add_generation_prompt=False,
                            return_dict=False, **kw):
        ids = [1]
        for m in chat:
            ids += [len(m["content"]) % 7 + 10] * (len(m["content"].split()) or 1)
        if add_generation_prompt:
            ids += [99]
        return ids


# T1: only answer tokens carry labels; the prompt is masked with IGNORE.
def test_preprocess_masks_prompt_only():
    tok = _Tok()
    out = tofu._preprocess(tok, "Who wrote it?", "An author did.")
    ids, labels = out["input_ids"], out["labels"]
    assert len(ids) == len(labels)
    n_prompt = sum(1 for l in labels if l == IGNORE)
    assert 0 < n_prompt < len(labels)                 # some masked, some not
    assert labels[-1] == tok.eos_token_id              # answer ends with EOS
    assert all(l == IGNORE for l in labels[:n_prompt])
    assert all(l != IGNORE for l in labels[n_prompt:])


# T2: collate right-pads ids with pad_id, labels with IGNORE, and masks padding out.
def test_collate_pads_and_masks():
    fn = _collate(pad_id=0)
    b = fn([{"input_ids": [5, 6, 7], "labels": [IGNORE, 6, 7]},
            {"input_ids": [8], "labels": [8]}])
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert b["labels"].tolist() == [[IGNORE, 6, 7], [8, IGNORE, IGNORE]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


# T3: the two paper conditions exist with the paper's alphas.
def test_paper_conditions():
    assert set(SCALES) == {"plain", "adaptive"}
    assert DEFAULT_ALPHA == {"plain": 0.6, "adaptive": 1.0}
    for name, mk in SCALES.items():
        assert callable(mk())


# T4: Report deltas and selectivity.
def test_report_selectivity():
    r = Report(level="quick", forget_nll=2.0, retain_nll=0.5, forget_delta=1.8, retain_delta=0.36)
    assert math.isclose(r.selectivity, 5.0)
    assert Report(level="quick", forget_nll=1.0, retain_nll=1.0).selectivity is None
    assert "dF=+1.800" in repr(r)


# T5: the objective — Overall needs the full level; constrained enforces the utility floor.
def test_objective():
    base = Report(level="utility", forget_nll=0.1, retain_nll=0.1, utility=0.60)
    good = Report(level="utility", forget_nll=2.0, retain_nll=0.4, forget_delta=1.9,
                  retain_delta=0.3, utility=0.56)
    bad = Report(level="utility", forget_nll=3.0, retain_nll=2.0, forget_delta=2.9,
                 retain_delta=1.9, utility=0.40)
    assert _objective(good, "constrained", 0.9, base) == pytest.approx(1.9)
    assert _objective(bad, "constrained", 0.9, base) == -math.inf     # below the floor
    with pytest.raises(ValueError):
        _objective(good, "overall", 0.9, base)                         # no Overall at this level
    full = Report(level="full", forget_nll=2.0, retain_nll=0.4, overall=0.81)
    assert _objective(full, "overall", 0.9, base) == 0.81
    assert _objective(good, lambda r: -r.retain_delta, 0.9, base) == pytest.approx(-0.3)
    with pytest.raises(ValueError):
        _objective(good, "nonsense", 0.9, base)


# T6: argument validation fails early, before any model is touched.
def test_validation_before_work():
    with pytest.raises(ValueError):
        tofu.evaluate(None, None, splits={}, level="nope")
    with pytest.raises(ValueError):
        tofu.run(None, None, scale="nope", splits={})


# T7: the official-metric port is importable and exposes what the module calls.
def test_official_port_surface():
    from engram.benchmarks import _tofu_official as off

    for name in ("compute_model_utility", "compute_full_tofu", "evaluate_scores"):
        assert callable(getattr(off, name))


# T8: the proxy objective, and objective/level mismatches rejected up front.
def test_proxy_objective_and_early_check():
    from engram.benchmarks.tofu import _check_objective

    base = Report(level="quick", forget_nll=0.1, retain_nll=0.1)
    r = Report(level="quick", forget_nll=2.0, retain_nll=0.4, forget_delta=1.9, retain_delta=0.3)
    assert _objective(r, "proxy", 0.9, base) == pytest.approx(1.6)
    _check_objective("proxy", "quick")
    _check_objective("overall", "full")
    _check_objective(lambda rep: 0.0, "quick")
    with pytest.raises(ValueError):
        _check_objective("overall", "quick")
    with pytest.raises(ValueError):
        _check_objective("constrained", "quick")
    with pytest.raises(ValueError):
        tofu.search(None, None, objective="overall", final_level="quick", splits={})
    with pytest.raises(ValueError):
        tofu.search(None, None, scales=("nope",), objective="proxy", final_level="quick", splits={})
