#!/usr/bin/env python3
"""Compare a re-run results directory against the committed results/.

Criterion (declared before reading any re-run output, 2026-09-19):
  load-bearing leaves (LOAD_BEARING) must all pass their own tolerance
    (None = identical; number = absolute tolerance);
  every other leaf: verdicts identical, numbers within ATOL + RTOL * |old|, and >= 95% pass.
Per-item lists and per-layer profiles are skipped (item-level bf16 noise is not a claim).

    python -m experiments.unlearning_audit.compare_results <rerun_dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATOL, RTOL = 0.05, 0.05
SKIP_KEYS = ("per_item", "rel_edit_", "layer_names")
COMMITTED = Path(__file__).with_name("results")
REST_PASS = 0.95
# Tolerances: unedited NLL 0.01 (no pinv, no training); edited NLL 0.05 + 5% via EDIT;
# degeneration rates 0.05; training-derived R 0.10 (recorded seed std up to 0.09).
EDIT = "edit"
LOAD_BEARING = {
    "sham_scr.json": {"S_O_original": 0.01, "base_retain": 0.01, "static.edited": EDIT,
                      "retain.edited": EDIT, "static_gate.pass": None},
    "e2_v2.json": {"S_O_original": 0.01, "arms.gold.C0.nll": 0.01, "arms.edited.C0.nll": EDIT,
                   "arms.edited.C3.nll": EDIT, "arms.edited.C3.degen_rate": 0.05,
                   "confirmatory.ch_C2.extraction_evidence": None,
                   "confirmatory.ch_C4.extraction_evidence": None},
    "c3_lenmatch.json": {"arms.edited.C3L_nll": EDIT, "arms.edited.C3L_degen": 0.05,
                         "arms.gold_ef.C3L_degen": 0.05, "verdict": None},
    "xval_3b.json": {"S_O_original": 0.01, "edited_static": EDIT, "icl.edited.C0.nll": EDIT,
                     "icl.edited.C3L.nll": EDIT, "icl.edited.C3L.degen": 0.05,
                     "R_gold_anchor_edited": 0.10, "prereg.R1_dissociation": None,
                     "prereg.R2_context_damage": None},
    "xval_forget05.json": {"S_O_original": 0.01, "edited_static": EDIT, "icl.edited.C0.nll": EDIT,
                           "icl.edited.C3L.nll": EDIT, "R_gold_anchor_edited": 0.10,
                           "prereg.R1_dissociation": None, "prereg.R2_context_damage": None},
}


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            if not any(key.startswith(s) for s in SKIP_KEYS):
                yield from leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list) and value and not isinstance(value[0], (dict, list)):
        for i, child in enumerate(value):
            yield f"{path}[{i}]", child
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, f"{path}[{i}]")
    else:
        yield path, value


def within(a, b, tol):
    if tol is None or isinstance(a, (bool, str)) or a is None:
        return a == b
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        return False
    limit = ATOL + RTOL * abs(a) if tol == EDIT else tol
    return abs(b - a) <= limit


def compare(old, new, load_bearing):
    new_leaves = dict(leaves(new))
    rows = []
    for key, a in leaves(old):
        b = new_leaves.get(key, "<missing>")
        lb = key in load_bearing
        rows.append({"key": key, "old": a, "new": b, "load_bearing": lb,
                     "ok": within(a, b, load_bearing[key] if lb else EDIT)})
    missing = set(load_bearing) - {r["key"] for r in rows}
    assert not missing, f"LOAD_BEARING keys absent from committed JSON: {missing}"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rerun_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = {}
    for old_path in sorted(COMMITTED.glob("*.json")):
        new_path = args.rerun_dir / old_path.name
        if not new_path.exists():
            report[old_path.name] = "not re-run"
            print(f"-- {old_path.name}: not re-run")
            continue
        rows = compare(json.loads(old_path.read_text()), json.loads(new_path.read_text()),
                       LOAD_BEARING.get(old_path.name, {}))
        lb_bad = [r for r in rows if r["load_bearing"] and not r["ok"]]
        rest = [r for r in rows if not r["load_bearing"]]
        rest_rate = sum(r["ok"] for r in rest) / len(rest) if rest else 1.0
        passed = not lb_bad and rest_rate >= REST_PASS
        report[old_path.name] = {"pass": passed, "load_bearing_fail": lb_bad, "rest_pass_rate": rest_rate,
                                 "rest_fail": [r for r in rest if not r["ok"]]}
        print(f"{'PASS' if passed else 'FAIL'} {old_path.name}: load-bearing {len(lb_bad)} fail"
              f" | rest {rest_rate:.0%} of {len(rest)}")
        for r in lb_bad + [r for r in rest if not r["ok"]][:8]:
            print(f"       {'*' if r['load_bearing'] else ' '} {r['key']}: {r['old']} -> {r['new']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
