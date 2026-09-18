#!/usr/bin/env python3
"""Compare a re-run results directory against the committed results/.

Criterion (declared before the Olaf re-run, 2026-09-19):
  verdict leaves (bool/str) must be identical;
  numeric leaves pass when |new - old| <= ATOL + RTOL * |old|.
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


def compare(old, new):
    new_leaves = dict(leaves(new))
    rows = []
    for key, a in leaves(old):
        b = new_leaves.get(key, "<missing>")
        if isinstance(a, bool) or isinstance(a, str) or a is None:
            ok = a == b
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            ok = abs(b - a) <= ATOL + RTOL * abs(a)
        else:
            ok = False
        rows.append({"key": key, "old": a, "new": b, "ok": ok,
                     "verdict": isinstance(a, (bool, str))})
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
        rows = compare(json.loads(old_path.read_text()), json.loads(new_path.read_text()))
        bad = [r for r in rows if not r["ok"]]
        bad_verdicts = [r for r in bad if r["verdict"]]
        report[old_path.name] = {"n": len(rows), "fail": bad}
        print(f"{'OK' if not bad else 'FAIL':4s} {old_path.name}: {len(rows) - len(bad)}/{len(rows)} leaves"
              f" | verdict mismatches {len(bad_verdicts)}")
        for r in bad[:12]:
            print(f"       {r['key']}: {r['old']} -> {r['new']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
