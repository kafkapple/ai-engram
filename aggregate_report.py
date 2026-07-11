#!/usr/bin/env python3
"""Aggregate ai-engram reproduction outputs into one self-contained HTML report.

Quant: parse the TOFU pytest logs for the Overall/NLL numbers vs paper targets.
Qual: harvest image outputs already embedded in the executed notebooks.
No re-plotting — the notebooks/tests are the source of truth (ponytail).

Usage: python aggregate_report.py <OUT_DIR>   # writes <OUT_DIR>/report.html
"""
from __future__ import annotations
import base64, json, re, sys
from pathlib import Path

# Paper targets (docs/tofu.md) — the reproduction is judged against these.
PAPER = {"gold": 0.998, "plain (a=0.6)": 0.698, "adaptive-norm (a=1,p=1)": 0.818}


def harvest_images(nb_path: Path):
    """Yield (title, data-uri) for every image output in an executed notebook."""
    try:
        nb = json.loads(nb_path.read_text())
    except Exception:
        return
    for i, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            for mime in ("image/png", "image/jpeg"):
                if mime in data:
                    b64 = data[mime]
                    b64 = "".join(b64) if isinstance(b64, list) else b64
                    yield f"{nb_path.stem} · cell {i}", f"data:{mime};base64,{b64}"


def parse_overall(log_text: str):
    """Pull the reproduced Overall per condition from the eval's printed table.

    Log line form: '  gold (retain90)  :  0.998   0.998   0.000' — the label word
    is left of ':' and the FIRST float right of ':' is 'ours'. Keying off ':'
    avoids matching digits inside labels (retain90, a=0.6). Best-effort.
    """
    rows = {}
    for line in log_text.splitlines():
        if ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        for label in ("gold", "plain", "adaptive"):
            if label in lhs.lower():
                m = re.search(r"([01]\.\d{2,4})", rhs)
                if m:
                    rows[label] = float(m.group(1))
    return rows


def main(out_dir: str):
    out = Path(out_dir)
    parts = ['<meta charset="utf-8"><title>ai-engram reproduction</title>',
             '<style>body{font:14px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}'
             'table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:4px 10px}'
             'img{max-width:100%;border:1px solid #eee;margin:.5rem 0}h2{margin-top:2rem}'
             '.ok{color:#127c2b}.bad{color:#b00}code{background:#f4f4f4;padding:1px 4px}</style>',
             '<h1>ai-engram — reproduction report</h1>']

    # --- Quant: TOFU ---
    parts.append("<h2>Quantitative — TOFU forget10 / Llama-3.2-1B</h2>")
    log = out / "T2b_tofu_overall.log"
    if log.exists():
        got = parse_overall(log.read_text(errors="ignore"))
        parts.append("<table><tr><th>condition</th><th>reproduced</th><th>paper</th><th>|Δ|</th></tr>")
        for k, paper_v in PAPER.items():
            short = next((lab for lab in ("gold", "plain", "adaptive") if lab in k.lower()), k.split()[0])
            g = got.get(short)
            cell = "—" if g is None else f"{g:.3f}"
            d = "" if g is None else f"{abs(g-paper_v):.3f}"
            cls = "" if g is None else (' class="ok"' if abs(g-paper_v) <= 0.02 else ' class="bad"')
            parts.append(f"<tr><td>{k}</td><td{cls}>{cell}</td><td>{paper_v:.3f}</td><td>{d}</td></tr>")
        parts.append("</table><p>Green = within ±0.02 of paper.</p>")
    else:
        parts.append("<p><em>T2b log not found — run <code>./setup_and_repro.sh t2</code>.</em></p>")

    for name, title in [("T2a_tofu_nll.log", "NLL proxy (forget should rise, retain hold)"),
                        ("T0_pytest.log", "T0 CPU unit tests")]:
        f = out / name
        if f.exists():
            tail = "\n".join(f.read_text(errors="ignore").splitlines()[-12:])
            parts.append(f"<h3>{title}</h3><pre><code>{tail}</code></pre>")

    # --- Qual: notebook figures ---
    parts.append("<h2>Qualitative — figures from executed notebooks</h2>")
    execed = sorted(out.glob("*.executed.ipynb"))
    if not execed:
        parts.append("<p><em>No executed notebooks yet — run t1/t3.</em></p>")
    for nb in execed:
        imgs = list(harvest_images(nb))
        parts.append(f"<h3>{nb.stem}</h3>")
        if not imgs:
            parts.append("<p><em>(no image outputs)</em></p>")
        for title, uri in imgs:
            parts.append(f'<div><small>{title}</small><br><img src="{uri}"></div>')

    report = out / "report.html"
    report.write_text("\n".join(parts))
    print(f"[report] wrote {report}  ({len(execed)} notebooks, TOFU log={'yes' if log.exists() else 'no'})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
