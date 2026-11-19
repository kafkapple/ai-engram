#!/usr/bin/env python3
"""Dump markdown headers + code + text/stream outputs from an executed notebook."""
import json, sys
nb = json.loads(open(sys.argv[1]).read())
for i, c in enumerate(nb.get("cells", [])):
    src = "".join(c.get("source", [])).strip()
    if c["cell_type"] == "markdown":
        head = "\n".join(l for l in src.splitlines() if l.startswith("#"))
        if head:
            print(f"\n### [md {i}] {head}")
        continue
    outs = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            outs.append("".join(o.get("text", [])))
        elif "data" in o and "text/plain" in o["data"]:
            outs.append("".join(o["data"]["text/plain"]))
    if outs:
        code1 = src.splitlines()[0][:80] if src else ""
        print(f"\n--- [code {i}] {code1}")
        txt = "".join(outs)
        print(txt[-1600:] if len(txt) > 1600 else txt)
