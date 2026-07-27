#!/usr/bin/env python3
"""Frontier figure: suppression vs destruction. Reads results/alpha_sweep.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "results")
d = json.load(open(f"{OUT}/alpha_sweep.json"))
rows = d["rows"]; base = d["base_retain"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
for stype, mk in [("adaptive", "o-"), ("plain", "s--")]:
    rr = [r for r in rows if r["scale"] == stype]
    a = [r["alpha"] for r in rr]
    ax1.plot(a, [r["retain"] for r in rr], mk, label=stype)
    ax2.plot(a, [r["recovery_R"] for r in rr], mk, label=stype)
ax1.axhline(base, color="gray", ls=":", label=f"base retain {base}")
ax1.set_xlabel("edit strength α"); ax1.set_ylabel("retain NLL (↑ = utility damage)")
ax1.set_title("Utility cost"); ax1.legend()
ax2.axhline(0, color="k", lw=.6); ax2.axhline(1, color="gray", ls=":")
ax2.fill_between([0.4, 2.1], 0, 1.1, color="#c0392b", alpha=0.06)
ax2.set_xlabel("edit strength α"); ax2.set_ylabel("recovery fraction R (↑ = relearnable = suppression)")
ax2.set_title("Relearning recovery (R>0 = suppression)"); ax2.legend()
fig.suptitle("AI Engram — suppression↔destruction frontier (TOFU/Llama-1B): "
             "usable α is always relearnable; erasure only via utility collapse", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/frontier.png", dpi=130)
print(f"wrote {OUT}/frontier.png")
