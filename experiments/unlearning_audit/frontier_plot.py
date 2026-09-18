#!/usr/bin/env python3
"""Frontier figure: suppression vs destruction. Reads results/alpha_sweep.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")


def main():
    with open(f"{OUT}/alpha_sweep.json") as stream:
        data = json.load(stream)
    rows, base = data["rows"], data["base_retain"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for scale, marker in [("adaptive", "o-"), ("plain", "s--")]:
        selected = [row for row in rows if row["scale"] == scale]
        alpha = [row["alpha"] for row in selected]
        ax1.plot(alpha, [row["retain"] for row in selected], marker, label=scale)
        ax2.plot(alpha, [row["recovery_R"] for row in selected], marker, label=scale)
    ax1.axhline(base, color="gray", ls=":", label=f"base retain {base}")
    ax1.set(xlabel="edit strength α", ylabel="retain NLL (↑ = utility damage)", title="Utility cost")
    ax1.legend()
    ax2.axhline(0, color="k", lw=.6)
    ax2.axhline(1, color="gray", ls=":")
    ax2.set(xlabel="edit strength α", ylabel="recovery fraction R", title="Relearning recovery")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/frontier.png", dpi=130)


if __name__ == "__main__":
    main()
