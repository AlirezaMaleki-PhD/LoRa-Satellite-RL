"""
plot_fig1_log.py
================
Standalone figure for a LaTeX paper: average PLR vs number of EDs, with a
LOGARITHMIC PLR axis, showing only three curves:

    * Random  (your reported MATLAB values)
    * PPO
    * DQN

Reads results/summary.json (written by train.py) and saves a vector EPS
(plus a PNG preview) into the figures/ folder.

Run from inside the lora_rl folder:
    python plot_fig1_log.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# FONT MODE  -- pick how the text should look:
#   "cm"      Computer Modern look (default LaTeX article font) WITHOUT
#             needing LaTeX installed. Recommended starting point.
#   "latex"   Render ALL text through your real LaTeX installation (exact
#             match to your paper). Needs MiKTeX/TeX Live + Ghostscript.
#   "times"   Times/STIX look (matches IEEE-style two-column papers),
#             no LaTeX needed.
#   "default" matplotlib's own font (no change).
# Set PAPER_FONT_PT to your document's font size (e.g. 10 or 11).
# ======================================================================
FONT_MODE = "cm"
PAPER_FONT_PT = 11

if FONT_MODE == "latex":
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
elif FONT_MODE == "cm":
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    matplotlib.rcParams["axes.unicode_minus"] = False
elif FONT_MODE == "times":
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False

matplotlib.rcParams["font.size"] = PAPER_FONT_PT
matplotlib.rcParams["axes.linewidth"] = 0.8

# ---- embed proper fonts in the EPS (avoids Type-3 fonts some journals reject)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "summary.json")
FIGDIR = os.path.join(HERE, "figures")

# colours
C_RANDOM, C_PPO, C_DQN = "#555555", "#1f77b4", "#d62728"

# On a log axis a value of exactly 0 cannot be drawn; anything <= this floor
# is clipped to it so the marker still appears. (PPO can be ~0 at small N.)
LOG_FLOOR = 1e-4


def main():
    with open(RESULTS) as f:
        summary = json.load(f)
    summary = sorted(summary, key=lambda s: s["N"])

    Ns = [s["N"] for s in summary]
    rnd = [s["plr_random_reported"] for s in summary]
    ppo = [s["plr_ppo"] for s in summary]
    dqn = [s["plr_dqn"] for s in summary]

    def clip(vals):
        out, floored = [], False
        for v in vals:
            if v is None or not np.isfinite(v) or v <= 0:
                out.append(LOG_FLOOR); floored = True
            elif v < LOG_FLOOR:
                out.append(LOG_FLOOR); floored = True
            else:
                out.append(v)
        return out, floored

    ppo_c, ppo_floored = clip(ppo)
    dqn_c, dqn_floored = clip(dqn)
    if ppo_floored or dqn_floored:
        print(f"[note] some PLR values were <= {LOG_FLOOR:g} and were clipped to "
              f"{LOG_FLOOR:g} so they render on the log axis.")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(Ns, rnd, "o-", color=C_RANDOM, lw=2, ms=7, label="Random")
    ax.semilogy(Ns, ppo_c, "s-", color=C_PPO, lw=2, ms=7, label="PPO")
    ax.semilogy(Ns, dqn_c, "^-", color=C_DQN, lw=2, ms=7, label="DQN")

    ax.set_xlabel("$N$")
    ax.set_ylabel("PLR")
    ax.set_xticks(Ns)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    eps = os.path.join(FIGDIR, "fig1_plr_vs_N_log.eps")
    png = os.path.join(FIGDIR, "fig1_plr_vs_N_log.png")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")   # preview only
    plt.close(fig)
    print("wrote", eps)
    print("wrote", png, "(preview)")


if __name__ == "__main__":
    main()
