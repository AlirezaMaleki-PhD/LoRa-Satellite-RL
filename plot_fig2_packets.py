"""
plot_fig2_packets.py
====================
Standalone figure for a LaTeX paper: total transmitted packets vs number of EDs,
as a grouped bar chart for Random, PPO and DQN.

Reads results/summary.json (written by train.py) and saves a vector EPS
(plus a PNG preview) into the figures/ folder.

Run from inside the lora_rl folder:
    python plot_fig2_packets.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# FONT MODE  -- pick how the text should look (see plot_fig1_log.py):
#   "cm"      Computer Modern look (default LaTeX article) - no LaTeX needed
#   "latex"   render through your real LaTeX install (needs MiKTeX/TeX Live)
#   "times"   Times/STIX look (IEEE two-column) - no LaTeX needed
#   "default" matplotlib's own font
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
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "summary.json")
FIGDIR = os.path.join(HERE, "figures")

C_RANDOM, C_PPO, C_DQN = "#9aa0a6", "#1f77b4", "#d62728"


def main():
    with open(RESULTS) as f:
        summary = json.load(f)
    summary = sorted(summary, key=lambda s: s["N"])

    Ns = [s["N"] for s in summary]
    rnd = [s["pkts_random"] for s in summary]
    ppo = [s["pkts_ppo"] for s in summary]
    dqn = [s["pkts_dqn"] for s in summary]

    x = np.arange(len(Ns))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(x - w, rnd, w, color=C_RANDOM, label="Random", edgecolor="black", linewidth=0.4)
    ax.bar(x,     ppo, w, color=C_PPO,    label="PPO",    edgecolor="black", linewidth=0.4)
    ax.bar(x + w, dqn, w, color=C_DQN,    label="DQN",    edgecolor="black", linewidth=0.4)

    ax.set_xlabel("$N$")
    ax.set_ylabel("Total transmitted packets")
    ax.set_xticks(x)
    ax.set_xticklabels(Ns)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    eps = os.path.join(FIGDIR, "fig2_packets_vs_N.eps")
    png = os.path.join(FIGDIR, "fig2_packets_vs_N_preview.png")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", eps)
    print("wrote", png, "(preview)")


if __name__ == "__main__":
    main()
