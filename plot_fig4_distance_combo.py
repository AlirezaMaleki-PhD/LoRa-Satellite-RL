"""
plot_fig4_distance_combo.py
===========================
Standalone figure for a LaTeX paper: each ED's closest-approach distance vs the
combination index the learned policy assigns it, showing that the distance
constraint (combo index >= slot(distance)) is always satisfied - no points fall
in the shaded infeasible region below the boundary line.

Reads results/results_N{FOCUS_N}.npz (written by train.py) and saves a vector
EPS (plus a PNG preview) into the figures/ folder.

Run from inside the lora_rl folder:
    python plot_fig4_distance_combo.py
Change FOCUS_N below to plot a different network size.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# Which network size to plot (must have been produced by train.py):
FOCUS_N = 1000
# ======================================================================
# FONT MODE  -- see plot_fig1_log.py for details:
#   "cm" | "latex" | "times" | "default"
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
FIGDIR = os.path.join(HERE, "figures")

N_COMBOS = 48
C_PPO, C_DQN = "#1f77b4", "#d62728"


def main():
    npz = os.path.join(HERE, "results", f"results_N{FOCUS_N}.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"Missing {npz}. Run train.py first "
                         f"(and set FOCUS_N to a size you trained).")
    d = dict(np.load(npz))

    dist_km = d["d_closest"] / 1e3
    d_max_km = float(d["d_max"]) / 1e3

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # constraint boundary and infeasible region
    xs = np.linspace(0, dist_km.max(), 200)
    boundary = N_COMBOS * (xs / d_max_km)
    ax.plot(xs, boundary, "k--", lw=1.5, label="constraint boundary")
    ax.fill_between(xs, 0, boundary, color="0.6", alpha=0.25, label="infeasible region")

    # jitter combo index vertically a touch so overlapping points are visible
    rng = np.random.default_rng(0)
    for combos, color, lab in [(d["ppo_combos"], C_PPO, "PPO"),
                               (d["dqn_combos"], C_DQN, "DQN")]:
        jit = rng.uniform(-0.25, 0.25, size=len(combos))
        ax.scatter(dist_km, combos + jit, s=14, alpha=0.55, color=color,
                   edgecolors="none", label=lab)

    ax.set_xlabel("Closest-approach distance to footprint centre (km)")
    ax.set_ylabel("Chosen combination index")
    ax.set_ylim(-1, N_COMBOS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    eps = os.path.join(FIGDIR, f"fig4_distance_vs_combo_N{FOCUS_N}.eps")
    png = os.path.join(FIGDIR, f"fig4_distance_vs_combo_N{FOCUS_N}_preview.png")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", eps)
    print("wrote", png, "(preview)")


if __name__ == "__main__":
    main()
