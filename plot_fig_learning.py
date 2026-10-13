"""
plot_fig_learning.py
====================
Learning curves: greedy-policy network PLR versus training iteration for PPO and
DQN, one panel per network size N, with the random-allocation baseline shown for
reference. Curves are averaged over a few fixed networks (see CURVE_SEEDS in
train.py) and lightly smoothed for readability.

Reads results/results_N*.npz and results/summary.json (from train.py).
Saves figures/fig_learning_curves.eps (+ PNG preview).

Run:  python plot_fig_learning.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- font (see plot_fig1_log.py) ----
FONT_MODE = "cm"
PAPER_FONT_PT = 11
if FONT_MODE == "cm":
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    matplotlib.rcParams["axes.unicode_minus"] = False
elif FONT_MODE == "latex":
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
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
C_PPO, C_DQN, C_RANDOM = "#1f77b4", "#d62728", "#9aa0a6"


def smooth(y, w=3):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode="same")
    # fix the ends (convolve 'same' tapers them)
    ys[0] = y[0]
    ys[-1] = np.mean(y[-w:])
    return ys


def _grid_figure(Ns, summary, key, ylabel, fname, baseline_key=None):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
    axes = axes.ravel()
    for ax, N in zip(axes, Ns):
        d = dict(np.load(os.path.join(HERE, "results", f"results_N{N}.npz")))
        pc, dc = d["ppo_" + key], d["dqn_" + key]
        ax.plot(pc[:, 0], smooth(pc[:, 1]), "-", color=C_PPO, lw=2, label="PPO")
        ax.plot(dc[:, 0], smooth(dc[:, 1]), "-", color=C_DQN, lw=2, label="DQN")
        if baseline_key is not None:
            ax.axhline(summary[N][baseline_key], ls="--", color=C_RANDOM, lw=1.4,
                       label="Random")
        ax.set_title(f"$N={N}$")
        ax.grid(True, alpha=0.3)
        if key == "curve":
            ax.set_ylim(bottom=0)
    for ax in axes[2:]:
        ax.set_xlabel("Training iteration")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True, fontsize=PAPER_FONT_PT - 1)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    eps = os.path.join(FIGDIR, fname + ".eps")
    png = os.path.join(FIGDIR, fname + "_preview.png")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", eps)
    print("wrote", png, "(preview)")


def main():
    summary = {r["N"]: r for r in json.load(open(os.path.join(HERE, "results", "summary.json")))}
    Ns = sorted(summary.keys())
    # PLR vs iteration (greedy policy, overlap metric)
    _grid_figure(Ns, summary, "curve", "PLR", "fig_learning_curves",
                 baseline_key="plr_random")
    # training reward vs iteration (mean per-ED success under the reward model)
    _grid_figure(Ns, summary, "rcurve", "Training reward", "fig_training_reward")


if __name__ == "__main__":
    main()
