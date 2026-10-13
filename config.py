"""
Central configuration for the LoRa-satellite RL project.

Everything that a user might want to tweak (paths, network sizes,
hyper-parameters, the random-allocation baselines) lives here so the
other files stay clean.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# Folder that contains the PLR_N*_new.mat files. By default we look in the
# same folder as the code. Change this if your .mat files live elsewhere.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(DATA_DIR, "results")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

# Filename pattern for the per-N collision PLR files.
# e.g. PLR_N100_new.mat, PLR_N500_new.mat, ...
def plr_mat_path(N):
    return os.path.join(DATA_DIR, f"PLR_N{N}_new.mat")

def ts_des_range_path():
    return os.path.join(DATA_DIR, "ts_des_range.mat")

# Reward / evaluation collision model:
#   "database" -> per-packet collision probability q(t,ToA) from PLR_N*_new.mat
#                 (matches your PLR^(pp) reward equation)
#   "overlap"  -> direct same-channel time-overlap (PLR_Tot_sim.m style)
# REWARD_MODE is what the agent TRAINS on; EVAL_MODE is how the reported PLR /
# packet counts are MEASURED. Train on q, evaluate by real overlap counting.
REWARD_MODE = "database"
EVAL_MODE = "overlap"

# Packet placement:
#   "spread" -> each packet has its own start time, duty-cycle-spaced across the
#               visibility window (spacing = ToA / duty cycle). Physical LoRaWAN.
#   "burst"  -> packets sent back-to-back (spacing = ToA) from one start time.
PACKET_MODE = "burst"


# ----------------------------------------------------------------------
# Network / physics constants (must match main.py)
# ----------------------------------------------------------------------
SATELLITE_HEIGHT = 500_000          # metres
NUM_DOPPLER_SAMPLES = 2000
DUTY_CYCLE = 0.01
PL = 55 * 8                          # payload constant C in the ToA formula
N_COMBOS = 48                        # 4 SF x 4 CR x 3 BW
NUM_TS_SAMPLES = 1000               # rows of the PLR matrix (start-time grid)

# The four network sizes we study.
N_LIST = [100, 500, 800, 1000]

# ----------------------------------------------------------------------
# Distance constraint
# ----------------------------------------------------------------------
# The problem statement defines d_max = r_SAT * sqrt(2) and d_min = 0, and
# slices [d_min, d_max] into N_COMBOS distance bins. An ED whose (closest
# approach) distance to the footprint centre falls in bin i may only use a
# combination with ToA >= ToA[i] (i.e. combo index >= i, since ToA is sorted).
# Set DMAX_MODE = "sqrt2" to follow the statement literally, or "r_sat" to use
# r_SAT (which spreads EDs across all 48 slots and makes the scatter figure
# use the full diagonal).
DMAX_MODE = "r_sat"                 # "sqrt2" | "r_sat"

# ----------------------------------------------------------------------
# Collision model (overlap simulator, matches PLR_Tot_sim.m)
# ----------------------------------------------------------------------
# Number of orthogonal channels. Collisions are only counted between EDs on
# the SAME channel (channel is assigned randomly, not controlled by the agent).
N_CHANNELS = 64
# Sign used for the collision-time of packet cnp:  t_start + SIGN*cnp*ToA.
# -1 reproduces your MATLAB exactly; +1 is the forward/physical version.
COLLISION_SIGN = -1
# Bandwidth set the AGENT may choose from (Hz). The random baseline fixes BW to
# 125 kHz (as in your MATLAB); the RL agent is allowed the full set.
BW_SET = [125_000, 250_000, 500_000]
SF_SET = [7, 8, 9, 10]
CR_SET = [4/5, 4/6, 5/7, 4/8]

# ----------------------------------------------------------------------
# Timing action
# ----------------------------------------------------------------------
# Number of discrete "where to place the packet burst inside the visibility
# window" options the agent can choose from (0 = as early as possible ... 1 =
# as late as possible). Packets are always spread with duty-cycle-safe spacing.
N_TIMING = 5

# ----------------------------------------------------------------------
# Random-allocation baselines supplied by you (used directly in Figure 1 so
# the plotted numbers match what you already reported). The training script
# also measures its own random baseline for cross-checking.
# ----------------------------------------------------------------------
RANDOM_PLR_BASELINE = {
    100: 0.029,
    500: 0.048,
    800: 0.059,
    1000: 0.062,
}

# ----------------------------------------------------------------------
# RL hyper-parameters
# ----------------------------------------------------------------------
SEED = 42
# trained policies (and the random baselines) are evaluated on this many fresh
# network realisations and averaged, for more robust / credible numbers.
N_EVAL_SCENARIOS = 40

PPO = dict(
    hidden=64,
    lr=3e-3,
    gamma=0.95,
    gae_lambda=0.95,
    clip=0.2,
    epochs=4,
    entropy_coef=0.01,
    value_coef=0.5,
    n_iterations=60,      # policy-update iterations
    ends_per_iter=1,      # how many full passes over the ED set per iteration
)

DQN = dict(
    hidden=64,
    lr=1e-3,
    gamma=0.95,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_iters=40,
    batch_size=128,
    buffer_size=20_000,
    target_sync=200,      # steps between target-network syncs
    n_iterations=60,      # passes over the ED set
    grad_steps_per_iter=200,
)

# Use time-averaged q(ToA)=mean_t q(t,ToA) so the collision term depends only
# on ToA (removes the misleading timing signal). Only used in database mode.
Q_TIME_AVERAGE = True
