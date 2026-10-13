"""
plr_sim.py
==========
Faithful Python port of PLR_Tot_sim.m -- the Monte-Carlo simulator that produced
the reported random-allocation PLR baselines (0.029 / 0.048 / 0.059 / 0.062).

Loss model per packet of a randomly-chosen "desired" ED:
    1. lost if Doppler shift        F_D >= 0.25*BW
    2. else lost if Doppler rate    |F_E| >= BW/(3*2^SF)
    3. else lost if it time-overlaps >=1 packet of any OTHER ED on the SAME
       channel (64 channels total)

Key differences from the .mat-lookup model used elsewhere in this project
(these are what make the numbers match your MATLAB, not the .mat surface):
    * BW is FIXED at 125 kHz; only SF, CR and channel are randomized.
    * 64 orthogonal channels -> collisions only within the same channel.
    * collisions are counted by direct packet time-overlap, not a lookup.

NOTE on one MATLAB quirk faithfully reproduced here:
    the Doppler check uses packet time  t_start + (cnp-1)*ToA   (forward),
    while the collision check uses       t_start - (cnp-1)*ToA   (backward).
    That sign difference is in your .m file; it is preserved so the numbers
    match. Set COLLISION_SIGN = +1 below to use the physically-forward version.
"""

import numpy as np

# ---- constants (identical to PLR_Tot_sim.m) --------------------------
R = 6371000.0
GR = 9.80665
H = 500000.0
C = 299792458.0

V = np.sqrt((GR * R) / (1 + H / R))
W = V / (R + H)
KK = R / (R + H)

E_MIN = 1.0  # degrees
def _d_E(E_deg):
    E = np.deg2rad(E_deg)
    return R * (np.sqrt(((H + R) / R) ** 2 - np.cos(E) ** 2) - np.sin(E))
R_SAT = R * np.arcsin(_d_E(E_MIN) * np.cos(np.deg2rad(E_MIN)) / (R + H))

SAT_DIST = 4 * R_SAT
ENTIRE_SAT_TIME = SAT_DIST / V

SF_POOL = np.array([7, 8, 9, 10])
CH_POOL = np.arange(1, 65)          # 64 channels
CR_POOL = np.array([4/5, 4/6, 5/7, 4/8])
FRQ = 910e6
BW = 125000.0
PAYLOAD = 55 * 8

NUM_IT = 30000                      # Doppler time-grid resolution
TIME_DUR = np.linspace(0, ENTIRE_SAT_TIME, NUM_IT)
RHO = FRQ * R * W / C

DUTY = 0.01
COLLISION_SIGN = -1                 # -1 reproduces MATLAB; +1 = forward/physical


# ---- ED location sampling (matches main.py stadium region) -----------
def ed_loc(N, rng):
    circle_area = 0.5 * np.pi * R_SAT ** 2
    rect_area = 2 * R_SAT * 4 * R_SAT
    A_tot = 2 * circle_area + rect_area
    N_rect = int(np.floor(N * rect_area / A_tot))
    N_cup = int(np.floor(N * circle_area / A_tot))
    N_cdown = N - N_rect - N_cup

    x_rect = -R_SAT + 2 * R_SAT * rng.random(N_rect)
    y_rect = -2 * R_SAT + 4 * R_SAT * rng.random(N_rect)

    r1 = R_SAT * np.sqrt(rng.random(N_cup))
    t1 = np.pi * rng.random(N_cup)
    x_cup, y_cup = r1 * np.cos(t1), 2 * R_SAT + r1 * np.sin(t1)

    r2 = R_SAT * np.sqrt(rng.random(N_cdown))
    t2 = -np.pi * rng.random(N_cdown)
    x_cdown, y_cdown = r2 * np.cos(t2), -2 * R_SAT + r2 * np.sin(t2)

    x = np.concatenate([x_rect, x_cup, x_cdown])
    y = np.concatenate([y_rect, y_cup, y_cdown])
    return x, y


def _doppler_over_time(x, y):
    d_time = np.sqrt(x ** 2 + (y - 2 * R_SAT + V * TIME_DUR) ** 2)
    gamma = 2 * np.arcsin(np.clip(d_time / (2 * R), -1, 1))
    gamma_min = gamma.min()
    num = np.maximum(np.cos(gamma_min) ** 2 - np.cos(gamma) ** 2, 0.0)
    den = 1 + KK ** 2 - 2 * KK * np.cos(gamma)
    return RHO * np.sqrt(num) / np.sqrt(den)


def _idx_at(t):
    return int(np.clip(round(t / ENTIRE_SAT_TIME * (NUM_IT - 1)), 0, NUM_IT - 1))


# ---- one random network realisation ----------------------------------
def _one_realisation(N, rng):
    """Return (n_lost, n_packets) for one random desired ED in a random net."""
    SF = SF_POOL[rng.integers(0, 4, N)]
    CH = CH_POOL[rng.integers(0, 64, N)]
    CR = CR_POOL[rng.integers(0, 4, N)]
    ToA = (PAYLOAD / (SF * CR)) * (2.0 ** SF / BW)

    x, y = ed_loc(N, rng)

    cond = ((x**2 + (y - 2*R_SAT)**2 <= R_SAT**2) |
            (x**2 + (y + 2*R_SAT)**2 <= R_SAT**2))
    Dist = np.where(cond,
                    2*R_SAT - np.abs(y) + np.sqrt(np.clip(R_SAT**2 - x**2, 0, None)),
                    2*np.sqrt(np.clip(R_SAT**2 - x**2, 0, None)))
    Tx_time = Dist / V
    Num_pack = np.floor(DUTY * Tx_time / ToA).astype(int)

    cond_tstart = (x**2 + (y - 2*R_SAT)**2 <= R_SAT**2)
    t_start = np.where(
        cond_tstart, 0.0,
        (2*R_SAT - y - np.sqrt(np.clip(R_SAT**2 - x**2, 0, None))) / V)

    des = rng.integers(0, N)
    if Num_pack[des] == 0:
        return 0, 0

    stat_thr = 0.25 * BW
    dyn_thr = BW / (3 * 2.0 ** SF[des])
    dopp = _doppler_over_time(x[des], y[des])

    # interferers on the same channel
    idx_CH = np.where(CH == CH[des])[0]
    idx_CH = idx_CH[idx_CH != des]
    idx_CH = idx_CH[Num_pack[idx_CH] > 0]

    # all interferer packet intervals (concatenated)
    int_starts, int_ends = [], []
    for j in idx_CH:
        q = np.arange(Num_pack[j])
        s = t_start[j] + q * ToA[j]
        int_starts.append(s)
        int_ends.append(s + ToA[j])
    if int_starts:
        int_starts = np.concatenate(int_starts)
        int_ends = np.concatenate(int_ends)
    else:
        int_starts = np.empty(0)
        int_ends = np.empty(0)

    npk = Num_pack[des]
    n_lost = 0
    for cnp in range(npk):
        tf = t_start[des] + cnp * ToA[des]           # forward (Doppler)
        F_D = dopp[_idx_at(tf)]
        F_E = dopp[_idx_at(tf)] - dopp[_idx_at(tf + ToA[des])]
        if F_D >= stat_thr:
            n_lost += 1
        elif abs(F_E) >= dyn_thr:
            n_lost += 1
        else:
            t_s = t_start[des] + COLLISION_SIGN * cnp * ToA[des]   # MATLAB quirk
            t_e = t_s + ToA[des]
            if int_starts.size and np.any((int_starts < t_e) & (int_ends > t_s)):
                n_lost += 1
    return n_lost, npk


# ---- driver: accumulate until enough losses --------------------------
def simulate_random_plr(N, target_losses=10000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    total_lost = 0
    total_pkts = 0
    reals = 0
    while total_lost <= target_losses:
        l, p = _one_realisation(N, rng)
        total_lost += l
        total_pkts += p
        reals += 1
        if verbose and reals % 2000 == 0:
            print(f"  N={N}: {reals} reals, lost={total_lost}, "
                  f"pkts={total_pkts}, PLR={total_lost/max(total_pkts,1):.4f}")
    plr = total_lost / total_pkts
    return dict(N=N, plr=plr, total_lost=total_lost,
                total_packets=total_pkts, realisations=reals)


if __name__ == "__main__":
    import sys
    Ns = [int(a) for a in sys.argv[1:]] or [100, 500, 800, 1000]
    tgt = 10000
    for N in Ns:
        res = simulate_random_plr(N, target_losses=tgt, seed=0)
        print(f"N={N:5d}  random PLR = {res['plr']:.4f}  "
              f"({res['realisations']} realisations)")
