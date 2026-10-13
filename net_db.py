"""
net_db.py
=========
Database (PLR_N*_new.mat) collision model for the RL reward.

Implements the per-packet PLR of your reward equation

    PLR^(pp)(tau, ToA) = 1 - 1{ xi(tau) <= 0.25 BW }
                           * 1{ |xi(tau) - xi(tau+ToA)| <= BW/(3 2^SF) }
                           * ( 1 - q(tau, ToA) )

where q(tau, ToA) is the per-packet collision loss probability read from
PLR_N{N}_new.mat (row = closest start-time in ts_des_range, column = combo whose
ToA matches ToA). This is exactly `old_packet_success_probability` in main.py:
a packet that fails either Doppler test contributes success 0; otherwise its
success probability is (1 - q).

Per-ED reward (your Eq. reward):
    reward_i = (1/N_p) * sum_k [ 1 - PLR^(pp)(tau_{i,k}) ]
with packets placed consecutively from the chosen burst start,
    tau_{i,k} = burst_i + (k-1) * ToA_i .
"""

import numpy as np
from scipy.io import loadmat

import config
import net_scenario as ns

# shared start-time axis (row index of every PLR_N*_new matrix)
_TS_DES = loadmat(config.ts_des_range_path())["ts_des_range"].ravel().astype(float)

_DB_CACHE = {}


def load_database(N):
    """Load and cache the collision-PLR surface + combo table for size N."""
    if N in _DB_CACHE:
        return _DB_CACHE[N]
    m = loadmat(config.plr_mat_path(N))
    P = m[f"PLR_N{N}_new"].astype(float)            # (n_ts, 48)
    if getattr(config, "Q_TIME_AVERAGE", False):
        # collapse q to a pure function of ToA: q(ToA) = mean_t q(t, ToA).
        # Removes the time dependence so the agent gets no (misleading) signal
        # to shift its burst in time; only the ToA->collision lesson remains.
        P = np.repeat(P.mean(axis=0, keepdims=True), P.shape[0], axis=0)
    db = dict(
        PLR=P,                                       # (n_ts, 48)
        ToA=m["ToA_pool_new"].ravel().astype(float),
        SF=m["SF_all"].ravel().astype(float),
        CR=m["CR_all"].ravel().astype(float),
        BW=m["BW_all"].ravel().astype(float),
        ts=_TS_DES,
    )
    # map each of our combos (sorted by ToA in net_scenario) to the database
    # column with the closest ToA, so indexing is robust to ordering.
    db["combo2col"] = np.array(
        [int(np.argmin(np.abs(db["ToA"] - t))) for t in ns.COMBO_TOA])
    _DB_CACHE[N] = db
    return db


def _ts_rows(db, tau):
    """Nearest ts_des_range row index for each start time in tau (vectorised)."""
    return np.abs(db["ts"][:, None] - tau[None, :]).argmin(axis=0)


def evaluate_network_db(sc, combos, timings, N=None):
    """
    Score a full allocation with the database collision model.

    Returns dict(plr, total_packets, per_ed_loss, n_packets, combos, timings),
    where per_ed_loss = 1 - reward_i (your Eq. reward) and plr is the
    packet-weighted expected PLR over all packets of all EDs.
    """
    combos = np.asarray(combos, int)
    timings = np.asarray(timings, int)
    N = sc.N if N is None else N
    db = load_database(N)

    toa, sf, bw, npk, burst, gap = sc._packets(combos, timings)

    per_ed_loss = np.ones(sc.N)
    total_success = 0.0
    total_pkts = 0

    for i in range(sc.N):
        n = npk[i]
        if n == 0:
            per_ed_loss[i] = 1.0                 # transmits nothing -> worst
            continue
        k = np.arange(n)
        tau = burst[i] + k * gap[i]              # packets (spread or burst)

        # --- Doppler gates (xi at tau and tau+ToA), same thresholds as main.py
        xi_s = sc._doppler(i, tau)
        xi_e = sc._doppler(i, tau + toa[i])
        shift_ok = xi_s < 0.25 * bw[i]
        rate_ok = np.abs(xi_s - xi_e) < bw[i] / (3.0 * 2.0 ** sf[i])
        gate = shift_ok & rate_ok

        # --- collision q(tau, ToA) from the database
        col = db["combo2col"][combos[i]]
        rows = _ts_rows(db, tau)
        q = db["PLR"][rows, col]

        # per-packet success = gate * (1 - q)  (= 1 - PLR^(pp))
        p_success = np.where(gate, 1.0 - q, 0.0)

        per_ed_loss[i] = 1.0 - p_success.mean()  # 1 - reward_i
        total_success += p_success.sum()
        total_pkts += n

    plr = 1.0 - total_success / total_pkts if total_pkts else 1.0
    return dict(plr=plr, total_packets=int(total_pkts),
                per_ed_loss=per_ed_loss, n_packets=npk,
                combos=combos, timings=timings)
