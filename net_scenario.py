"""
net_scenario.py
===============
Builds ONE fixed network of N EDs and evaluates the packet-loss ratio (PLR) of
any full allocation using the SAME loss model as PLR_Tot_sim.m:

    * per packet Doppler-shift and Doppler-rate loss
    * collision loss by direct time-overlap against OTHER EDs on the SAME
      channel (64 random channels)

Unlike the earlier .mat-lookup version, collisions are computed directly from
the chosen allocation, so PPO/DQN are scored on exactly the model behind your
reported random baseline. Channel is random (not controlled by the agent); the
agent controls (SF, CR, BW) via a combo index and the burst timing.

The scenario is fixed per (N, seed) so Random / PPO / DQN are compared on an
identical network.
"""

import numpy as np
import config
import plr_sim as P          # reuse the exact physical constants


# ---- build the 48 combos (SF x CR x BW), sorted by ToA ----------------
def _build_combos():
    rows = []
    for sf in config.SF_SET:
        for cr in config.CR_SET:
            for bw in config.BW_SET:
                toa = (P.PAYLOAD / (sf * cr)) * (2.0 ** sf / bw)
                rows.append((toa, sf, cr, bw))
    rows.sort(key=lambda r: r[0])          # ascending ToA
    toa = np.array([r[0] for r in rows])
    sf = np.array([r[1] for r in rows])
    cr = np.array([r[2] for r in rows])
    bw = np.array([r[3] for r in rows])
    return toa, sf, cr, bw


COMBO_TOA, COMBO_SF, COMBO_CR, COMBO_BW = _build_combos()
N_COMBOS = len(COMBO_TOA)                   # 48


def _closest_distance(x, y):
    """Perpendicular (closest-approach) distance of each ED to the ground track."""
    within = np.abs(y) <= 2 * P.R_SAT
    return np.where(within, np.abs(x),
                    np.sqrt(x ** 2 + (np.abs(y) - 2 * P.R_SAT) ** 2))


class Scenario:
    def __init__(self, N, seed=0):
        self.N = N
        rng = np.random.default_rng(seed)

        # --- geometry (same sampling as main.py / plr_sim) ---
        x, y = P.ed_loc(N, rng)
        self.x, self.y = x, y
        self.channel = rng.integers(0, config.N_CHANNELS, N)

        cond = ((x**2 + (y - 2*P.R_SAT)**2 <= P.R_SAT**2) |
                (x**2 + (y + 2*P.R_SAT)**2 <= P.R_SAT**2))
        Dist = np.where(cond,
                        2*P.R_SAT - np.abs(y) + np.sqrt(np.clip(P.R_SAT**2 - x**2, 0, None)),
                        2*np.sqrt(np.clip(P.R_SAT**2 - x**2, 0, None)))
        self.Txw = Dist / P.V

        cond_ts = (x**2 + (y - 2*P.R_SAT)**2 <= P.R_SAT**2)
        self.t_start = np.where(
            cond_ts, 0.0,
            (2*P.R_SAT - y - np.sqrt(np.clip(P.R_SAT**2 - x**2, 0, None))) / P.V)

        # --- Doppler geometry per ED (analytic; evaluated at packet times) ---
        self.d_closest = _closest_distance(x, y)
        gmin = 2 * np.arcsin(np.clip(self.d_closest / (2 * P.R), -1, 1))
        self.cos_gmin_sq = np.cos(gmin) ** 2

        # --- distance -> slot constraint ---
        d_max = P.R_SAT * (np.sqrt(2) if config.DMAX_MODE == "sqrt2" else 1.0)
        self.d_max = d_max
        self.slot = np.clip((N_COMBOS * self.d_closest / d_max).astype(int),
                            0, N_COMBOS - 1)
        self.allowed_mask = (np.arange(N_COMBOS)[None, :] >= self.slot[:, None])

        # --- channel load feature ---
        load = np.zeros(config.N_CHANNELS)
        for ch in self.channel:
            load[ch] += 1
        self.channel_load = load[self.channel]        # per-ED

        # normalisation helpers
        self._d_ref = d_max
        self._t_ref = P.ENTIRE_SAT_TIME
        self._txw_ref = max(self.Txw.max(), 1e-9)
        self._load_ref = max(self.channel_load.max(), 1.0)

    # ------------------------------------------------------------------
    def _doppler(self, ed, t):
        """Analytic Doppler shift magnitude for ED `ed` at time(s) t."""
        d = np.sqrt(self.x[ed]**2 + (self.y[ed] - 2*P.R_SAT + P.V*t)**2)
        g = 2 * np.arcsin(np.clip(d / (2*P.R), -1, 1))
        num = np.maximum(self.cos_gmin_sq[ed] - np.cos(g)**2, 0.0)
        den = 1 + P.KK**2 - 2*P.KK*np.cos(g)
        return P.RHO * np.sqrt(num) / np.sqrt(den)

    # ------------------------------------------------------------------
    def _packets(self, combos, timings):
        """Return per-ED packet timing. `gap` is the spacing between consecutive
        packet start times: ToA (burst) or ToA/duty (spread across the window).
        Each packet still has duration ToA."""
        toa = COMBO_TOA[combos]
        sf = COMBO_SF[combos]
        bw = COMBO_BW[combos]
        npk = np.floor(config.DUTY_CYCLE * self.Txw / toa).astype(int)
        npk = np.clip(npk, 0, None)
        if config.PACKET_MODE == "spread":
            gap = toa / config.DUTY_CYCLE          # duty-cycle spacing
        else:
            gap = toa.copy()                       # back-to-back burst
        f = timings / max(config.N_TIMING - 1, 1)
        slack = np.maximum(self.Txw - npk * gap, 0.0)
        burst = self.t_start + f * slack
        return toa, sf, bw, npk, burst, gap

    # ------------------------------------------------------------------
    def evaluate_network(self, combos, timings):
        """
        Score a full allocation with the SAME model as PLR_Tot_sim.m.

        Per packet of every ED:
          1) Doppler shift / rate checked at the FORWARD time  burst + cnp*ToA
          2) if it survives, collision checked between this ED's packet at the
             COLLISION_SIGN time (burst + SIGN*cnp*ToA, SIGN=-1 -> backward, as
             in your MATLAB) and every OTHER same-channel ED's FORWARD packets.

        Returns packet-weighted PLR, total packets, and per-ED loss fraction.
        """
        combos = np.asarray(combos, int)
        timings = np.asarray(timings, int)
        toa, sf, bw, npk, burst, gap = self._packets(combos, timings)

        # ---- forward packet intervals per ED (used for Doppler AND as the
        #      interferer set other EDs collide against) ----
        fwd_s = [None] * self.N        # forward starts per ED
        for i in range(self.N):
            if npk[i] > 0:
                fwd_s[i] = burst[i] + np.arange(npk[i]) * gap[i]

        # per-channel pooled FORWARD interferer intervals (with ED labels)
        chan_pack = {}                 # channel -> (ed_ids, starts, ends)
        for c in range(config.N_CHANNELS):
            members = np.where(self.channel == c)[0]
            members = [j for j in members if npk[j] > 0]
            if not members:
                continue
            eds, ss, ee = [], [], []
            for j in members:
                s = fwd_s[j]
                eds.append(np.full(s.size, j))
                ss.append(s)
                ee.append(s + toa[j])
            chan_pack[c] = (np.concatenate(eds),
                            np.concatenate(ss), np.concatenate(ee))

        per_ed_loss = np.ones(self.N)
        total_lost = 0
        total_pkts = 0

        for i in range(self.N):
            n = npk[i]
            if n == 0:
                per_ed_loss[i] = 1.0            # transmitted nothing -> worst
                continue
            cnp = np.arange(n)
            fwd = burst[i] + cnp * gap[i]                      # Doppler time
            if config.PACKET_MODE == "spread":
                col = fwd                                      # physical, forward
            else:
                col = burst[i] + config.COLLISION_SIGN * cnp * toa[i]  # MATLAB quirk

            # 1 & 2: Doppler
            Fs = self._doppler(i, fwd)
            Fe = self._doppler(i, fwd + toa[i])
            stat_thr = 0.25 * bw[i]
            dyn_thr = bw[i] / (3 * 2.0 ** sf[i])
            lost = (Fs >= stat_thr) | (np.abs(Fs - Fe) >= dyn_thr)

            # 3: collision vs OTHER same-channel EDs' forward packets
            c = self.channel[i]
            if c in chan_pack:
                eids, iss, iee = chan_pack[c]
                other = eids != i
                if other.any():
                    o_s = iss[other]; o_e = iee[other]
                    surv = np.where(~lost)[0]
                    for k in surv:
                        ts = col[k]; te = ts + toa[i]
                        if np.any((o_s < te) & (o_e > ts)):
                            lost[k] = True

            li = int(lost.sum())
            per_ed_loss[i] = li / n
            total_lost += li
            total_pkts += n

        plr = total_lost / total_pkts if total_pkts else 1.0
        return dict(plr=plr, total_packets=int(total_pkts),
                    per_ed_loss=per_ed_loss, n_packets=npk,
                    combos=combos, timings=timings)

    # ------------------------------------------------------------------
    def states(self):
        d = self.d_closest / self._d_ref
        slot = self.slot / N_COMBOS
        ts = self.t_start / self._t_ref
        txw = self.Txw / self._txw_ref
        load = self.channel_load / self._load_ref
        return np.stack([d, slot, ts, txw, load], axis=1).astype(np.float64)

    def combo_masks(self):
        return self.allowed_mask.copy()

    def random_allocation(self, rng):
        combos = np.array([rng.choice(np.where(self.allowed_mask[i])[0])
                           for i in range(self.N)])
        timings = rng.integers(0, config.N_TIMING, self.N)
        return combos, timings


if __name__ == "__main__":
    for N in [100, 500, 1000]:
        sc = Scenario(N, seed=config.SEED)
        rng = np.random.default_rng(config.SEED)
        c, t = sc.random_allocation(rng)
        r = sc.evaluate_network(c, t)
        print(f"N={N:5d}  random-on-scenario PLR={r['plr']:.4f}  "
              f"packets={r['total_packets']}  slots={sc.slot.min()}..{sc.slot.max()}")
