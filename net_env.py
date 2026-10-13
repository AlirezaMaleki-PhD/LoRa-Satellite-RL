"""
net_env.py
==========
Thin RL wrapper around net_scenario.Scenario for the coupled (overlap-collision)
problem. The agent chooses, for every ED, a discrete action encoding

        action = combo * N_TIMING + timing

subject to the distance constraint (combo index >= the ED's slot). Channel is
random and fixed by the scenario (not an action).

Reward for an ED = 1 - its packet-loss fraction under the CURRENT joint
allocation of all EDs (so collisions couple the agents, exactly as in the
overlap simulator).
"""

import numpy as np
import config
from net_scenario import Scenario, N_COMBOS


class NetEnv:
    def __init__(self, N, seed=0):
        self.sc = Scenario(N, seed=seed)
        self.N = N
        self.n_combos = N_COMBOS
        self.n_timing = config.N_TIMING
        self.n_actions = N_COMBOS * config.N_TIMING
        self._states = self.sc.states()
        self.n_state = self._states.shape[1]
        # action mask: repeat each combo's allowed-flag across the timing slots
        cmask = self.sc.combo_masks()                      # (N, N_COMBOS)
        self._amask = np.repeat(cmask, config.N_TIMING, axis=1)   # (N, n_actions)

    # ---- static per-ED info ----
    def all_states(self):
        return self._states

    def all_masks(self):
        return self._amask

    def decode(self, action):
        return action // config.N_TIMING, action % config.N_TIMING

    def decode_all(self, actions):
        actions = np.asarray(actions, int)
        combos = actions // config.N_TIMING
        timings = actions % config.N_TIMING
        # safety clamp to the allowed region
        combos = np.maximum(combos, self.sc.slot)
        return combos, timings

    # ---- evaluation / reward ----
    def _eval(self, actions, mode):
        combos, timings = self.decode_all(actions)
        if mode == "database":
            import net_db
            return net_db.evaluate_network_db(self.sc, combos, timings, N=self.N)
        return self.sc.evaluate_network(combos, timings)

    def evaluate_actions(self, actions):
        """Reward model (what the agent TRAINS on): REWARD_MODE."""
        return self._eval(actions, config.REWARD_MODE)

    def evaluate_report(self, actions):
        """Evaluation model (how PLR / packets are MEASURED and reported):
        EVAL_MODE. With EVAL_MODE='overlap' this is Rule A - real packet counting
        via the same-channel overlap simulator."""
        return self._eval(actions, config.EVAL_MODE)

    def rewards(self, actions):
        res = self.evaluate_actions(actions)          # REWARD_MODE (q)
        return 1.0 - res["per_ed_loss"], res
