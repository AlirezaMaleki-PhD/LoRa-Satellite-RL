"""
train.py  (coupled / overlap-collision version)
===============================================

Trains PPO and DQN on the coupled LoRa allocation problem and compares them to
the random baseline, all scored with the SAME overlap-collision model as
PLR_Tot_sim.m (see net_scenario.py). Channel is random; the agent controls
(SF, CR, BW) via a combo index plus the burst timing, subject to the distance
constraint.

Outputs:
    results/results_N{N}.npz   per-N arrays (combos, slots, distances, curves)
    results/summary.json       scalar PLR / packet totals for every method & N

Run:
    python train.py
    python train.py --sizes 100 500
    python train.py --iters 30
"""

import os, json, argparse
import numpy as np

import config
from net_env import NetEnv
from net_scenario import COMBO_BW
from agents import PPOAgent, DQNAgent, masked_softmax


def _sample_from_probs(P, rng):
    c = np.cumsum(P, axis=1)
    r = rng.random((P.shape[0], 1))
    return (r < c).argmax(axis=1)


def greedy_plr(agent, env):
    S, M = env.all_states(), env.all_masks()
    A = agent.act_greedy(S, M)
    return env.evaluate_report(A)          # Rule A: overlap counting


# fixed set of networks used only to trace the learning curve (smoother than a
# single scenario); kept separate from the held-out evaluation seeds.
CURVE_SEEDS = [config.SEED + 200 + k for k in range(3)]
_CURVE_ENVS = {}

def curve_plr(agent, N):
    envs = _CURVE_ENVS.setdefault(N, [NetEnv(N, seed=s) for s in CURVE_SEEDS])
    return float(np.mean([greedy_plr(agent, e)["plr"] for e in envs]))


def train_reward(agent, env):
    """Mean per-ED reward (success prob. under the REWARD_MODE model) of the
    greedy policy on the training scenario -- i.e. the objective being maximized."""
    S, M = env.all_states(), env.all_masks()
    A = agent.act_greedy(S, M)
    R, _ = env.rewards(A)
    return float(R.mean())


def _mean(dicts, key):
    return float(np.mean([d[key] for d in dicts]))


def eval_agent_multi(agent, N, seeds):
    plr, pk = [], []
    for s in seeds:
        e = NetEnv(N, seed=s)
        r = greedy_plr(agent, e)
        plr.append(r["plr"]); pk.append(r["total_packets"])
    return float(np.mean(plr)), int(np.mean(pk))


def eval_random_multi(N, seeds, kind):
    plr, pk = [], []
    for s in seeds:
        e = NetEnv(N, seed=s)
        r = matlab_like_random(e, s) if kind == "matlab" else scenario_random(e, s)
        plr.append(r["plr"]); pk.append(r["total_packets"])
    return float(np.mean(plr)), int(np.mean(pk))


# ----------------------------------------------------------------------
def matlab_like_random(env, seed):
    """Random allocation matching PLR_Tot_sim.m: fixed BW=125 kHz, random SF/CR,
    NO distance constraint, burst at t_start. Reproduces your reported baseline."""
    rng = np.random.default_rng(seed)
    bw125 = np.where(COMBO_BW == 125_000)[0]
    combos = rng.choice(bw125, size=env.N)
    timings = np.zeros(env.N, dtype=int)
    return env.sc.evaluate_network(combos, timings)


def scenario_random(env, seed):
    """Random over the agent's OWN action space (allowed combos + random timing),
    scored with the ACTIVE reward model (database or overlap)."""
    rng = np.random.default_rng(seed)
    combos, timings = env.sc.random_allocation(rng)
    actions = combos * config.N_TIMING + timings
    return env.evaluate_actions(actions)


# ----------------------------------------------------------------------
def train_ppo(env, hp, seed):
    rng = np.random.default_rng(seed)
    agent = PPOAgent(env.n_state, env.n_actions, hidden=hp["hidden"], lr=hp["lr"],
                     gamma=hp["gamma"], clip=hp["clip"], epochs=hp["epochs"],
                     entropy_coef=hp["entropy_coef"], value_coef=hp["value_coef"],
                     seed=seed)
    S, M = env.all_states(), env.all_masks()
    curve = []
    rcurve = []
    for it in range(hp["n_iterations"]):
        logits = agent.pi.forward(S)
        P = masked_softmax(logits, M)
        A = _sample_from_probs(P, rng)
        OLDLP = np.log(P[np.arange(env.N), A] + 1e-12)
        R, _ = env.rewards(A)
        agent.update(S, A, OLDLP, R, M)
        rcurve.append((it, train_reward(agent, env)))
        if it % 2 == 0 or it == hp["n_iterations"] - 1:
            curve.append((it, curve_plr(agent, env.N)))
    return agent, np.array(curve), np.array(rcurve)


def train_dqn(env, hp, seed):
    rng = np.random.default_rng(seed + 7)
    agent = DQNAgent(env.n_state, env.n_actions, hidden=hp["hidden"], lr=hp["lr"],
                     gamma=hp["gamma"], buffer_size=hp["buffer_size"],
                     batch_size=hp["batch_size"], seed=seed)
    S, M = env.all_states(), env.all_masks()
    curve = []
    rcurve = []
    for it in range(hp["n_iterations"]):
        frac = min(1.0, it / max(hp["eps_decay_iters"], 1))
        eps = hp["eps_start"] + frac * (hp["eps_end"] - hp["eps_start"])
        Q = agent.q.forward(S)
        Qm = np.where(M, Q, -1e30)
        A = Qm.argmax(axis=1)
        rmask = rng.random(env.N) < eps
        for i in np.where(rmask)[0]:
            valid = np.where(M[i])[0]
            A[i] = rng.choice(valid)
        R, _ = env.rewards(A)
        for i in range(env.N):
            agent.push(S[i], int(A[i]), R[i], M[i])
        for _ in range(hp["grad_steps_per_iter"]):
            agent.train_step()
        rcurve.append((it, train_reward(agent, env)))
        if it % 2 == 0 or it == hp["n_iterations"] - 1:
            curve.append((it, curve_plr(agent, env.N)))
    return agent, np.array(curve), np.array(rcurve)


# ----------------------------------------------------------------------
def run_for_N(N, iters_override=None):
    print(f"\n================  N = {N}  ================")
    env = NetEnv(N, seed=config.SEED)

    ppo_hp, dqn_hp = dict(config.PPO), dict(config.DQN)
    if iters_override:
        ppo_hp["n_iterations"] = dqn_hp["n_iterations"] = iters_override

    eval_seeds = [config.SEED + k for k in range(config.N_EVAL_SCENARIOS)]

    ppo_agent, ppo_curve, ppo_rcurve = train_ppo(env, ppo_hp, config.SEED)
    dqn_agent, dqn_curve, dqn_rcurve = train_dqn(env, dqn_hp, config.SEED)

    # Reporting uses EVAL_MODE (overlap / Rule A). The random baseline is the
    # RANDOM ALLOCATION you defined (random SF/CR, transmit immediately),
    # measured by real overlap counting -> reproduces your reported 0.029-0.062.
    r_plr, r_pk = eval_random_multi(N, eval_seeds, "matlab")
    p_plr, p_pk = eval_agent_multi(ppo_agent, N, eval_seeds)
    d_plr, d_pk = eval_agent_multi(dqn_agent, N, eval_seeds)
    print(f"  RANDOM (overlap/Rule A)  PLR={r_plr:.4f}  packets={r_pk}")
    print(f"  PPO  (train={config.REWARD_MODE}, eval={config.EVAL_MODE})  PLR={p_plr:.4f}  packets={p_pk}")
    print(f"  DQN  (train={config.REWARD_MODE}, eval={config.EVAL_MODE})  PLR={d_plr:.4f}  packets={d_pk}")

    # arrays from the PRIMARY scenario (for the action / scatter figures)
    ppo0 = greedy_plr(ppo_agent, env)
    dqn0 = greedy_plr(dqn_agent, env)
    scen0 = scenario_random(env, config.SEED)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    np.savez(
        os.path.join(config.RESULTS_DIR, f"results_N{N}.npz"),
        slot=env.sc.slot, d_closest=env.sc.d_closest, d_max=env.sc.d_max,
        ppo_combos=ppo0["combos"], ppo_timings=ppo0["timings"],
        dqn_combos=dqn0["combos"], dqn_timings=dqn0["timings"],
        scen_combos=scen0["combos"], scen_timings=scen0["timings"],
        ppo_curve=ppo_curve, dqn_curve=dqn_curve,
        ppo_rcurve=ppo_rcurve, dqn_rcurve=dqn_rcurve,
    )
    return {
        "N": N,
        "reward_mode": config.REWARD_MODE,
        "plr_random": r_plr,
        "plr_random_reported": float(config.RANDOM_PLR_BASELINE.get(N, np.nan)),
        "plr_ppo": p_plr,
        "plr_dqn": d_plr,
        "pkts_random": r_pk,
        "pkts_ppo": p_pk,
        "pkts_dqn": d_pk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=config.N_LIST)
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args()

    summary = [run_for_N(N, args.iters) for N in args.sizes]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved", os.path.join(config.RESULTS_DIR, "summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
