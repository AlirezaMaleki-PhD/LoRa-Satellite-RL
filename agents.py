"""
agents.py
=========

Self-contained PPO and DQN agents implemented in pure NumPy (no PyTorch /
TensorFlow needed). The networks are small one-hidden-layer MLPs with manual
forward/backward passes and an Adam optimiser.

Both agents solve the same decision: given an ED's state, choose an action
(combo x timing) subject to an action MASK that enforces the distance
constraint (an ED in distance slot i may only use combo index >= i).

Because each ED's reward depends only on its own action (the collision surface
is a fixed lookup), the problem is a *contextual bandit*: every step is a
one-step terminal episode. PPO and DQN both handle this cleanly; there is no
bootstrapping across EDs.
"""

import numpy as np


# ----------------------------------------------------------------------
# Small MLP with Adam
# ----------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        # He-ish init
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.params = ["W1", "b1", "W2", "b2"]
        # Adam state
        self.lr = lr
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._t = 0

    def forward(self, x):
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.tanh(self._z1)
        out = self._a1 @ self.W2 + self.b2
        return out

    def backward(self, d_out):
        """Given gradient wrt the output, return parameter gradients."""
        dW2 = self._a1.T @ d_out
        db2 = d_out.sum(0)
        da1 = d_out @ self.W2.T
        dz1 = da1 * (1.0 - self._a1 ** 2)
        dW1 = self._x.T @ dz1
        db1 = dz1.sum(0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    def adam_step(self, grads, clip=5.0):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            g = grads[p]
            gn = np.linalg.norm(g)
            if gn > clip:                       # gradient clipping
                g = g * (clip / (gn + 1e-12))
            self._m[p] = b1 * self._m[p] + (1 - b1) * g
            self._v[p] = b2 * self._v[p] + (1 - b2) * (g * g)
            mhat = self._m[p] / (1 - b1 ** self._t)
            vhat = self._v[p] / (1 - b2 ** self._t)
            setattr(self, p, getattr(self, p) - self.lr * mhat / (np.sqrt(vhat) + eps))


# ----------------------------------------------------------------------
# Masked softmax helpers
# ----------------------------------------------------------------------
NEG = -1e30


def masked_softmax(logits, mask):
    z = np.where(mask, logits, NEG)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    e = np.where(mask, e, 0.0)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


# ======================================================================
# PPO
# ======================================================================
class PPOAgent:
    def __init__(self, n_state, n_actions, hidden=64, lr=3e-3, gamma=0.95,
                 clip=0.2, epochs=4, entropy_coef=0.01, value_coef=0.5, seed=0):
        self.pi = MLP(n_state, hidden, n_actions, lr=lr, seed=seed)
        self.vf = MLP(n_state, hidden, 1, lr=lr, seed=seed + 1)
        self.n_actions = n_actions
        self.clip = clip
        self.epochs = epochs
        self.ent_c = entropy_coef
        self.val_c = value_coef
        self.rng = np.random.default_rng(seed)

    def act_sample(self, state, mask):
        logits = self.pi.forward(state[None, :])[0]
        p = masked_softmax(logits, mask)
        a = self.rng.choice(self.n_actions, p=p)
        logp = np.log(p[a] + 1e-12)
        v = self.vf.forward(state[None, :])[0, 0]
        return a, logp, v

    def act_greedy(self, states, masks):
        logits = self.pi.forward(states)
        p = masked_softmax(logits, masks)
        return p.argmax(axis=1)

    def update(self, S, A, OLDLP, RET, MASK):
        """One PPO update over a batch (S states, A actions, ...)."""
        A = A.astype(int)
        RET = RET.astype(np.float64)
        # advantage = return - value baseline (single-step / bandit)
        V = self.vf.forward(S)[:, 0]
        adv = RET - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = len(A)
        idx = np.arange(B)

        for _ in range(self.epochs):
            # ---- policy ----
            logits = self.pi.forward(S)
            p = masked_softmax(logits, MASK)
            logp_a = np.log(p[idx, A] + 1e-12)
            ratio = np.exp(logp_a - OLDLP)

            unclipped = ratio * adv
            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
            # per-sample d(objective)/d(logp_a)
            use_unclipped = unclipped <= clipped
            in_band = (ratio > 1 - self.clip) & (ratio < 1 + self.clip)
            dobj_dlogp = np.where(use_unclipped | in_band, adv * ratio, 0.0)
            dloss_dlogp = -dobj_dlogp / B          # minimise -objective

            # d logp_a / d logits_j = 1[j==a] - p_j
            d_logits = -p * dloss_dlogp[:, None]
            d_logits[idx, A] += dloss_dlogp

            # entropy bonus: minimise -ent_c * H  ->  grad = ent_c * p*(logp + H)
            logp_all = np.log(p + 1e-12)
            H = -(p * logp_all).sum(axis=1, keepdims=True)
            d_logits += self.ent_c * p * (logp_all + H) / B

            self.pi.adam_step(self.pi.backward(d_logits))

            # ---- value ----
            V = self.vf.forward(S)[:, 0]
            d_v = (self.val_c * 2.0 * (V - RET) / B)[:, None]
            self.vf.adam_step(self.vf.backward(d_v))


# ======================================================================
# DQN
# ======================================================================
class ReplayBuffer:
    def __init__(self, cap, n_state):
        self.cap = cap
        self.S = np.zeros((cap, n_state), np.float32)
        self.A = np.zeros(cap, np.int64)
        self.R = np.zeros(cap, np.float32)
        self.M = np.zeros((cap, 0))          # masks stored lazily below
        self._mask = None
        self.i = 0
        self.full = False

    def push(self, s, a, r, mask):
        if self._mask is None:
            self._mask = np.zeros((self.cap, len(mask)), bool)
        self.S[self.i] = s
        self.A[self.i] = a
        self.R[self.i] = r
        self._mask[self.i] = mask
        self.i = (self.i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, batch, rng):
        n = len(self)
        j = rng.integers(0, n, size=batch)
        return self.S[j], self.A[j], self.R[j], self._mask[j]


class DQNAgent:
    def __init__(self, n_state, n_actions, hidden=64, lr=1e-3, gamma=0.95,
                 buffer_size=20000, batch_size=128, seed=0):
        self.q = MLP(n_state, hidden, n_actions, lr=lr, seed=seed)
        self.n_actions = n_actions
        self.batch = batch_size
        self.buf = ReplayBuffer(buffer_size, n_state)
        self.rng = np.random.default_rng(seed)

    def act_eps(self, state, mask, eps):
        if self.rng.random() < eps:
            valid = np.where(mask)[0]
            return int(self.rng.choice(valid))
        q = self.q.forward(state[None, :])[0]
        q = np.where(mask, q, NEG)
        return int(q.argmax())

    def act_greedy(self, states, masks):
        q = self.q.forward(states)
        q = np.where(masks, q, NEG)
        return q.argmax(axis=1)

    def push(self, s, a, r, mask):
        self.buf.push(s, a, r, mask)

    def train_step(self):
        if len(self.buf) < self.batch:
            return
        S, A, R, _ = self.buf.sample(self.batch, self.rng)
        A = A.astype(int)
        # contextual bandit: terminal target = reward (no bootstrap)
        Q = self.q.forward(S)
        target = R.astype(np.float64)
        idx = np.arange(len(A))
        d_out = np.zeros_like(Q)
        d_out[idx, A] = 2.0 * (Q[idx, A] - target) / len(A)   # MSE grad on taken action
        self.q.adam_step(self.q.backward(d_out))
