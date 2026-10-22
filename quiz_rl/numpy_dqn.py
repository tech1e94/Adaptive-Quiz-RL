"""A small DQN written in plain NumPy.

Same algorithm as the TF-Agents version (replay buffer, epsilon-greedy
exploration, target network, TD loss): it exists so the project can be trained
and tested on any machine without TensorFlow.  Exports the Q-network as .npz
in the format read by quiz_rl.policies.MLPPolicy.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.W.append((rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))

    def forward(self, x, cache=False):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            acts.append(x)
        return (x, acts) if cache else x

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def save(self, path):
        arrays = {}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(path, **arrays)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng):
        self.cap, self.rng = capacity, rng
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.n = 0
        self.i = 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.d[self.i] = s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class NumpyDQN:
    def __init__(self, obs_dim, n_actions, hidden=(64, 64), lr=1e-3, gamma=0.95, seed=0):
        self.rng = np.random.default_rng(seed)
        sizes = [obs_dim, *hidden, n_actions]
        self.q = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.q)
        self.opt = Adam([*self.q.W, *self.q.b], lr=lr)
        self.gamma, self.n_actions = gamma, n_actions

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q.forward(obs[None, :])[0]))

    def train_step(self, batch):
        s, a, r, s2, d = batch
        B = len(s)
        q_next = self.target.forward(s2).max(axis=1)
        y = r + self.gamma * (1.0 - d) * q_next
        q, acts = self.q.forward(s, cache=True)
        err = q[np.arange(B), a] - y
        loss = float(np.mean(np.where(np.abs(err) < 1, 0.5 * err ** 2, np.abs(err) - 0.5)))  # Huber
        dq = np.zeros_like(q)
        dq[np.arange(B), a] = np.clip(err, -1.0, 1.0) / B
        gW, gb = [None] * len(self.q.W), [None] * len(self.q.W)
        delta = dq
        for i in reversed(range(len(self.q.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.q.W[i].T) * (acts[i] > 0)
        self.opt.step([*gW, *gb])
        return loss
