"""Train the DQN with the NumPy implementation (no TensorFlow required).

    python train_numpy.py --episodes 3000

Outputs: results/policy_numpy.npz and results/training_log_numpy.csv
"""
import argparse
import csv
import os

import numpy as np

from quiz_rl.env import QuizEnv
from quiz_rl.numpy_dqn import NumpyDQN, ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = QuizEnv(profile=None, seed=args.seed)  # random student profile each episode
    agent = NumpyDQN(env.obs_dim, env.n_actions, seed=args.seed)
    buf = ReplayBuffer(50_000, env.obs_dim, np.random.default_rng(args.seed + 1))

    steps_total = args.episodes * env.episode_length
    eps_decay_steps = int(0.6 * steps_total)
    step, log = 0, []

    for ep in range(1, args.episodes + 1):
        obs, ret, done = env.reset(), 0.0, False
        while not done:
            eps = max(0.05, 1.0 - 0.95 * step / eps_decay_steps)
            a = agent.act(obs, eps)
            obs2, r, done, _ = env.step(a)
            buf.add(obs, a, r, obs2, float(done))
            obs, ret, step = obs2, ret + r, step + 1
            if buf.n >= 1000:
                agent.train_step(buf.sample(64))
            if step % 500 == 0:
                agent.target.copy_from(agent.q)
        log.append((ep, ret, env.profile_name))
        if ep % 250 == 0:
            recent = np.mean([x[1] for x in log[-250:]])
            print(f"episode {ep:5d}  avg return (last 250, with exploration) = {recent:6.2f}  eps = {eps:.2f}")

    agent.q.save(os.path.join(args.out, "policy_numpy.npz"))
    with open(os.path.join(args.out, "training_log_numpy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "profile"])
        w.writerows(log)
    print("saved", os.path.join(args.out, "policy_numpy.npz"))


if __name__ == "__main__":
    main()
