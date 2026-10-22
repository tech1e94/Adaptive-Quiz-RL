"""Recommendation policies: two baselines, a random sanity check, and a trained DQN."""
import numpy as np


class FixedPolicy:
    """Baseline 1: always recommend the same difficulty (default: medium)."""

    def __init__(self, level=1):
        self.level = level
        self.name = "Fixed (" + ["easy", "medium", "hard"][level] + ")"

    def reset(self):
        pass

    def act(self, obs, env):
        return self.level


class RuleBasedPolicy:
    """Baseline 2: simple rule-based adaptation.

    Start on medium. After 3 correct answers in a row go one level harder; after
    2 mistakes in a row go one level easier.
    """

    def __init__(self, up_after=3, down_after=2):
        self.up_after, self.down_after = up_after, down_after
        self.name = "Rule-based"
        self.reset()

    def reset(self):
        self.level = 1
        self.run_right = 0
        self.run_wrong = 0

    def act(self, obs, env):
        if env.t == 0:
            self.reset()
        else:  # look at the outcome of the previous question
            if env.tracker.recent[-1] == 1.0:
                self.run_right += 1
                self.run_wrong = 0
            else:
                self.run_wrong += 1
                self.run_right = 0
        if self.run_right >= self.up_after:
            self.level = min(2, self.level + 1)
            self.run_right = 0
        elif self.run_wrong >= self.down_after:
            self.level = max(0, self.level - 1)
            self.run_wrong = 0
        return self.level


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.name = "Random"

    def reset(self):
        pass

    def act(self, obs, env):
        return int(self.rng.integers(3))


class MLPPolicy:
    """Greedy policy from a trained Q-network stored in a .npz file.

    Both train_numpy.py and train_tfagents.py export the Q-network in this
    format (W0,b0,W1,b1,... ), so evaluation and the Streamlit demo never need
    TensorFlow.
    """

    def __init__(self, path, name="DQN"):
        data = np.load(path)
        n = len([k for k in data.files if k.startswith("W")])
        self.W = [data[f"W{i}"] for i in range(n)]
        self.b = [data[f"b{i}"] for i in range(n)]
        self.name = name

    def reset(self):
        pass

    def q_values(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
        return x

    def act(self, obs, env=None):
        return int(np.argmax(self.q_values(obs)))
