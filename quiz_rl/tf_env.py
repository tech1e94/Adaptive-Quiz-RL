"""TF-Agents wrapper around QuizEnv (needs tensorflow + tf-agents).

PyEnvironment  ->  tf_py_environment.TFPyEnvironment  ->  DqnAgent
"""
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .env import QuizEnv


class QuizPyEnv(py_environment.PyEnvironment):
    def __init__(self, profile=None, seed=None):
        super().__init__()
        self._env = QuizEnv(profile=profile, seed=seed)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name="action")      # easy / medium / hard
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0.0, maximum=1.0, name="observation")
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        return ts.restart(self._env.reset())

    def _step(self, action):
        if self._episode_ended:                 # previous step ended the episode
            return self.reset()
        obs, reward, done, _ = self._env.step(int(action))
        reward = np.float32(reward)
        if done:
            self._episode_ended = True
            return ts.termination(obs, reward)
        return ts.transition(obs, reward, discount=1.0)
