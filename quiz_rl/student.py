"""Simulated student used to generate quiz responses during training.

The student has a hidden *true mastery* in [0, 1].  Questions have a difficulty
value (easy=0.2, medium=0.5, hard=0.8).

* P(correct) rises when mastery is above the question difficulty (with a small
  guess / slip probability, like a multiple-choice quiz).
* Learning is strongest when the question is a little ABOVE the current mastery
  (a "zone of proximal development").  Questions that are far too easy or far
  too hard teach very little.  This is what gives the RL agent something real
  to learn: neither "always easy" nor "always hard" is a good policy.

This is a prototype model.  It does not prove real-world learning gains.
"""
from dataclasses import dataclass

import numpy as np

DIFFICULTY_VALUES = np.array([0.2, 0.5, 0.8], dtype=np.float32)  # easy, medium, hard
DIFFICULTY_NAMES = ["easy", "medium", "hard"]

# Zone of proximal development, relative to mastery (shared by reward + metrics)
ZPD_LOW = -0.15
ZPD_HIGH = 0.35


@dataclass(frozen=True)
class Profile:
    name: str
    mastery0: float      # starting true mastery
    learn_rate: float    # how fast this student learns


PROFILES = {
    "struggling": Profile("struggling", mastery0=0.10, learn_rate=0.030),
    "average": Profile("average", mastery0=0.30, learn_rate=0.040),
    "advanced": Profile("advanced", mastery0=0.55, learn_rate=0.050),
}

GUESS = 0.20   # chance of answering correctly by guessing
SLIP = 0.05    # chance of a careless mistake even when the topic is known


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SimulatedStudent:
    def __init__(self, profile: Profile, rng: np.random.Generator):
        self.profile = profile
        self.rng = rng
        jitter = rng.normal(0.0, 0.04)
        self.mastery = float(np.clip(profile.mastery0 + jitter, 0.02, 0.95))
        self.initial_mastery = self.mastery

    def p_correct(self, difficulty_idx: int) -> float:
        d = float(DIFFICULTY_VALUES[difficulty_idx])
        core = sigmoid(8.0 * (self.mastery - d + 0.05))
        return GUESS + (1.0 - GUESS - SLIP) * core

    def answer(self, difficulty_idx: int):
        """Answer one question. Returns (correct, mastery_gain)."""
        correct = bool(self.rng.random() < self.p_correct(difficulty_idx))
        d = float(DIFFICULTY_VALUES[difficulty_idx])
        centre = self.mastery + 0.15
        zone = np.exp(-((d - centre) ** 2) / (2 * 0.20 ** 2))
        gain = self.profile.learn_rate * zone * (1.0 if correct else 0.5)
        gain *= (1.0 - self.mastery)  # diminishing returns near full mastery
        self.mastery = float(min(1.0, self.mastery + gain))
        return correct, float(gain)
