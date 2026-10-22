"""Quiz environment (plain Python, no TensorFlow needed).

RL formulation
--------------
Agent        : adaptive quiz recommendation agent
Environment  : simulated student + quiz system
State        : [recent_accuracy, estimated_mastery, previous_difficulty]  (all in [0, 1])
Action       : 0 = easy, 1 = medium, 2 = hard   (the quiz system then picks a
               question from that difficulty category in the question bank)
Reward       : correctness + suitable challenge + mastery improvement (see _reward)
Episode      : one quiz session of EPISODE_LENGTH questions
"""
from collections import deque

import numpy as np

from .student import (DIFFICULTY_VALUES, PROFILES, SimulatedStudent, ZPD_HIGH,
                      ZPD_LOW, sigmoid)

EPISODE_LENGTH = 30
WINDOW = 5            # recent accuracy is computed over the last 5 answers
EST_MASTERY_START = 0.30
EST_LEARN = 0.12
MASTERY_BONUS = 4.0   # max reward bonus per question for real mastery improvement (tunable; 2, 4, 6 compared in results/reward_tuning)


class MasteryTracker:
    """What the *agent* can observe about a student.

    Keeps recent accuracy, an estimated mastery (Elo / IRT-style update from the
    student's responses) and the previous difficulty.  The same class is used by
    the simulator and by the Streamlit demo, so the state is identical.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.recent = deque(maxlen=WINDOW)
        self.est_mastery = EST_MASTERY_START
        self.prev_difficulty = 1          # start on "medium"
        self.streak_correct = 0
        self.streak_wrong = 0

    def update(self, difficulty_idx: int, correct: bool):
        d = float(DIFFICULTY_VALUES[difficulty_idx])
        p_pred = sigmoid(8.0 * (self.est_mastery - d + 0.05))
        self.est_mastery = float(np.clip(
            self.est_mastery + EST_LEARN * (float(correct) - p_pred), 0.0, 1.0))
        self.recent.append(1.0 if correct else 0.0)
        self.prev_difficulty = difficulty_idx
        if correct:
            self.streak_correct += 1
            self.streak_wrong = 0
        else:
            self.streak_wrong += 1
            self.streak_correct = 0

    @property
    def recent_accuracy(self) -> float:
        return float(np.mean(self.recent)) if self.recent else 0.5

    def observation(self) -> np.ndarray:
        return np.array([self.recent_accuracy,
                         self.est_mastery,
                         self.prev_difficulty / 2.0], dtype=np.float32)


class QuizEnv:
    """Gym-style environment: reset() -> obs, step(a) -> obs, reward, done, info."""

    n_actions = 3
    obs_dim = 3

    def __init__(self, profile=None, seed=None, episode_length=EPISODE_LENGTH):
        """profile=None samples a random student profile every episode."""
        self.fixed_profile = profile
        self.rng = np.random.default_rng(seed)
        self.episode_length = episode_length
        self.tracker = MasteryTracker()
        self.student = None
        self.t = 0

    def reset(self, profile=None):
        name = profile or self.fixed_profile
        if name is None:
            name = self.rng.choice(list(PROFILES))
        self.profile_name = str(name)
        self.student = SimulatedStudent(PROFILES[self.profile_name], self.rng)
        self.tracker.reset()
        self.t = 0
        return self.tracker.observation()

    # ------------------------------------------------------------------
    def _reward(self, difficulty_idx, correct, est_before, true_gain, streak_wrong):
        """Reward balancing correctness, learning progress and suitable challenge.

        The reward does NOT simply maximise correct answers: a correct answer on
        a question that is too easy earns little, so "always easy" is a bad policy.
        Values are illustrative and tunable.
        """
        d = float(DIFFICULTY_VALUES[difficulty_idx])
        gap = d - est_before                      # >0: harder than current level
        appropriate = ZPD_LOW <= gap <= ZPD_HIGH
        r = 0.0
        if correct:
            if appropriate:
                r += 2.0                          # correct + suitably challenging
            elif gap < ZPD_LOW:
                r += 1.0 if est_before >= 0.7 else 0.0   # correct but too easy
            else:
                r += 1.0                          # correct on a stretch question
        elif gap > 0.40:
            r -= 1.0                              # wrong on a very difficult question
        if streak_wrong >= 2 and gap > 0.20:
            r -= 2.0                              # repeated mistakes beyond level
        r += MASTERY_BONUS * min(1.0, true_gain / 0.03)     # mastery-improvement bonus (simulator signal)
        return float(r)

    def step(self, action: int):
        action = int(action)
        est_before = self.tracker.est_mastery
        true_before = self.student.mastery
        correct, gain = self.student.answer(action)
        self.tracker.update(action, correct)
        reward = self._reward(action, correct, est_before, gain, self.tracker.streak_wrong)
        self.t += 1
        done = self.t >= self.episode_length
        d = float(DIFFICULTY_VALUES[action])
        gap_true = d - true_before
        info = {
            "correct": correct,
            "difficulty": action,
            "true_mastery": self.student.mastery,
            "est_mastery": self.tracker.est_mastery,
            "appropriate": bool(ZPD_LOW <= gap_true <= ZPD_HIGH),
            "profile": self.profile_name,
        }
        return self.tracker.observation(), reward, done, info
