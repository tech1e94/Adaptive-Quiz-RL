# Adaptive Learning and Quiz Recommendation Agent (Reinforcement Learning)

A DQN agent learns which question difficulty (easy / medium / hard) to give a student next,
based on how the student has been doing. It is trained on simulated students and compared
with a fixed-difficulty baseline and a rule-based baseline. A Streamlit app lets you take a
real Python quiz where the trained agent picks the difficulty.

## Quick start

```bash
pip install numpy matplotlib pandas streamlit      # enough for everything except TF-Agents

# Option A - TF-Agents DQN (your syllabus version). Use Python 3.9-3.11.
pip install tensorflow==2.15.0 tf-agents==0.19.0
python train_tfagents.py --episodes 3000
python evaluate.py --policy results/policy_tfagents.npz --log results/training_log_tfagents.csv --out results_tfagents

# Option B - NumPy DQN (same algorithm, no TensorFlow). Runs in ~35 s.
python train_numpy.py --episodes 3000
python evaluate.py

# Demo (uses results/policy_tfagents.npz if present, otherwise results/policy_numpy.npz)
streamlit run app.py
```

> **Important:** `train_numpy.py`, `evaluate.py` and the app logic were run and tested while
> building this. `train_tfagents.py` and `quiz_rl/tf_env.py` were written against the TF-Agents
> 0.19 API but **could not be run in my build environment** (no access to install TensorFlow).
> Run it once on your machine. It ends with a self-check that its exported weights reproduce
> the TF-Agents Q-network. If it raises an error, send me the traceback and I'll fix it.
> The results below come from the NumPy DQN.

## Files

| File | Purpose |
|---|---|
| `quiz_rl/student.py` | Simulated student (3 profiles: struggling, average, advanced) |
| `quiz_rl/env.py` | RL environment, state tracker, reward function |
| `quiz_rl/tf_env.py` | TF-Agents `PyEnvironment` wrapper |
| `quiz_rl/policies.py` | Fixed and rule-based baselines, loader for trained policy |
| `quiz_rl/numpy_dqn.py` | DQN in plain NumPy |
| `quiz_rl/question_bank.py` | 24 Python questions (8 per level) for the demo |
| `train_tfagents.py` | **TF-Agents DQN training** (QNetwork, DqnAgent, replay buffer, driver) |
| `train_numpy.py` | NumPy DQN training |
| `evaluate.py` | Baseline comparison, tables and plots |
| `app.py` | Streamlit quiz and progress dashboard |
| `results/` | Policy, training log, tables and figures from my run |

## RL formulation (for your viva)

| Component | Implementation |
|---|---|
| Agent | Adaptive quiz recommendation agent (DQN, 2 hidden layers of 64) |
| Environment | Simulated student + quiz system |
| State | `[recent_accuracy (last 5), estimated_mastery, previous_difficulty]`, all scaled to [0, 1] |
| Action | 0 = easy, 1 = medium, 2 = hard (the quiz picks a question inside that level) |
| Reward | correctness on a suitably hard question, penalties for overshooting, bonus for real mastery gain |
| Episode | One quiz session of 30 questions, with a random student profile each time |
| Goal | A policy that maximises cumulative reward |

**Simulated student.** Hidden true mastery in [0, 1]. The chance of a correct answer rises
when mastery is above the question's difficulty (with guess = 20% and slip = 5%). Learning is
largest when the question is slightly *above* the student's level, and small when it is far too
easy or far too hard. That is why neither "always easy" nor "always hard" is optimal.

**Estimated mastery.** The agent never sees true mastery. It sees an Elo-style estimate that
is updated from the student's answers.

**Reward** (per question, see `QuizEnv._reward`):

| Outcome | Reward |
|---|---|
| Correct, difficulty suitably above current level | +2 |
| Correct on a stretch question (harder than suitable) | +1 |
| Correct on an easy question when mastery is already high | +1 (otherwise 0) |
| Wrong on a very difficult question | -1 |
| Repeated mistakes on questions beyond the student's level | -2 |
| Mastery improvement (true mastery gain, capped) | up to +4 (`MASTERY_BONUS`) |

The mastery bonus uses the simulator's true mastery, which is only available in simulation.
It is a training-time signal, not something a real deployment would have.

**Algorithm.** DQN: epsilon-greedy exploration (1.0 to 0.05), replay buffer of 50,000
transitions, target network updated every 500 steps, Huber TD loss, Adam (lr 1e-3), gamma 0.95.

## Results (NumPy DQN, 300 sessions per profile, mean +/- 95% CI)

Every method faced the same sequence of simulated students.

| Metric | Method | Struggling | Average | Advanced |
|---|---|---|---|---|
| Cumulative reward | Fixed (medium) | 51.9 +/- 2.3 | 76.5 +/- 1.0 | 58.0 +/- 0.7 |
| | Rule-based | 72.1 +/- 0.7 | 70.9 +/- 0.9 | 65.0 +/- 0.9 |
| | **DQN** | **83.2 +/- 0.9** | **82.8 +/- 0.7** | **69.8 +/- 0.8** |
| Mastery improvement | Fixed (medium) | 0.341 | 0.341 | 0.167 |
| | Rule-based | 0.336 | 0.316 | **0.235** |
| | DQN | **0.370** | 0.331 | 0.211 |
| Difficulty appropriateness | Fixed (medium) | 0.753 | **0.993** | 0.419 |
| | Rule-based | 0.771 | 0.771 | **0.734** |
| | DQN | **0.915** | 0.902 | 0.673 |

Full tables: `results/summary.md`. Figures: `results/comparison.png`,
`results/training_curve.png`, `results/session_trace.png`.

**What the results actually show (report this honestly):**

- DQN gets the highest cumulative reward for every profile. That is the objective it was trained on, so it is the least surprising result.
- On the more meaningful metric, mastery improvement, the picture is mixed. DQN is best for struggling students, roughly tied with fixed-medium for average students (fixed-medium is slightly higher), and **the rule-based baseline is better for advanced students**.
- Fixed-medium is nearly perfect for the average profile, because medium happens to match that profile. It fails for struggling and advanced students, which is the argument for adaptation.
- Qualitatively, the agent starts easy, moves to medium once the student succeeds, and moves advanced students to hard (see `session_trace.png`). It is a bit jumpy in the first ten questions.
- Reward tuning: I compared a mastery bonus of 2, 4 and 6 (`results/reward_tuning/`). Larger bonuses raised DQN's mastery improvement (e.g. struggling 0.355, 0.370, 0.381). I kept 4 as a compromise. Tuning the reward on the same simulator used for evaluation flatters the results, so say that in the viva.
- All of this is on a simulator I designed. It shows the method works in that simulator, not that it improves real students' learning.

## Scope and limitations

- Single subject (Python), small hand-written question bank, three difficulty levels.
- Simulated students only; the DQN is trained offline and the trained policy is used in the demo.
- The demo's estimate of mastery uses the same tracker as training, but real students differ from the simulator, so the agent's choices in the demo are not validated.
- Weaknesses to be ready for: reward design is hand-made and the agent is sensitive to it; the evaluation uses a simulator written by the same person who tuned the reward; single training seed.

## Viva cheat-sheet

- *Why RL and not rules?* The rules (3 right in a row means harder) ignore a student's mastery level and can't be tuned automatically. RL learns the trade-off between correctness, challenge and learning.
- *Why DQN?* Small discrete action space (3), state is a small vector, and we need long-term value of a choice (a harder question now can pay off later).
- *Why not reward correct answers only?* The agent would learn to give only easy questions.
- *What does the replay buffer do?* It breaks the correlation between consecutive samples and reuses experience.
- *What does the target network do?* It makes the TD target stable.
- *What would you do next?* Multiple seeds, a richer student model (forgetting, topics), real student data, offline evaluation.
