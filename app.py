"""Streamlit demo: a real Python quiz where the trained RL agent picks the difficulty.

    streamlit run app.py
"""
import os
import random
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from quiz_rl.env import MasteryTracker
from quiz_rl.policies import FixedPolicy, MLPPolicy, RuleBasedPolicy
from quiz_rl.question_bank import QUESTION_BANK
from quiz_rl.student import DIFFICULTY_NAMES

st.set_page_config(page_title="Adaptive Quiz (RL)", page_icon="🎯", layout="centered")

POLICY_FILES = ["results/policy_tfagents.npz", "results/policy_numpy.npz"]


def find_policy_file():
    for p in POLICY_FILES:
        if os.path.exists(p):
            return p
    return None


def build_policy(choice):
    if choice == "DQN (trained agent)":
        return MLPPolicy(find_policy_file(), name="DQN")
    if choice == "Rule-based":
        return RuleBasedPolicy()
    return FixedPolicy(1)


def choose_next(ss):
    """Ask the policy for a difficulty, then pick an unused question of that level."""
    obs = ss.tracker.observation()
    level = ss.policy.act(obs, ss.ctx)
    pool = [i for i in range(len(QUESTION_BANK[level])) if (level, i) not in ss.asked]
    if not pool:  # level exhausted: recycle its questions (keeps the agent's choice intact)
        ss.asked = {q for q in ss.asked if q[0] != level}
        pool = list(range(len(QUESTION_BANK[level])))
    idx = random.choice(pool)
    ss.asked.add((level, idx))
    ss.current = (level, idx)


def start_session(policy_choice, n_questions):
    ss = st.session_state
    ss.tracker = MasteryTracker()
    ss.ctx = SimpleNamespace(t=0, tracker=ss.tracker)   # what the rule-based policy looks at
    ss.policy = build_policy(policy_choice)
    ss.policy.reset()
    ss.n_questions = n_questions
    ss.policy_choice = policy_choice
    ss.asked, ss.history, ss.feedback = set(), [], None
    ss.started = True
    choose_next(ss)


def submit_answer(ss, options, answer_idx, choice):
    level, _ = ss.current
    correct = options.index(choice) == answer_idx
    ss.tracker.update(level, correct)
    ss.ctx.t += 1
    ss.history.append(dict(
        question=len(ss.history) + 1,
        difficulty=DIFFICULTY_NAMES[level],
        level=level,
        correct=correct,
        est_mastery=ss.tracker.est_mastery,
        recent_accuracy=ss.tracker.recent_accuracy,
    ))
    ss.feedback = (correct, options[answer_idx])
    if len(ss.history) < ss.n_questions:
        choose_next(ss)


def show_dashboard(ss):
    df = pd.DataFrame(ss.history)
    st.subheader("Progress dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions answered", len(df))
    c2.metric("Accuracy", f"{df.correct.mean():.0%}")
    c3.metric("Estimated mastery", f"{ss.tracker.est_mastery:.2f}")
    st.caption("Estimated mastery and recent accuracy after each question")
    st.line_chart(df.set_index("question")[["est_mastery", "recent_accuracy"]])
    st.caption("Difficulty chosen for each question (0 = easy, 1 = medium, 2 = hard)")
    st.line_chart(df.set_index("question")[["level"]])
    with st.expander("Table view"):
        st.dataframe(df[["question", "difficulty", "correct", "est_mastery", "recent_accuracy"]],
                     hide_index=True)


def main():
    ss = st.session_state
    st.title("🎯 Adaptive Python Quiz")
    st.caption("A reinforcement-learning agent (DQN) chooses the difficulty of each question "
               "from your recent performance. Prototype trained on simulated students.")

    if not ss.get("started"):
        options = ["Rule-based", "Fixed (medium)"]
        if find_policy_file():
            options.insert(0, "DQN (trained agent)")
        else:
            st.info("No trained policy found. Run `python train_numpy.py` (or `train_tfagents.py`) first "
                    "to enable the DQN agent.")
        choice = st.radio("Who chooses the question difficulty?", options)
        n = st.slider("Number of questions", 5, 20, 12)
        if st.button("Start quiz", type="primary"):
            start_session(choice, n)
            st.rerun()
        return

    st.sidebar.markdown(f"**Recommender:** {ss.policy_choice}")
    st.sidebar.metric("Estimated mastery", f"{ss.tracker.est_mastery:.2f}")
    st.sidebar.metric("Recent accuracy (last 5)", f"{ss.tracker.recent_accuracy:.0%}")
    if st.sidebar.button("Restart"):
        ss.started = False
        st.rerun()

    done = len(ss.history) >= ss.n_questions
    if ss.feedback is not None:
        ok, right = ss.feedback
        (st.success if ok else st.error)("Correct!" if ok else f"Not quite. The answer was: {right}")

    if done:
        st.success("Quiz complete!")
        show_dashboard(ss)
        return

    level, idx = ss.current
    text, options, answer_idx = QUESTION_BANK[level][idx]
    n = len(ss.history) + 1
    st.progress((n - 1) / ss.n_questions, text=f"Question {n} of {ss.n_questions}")
    st.markdown(f"**Difficulty chosen by the agent: {DIFFICULTY_NAMES[level]}**")
    st.markdown(text.replace("\n", "  \n"))
    with st.form(key=f"form_{n}"):
        choice = st.radio("Your answer", options, index=None)
        submitted = st.form_submit_button("Submit answer")
    if submitted:
        if choice is None:
            st.warning("Pick an answer first.")
        else:
            submit_answer(ss, options, answer_idx, choice)
            st.rerun()
    if ss.history:
        show_dashboard(ss)


main()
