"""Evaluate the trained DQN against the two baselines on simulated students.

    python evaluate.py --policy results/policy_numpy.npz --log results/training_log_numpy.csv
    python evaluate.py --policy results/policy_tfagents.npz --log results/training_log_tfagents.csv

Every method faces the same sequence of simulated students (same starting
mastery per episode), for every profile, over many repeated sessions.

Outputs (in --out): eval_results.csv, summary.md, training_curve.png,
comparison.png, session_trace.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quiz_rl.env import QuizEnv
from quiz_rl.policies import FixedPolicy, MLPPolicy, RuleBasedPolicy
from quiz_rl.student import DIFFICULTY_NAMES, PROFILES

PROFILE_ORDER = ["struggling", "average", "advanced"]
# Fixed colour per method (colour-blind-safe, Okabe-Ito) + hatch so identity
# never depends on colour alone.
STYLE = {
    "Fixed (medium)": dict(color="#0072B2", hatch=""),
    "Rule-based": dict(color="#E69F00", hatch="//"),
    "DQN": dict(color="#009E73", hatch="xx"),
}
INK, MUTED = "#222222", "#666666"


def run_episode(policy, env, profile, record_trace=False):
    obs = env.reset(profile)
    policy.reset()
    m0 = env.student.mastery
    total, correct, appropriate, n = 0.0, 0, 0, 0
    trace = []
    done = False
    while not done:
        a = policy.act(obs, env)
        obs, r, done, info = env.step(a)
        total += r
        correct += info["correct"]
        appropriate += info["appropriate"]
        n += 1
        if record_trace:
            trace.append((a, info["true_mastery"], info["correct"]))
    return dict(reward=total, accuracy=correct / n, mastery_gain=env.student.mastery - m0,
                appropriateness=appropriate / n), trace


def evaluate(policies, n_episodes, base_seed=12345):
    rows = []
    for profile in PROFILE_ORDER:
        for pol in policies:
            for ep in range(n_episodes):
                env = QuizEnv(seed=base_seed + ep)  # same student draw for every method
                res, _ = run_episode(pol, env, profile)
                rows.append(dict(method=pol.name, profile=profile, episode=ep, **res))
    return rows


def summarise(rows, metric, method, profile):
    v = np.array([r[metric] for r in rows if r["method"] == method and r["profile"] == profile])
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(rows, methods, path, n_episodes):
    metrics = [("reward", "Cumulative reward per session"),
               ("accuracy", "Quiz accuracy"),
               ("mastery_gain", "Mastery improvement (true, start to end)"),
               ("appropriateness", "Difficulty appropriateness (share of questions in the right zone)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    width = 0.26
    x = np.arange(len(PROFILE_ORDER))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for i, m in enumerate(methods):
            stats = [summarise(rows, metric, m, p) for p in PROFILE_ORDER]
            ax.bar(x + (i - 1) * width, [s[0] for s in stats], width * 0.92,
                   yerr=[s[1] for s in stats], capsize=3, label=m,
                   color=STYLE[m]["color"], hatch=STYLE[m]["hatch"],
                   edgecolor="white", linewidth=0.8, error_kw=dict(ecolor=INK, lw=1))
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in PROFILE_ORDER], color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"Simulated students, {n_episodes} sessions per profile (bars: mean, whiskers: 95% CI)",
                 y=0.02, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(log_path, path):
    returns = []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            returns.append(float(row["return"]))
    returns = np.array(returns)
    w = min(100, len(returns))
    smooth = np.convolve(returns, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(np.arange(len(returns)) + 1, returns, color="#009E73", alpha=0.15, lw=0.8)
    ax.plot(np.arange(w, len(returns) + 1), smooth, color="#009E73", lw=2, label=f"DQN, {w}-episode moving average")
    ax.set_xlabel("Training episode (one quiz session)", color=MUTED)
    ax.set_ylabel("Cumulative reward per session", color=MUTED)
    ax.set_title("DQN training curve (random student profile each episode, includes exploration)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trace(dqn, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, facecolor="white")
    for ax, profile in zip(axes, PROFILE_ORDER):
        env = QuizEnv(seed=777)
        _, trace = run_episode(dqn, env, profile, record_trace=True)
        diff = [t[0] for t in trace]
        mastery = [t[1] for t in trace]
        q = np.arange(1, len(trace) + 1)
        ax.step(q, diff, where="mid", color="#009E73", lw=2, label="Difficulty chosen by DQN")
        ax.scatter(q, diff, s=40, color=["#009E73" if t[2] else "white" for t in trace],
                   edgecolor="#009E73", zorder=3, label=None)
        ax.plot(q, np.array(mastery) * 2, color="#E69F00", lw=2, ls="--", label="True mastery (scaled 0-2)")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(DIFFICULTY_NAMES)
        ax.set_title(profile.capitalize(), loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Question number", color=MUTED)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("One sample session per profile (filled dot = correct answer, hollow = wrong)",
                 y=0.03, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(rows, methods, path, n_episodes):
    lines = [f"# Evaluation summary ({n_episodes} sessions per profile, mean +/- 95% CI)\n"]
    for metric, title in [("reward", "Cumulative reward"), ("accuracy", "Quiz accuracy"),
                          ("mastery_gain", "Mastery improvement"),
                          ("appropriateness", "Difficulty appropriateness")]:
        lines.append(f"\n## {title}\n")
        lines.append("| Method | " + " | ".join(p.capitalize() for p in PROFILE_ORDER) + " |")
        lines.append("|---|" + "---|" * len(PROFILE_ORDER))
        for m in methods:
            cells = []
            for p in PROFILE_ORDER:
                mean, ci = summarise(rows, metric, m, p)
                cells.append(f"{mean:.3f} +/- {ci:.3f}" if metric != "reward" else f"{mean:.1f} +/- {ci:.1f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="results/policy_numpy.npz")
    ap.add_argument("--log", default="results/training_log_numpy.csv")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dqn = MLPPolicy(args.policy, name="DQN")
    policies = [FixedPolicy(1), RuleBasedPolicy(), dqn]
    policies[0].name = "Fixed (medium)"
    methods = [p.name for p in policies]

    rows = evaluate(policies, args.episodes)
    with open(os.path.join(args.out, "eval_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(write_summary(rows, methods, os.path.join(args.out, "summary.md"), args.episodes))
    plot_comparison(rows, methods, os.path.join(args.out, "comparison.png"), args.episodes)
    plot_trace(dqn, os.path.join(args.out, "session_trace.png"))
    if os.path.exists(args.log):
        plot_training(args.log, os.path.join(args.out, "training_curve.png"))
    print("\nSaved plots and tables to", args.out)


if __name__ == "__main__":
    main()
