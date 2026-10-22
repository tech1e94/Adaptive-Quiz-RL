"""Train the quiz-recommendation agent with TF-Agents DQN.

    python train_tfagents.py --episodes 3000

Needs tensorflow + tf-agents (see requirements.txt; Python 3.9-3.11).
Outputs:
    results/policy_tfagents.npz        Q-network weights (used by evaluate.py and the Streamlit app)
    results/training_log_tfagents.csv  return of every training episode
    results/tfagents_policy/           full TF-Agents SavedModel policy

Pipeline: QuizPyEnv -> TFPyEnvironment -> QNetwork -> DqnAgent
          -> TFUniformReplayBuffer -> DynamicStepDriver collect/train loop
"""
import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from quiz_rl.policies import MLPPolicy
from quiz_rl.tf_env import QuizPyEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tf.random.set_seed(args.seed)

    # 1) Environment (random student profile every episode)
    train_env = tf_py_environment.TFPyEnvironment(QuizPyEnv(profile=None, seed=args.seed))
    episode_len = 30
    total_steps = args.episodes * episode_len
    decay_steps = int(0.6 * total_steps)

    # 2) Q-network and DQN agent
    q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(),
                               fc_layer_params=(64, 64))
    train_step = tf.Variable(0, dtype=tf.int64)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=args.gamma,
        target_update_period=500,
        epsilon_greedy=lambda: tf.compat.v1.train.polynomial_decay(
            1.0, train_step, decay_steps, end_learning_rate=0.05, power=1.0),
        train_step_counter=train_step,
    )
    agent.initialize()

    # 3) Replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, batch_size=train_env.batch_size, max_length=50_000)

    # 4) Warm-up with a random policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    warmup = dynamic_step_driver.DynamicStepDriver(
        train_env, random_policy, observers=[replay_buffer.add_batch], num_steps=1000)
    warmup.run()

    # 5) Collection driver with metrics (buffer_size=1 -> return of the last finished episode)
    last_return = tf_metrics.AverageReturnMetric(buffer_size=1, batch_size=train_env.batch_size)
    n_episodes = tf_metrics.NumberOfEpisodes()
    driver = dynamic_step_driver.DynamicStepDriver(
        train_env, agent.collect_policy,
        observers=[replay_buffer.add_batch, last_return, n_episodes], num_steps=1)

    dataset = replay_buffer.as_dataset(num_parallel_calls=2, sample_batch_size=args.batch,
                                       num_steps=2).prefetch(3)
    iterator = iter(dataset)

    agent.train = common.function(agent.train)
    driver.run = common.function(driver.run)

    # 6) Train
    time_step = train_env.reset()
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    log, seen = [], 0
    for step in range(1, total_steps + 1):
        time_step, policy_state = driver.run(time_step, policy_state)
        experience, _ = next(iterator)
        loss = agent.train(experience).loss

        done_eps = int(n_episodes.result().numpy())
        if done_eps > seen:
            seen = done_eps
            log.append((seen, float(last_return.result().numpy()), ""))
            if seen % 250 == 0:
                recent = np.mean([x[1] for x in log[-250:]])
                print(f"episode {seen:5d}  avg return (last 250, with exploration) = {recent:6.2f}"
                      f"  loss = {float(loss):.4f}")

    # 7) Export
    with open(os.path.join(args.out, "training_log_tfagents.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "profile"])
        w.writerows(log)

    weights = q_net.get_weights()  # [W0, b0, W1, b1, W2, b2]
    arrays = {}
    for i in range(0, len(weights), 2):
        arrays[f"W{i // 2}"], arrays[f"b{i // 2}"] = weights[i], weights[i + 1]
    npz_path = os.path.join(args.out, "policy_tfagents.npz")
    np.savez(npz_path, **arrays)

    # Safety check: the NumPy copy must give the same Q-values as the TF network
    probe = np.random.default_rng(0).random((5, 3)).astype(np.float32)
    tf_q = q_net(probe)[0].numpy()
    np_policy = MLPPolicy(npz_path)
    np_q = np.stack([np_policy.q_values(p) for p in probe])
    if not np.allclose(tf_q, np_q, atol=1e-4):
        raise RuntimeError("Exported NumPy weights do not reproduce the TF-Agents Q-network; "
                           "check the layer order returned by q_net.get_weights().")
    print("Exported weights verified against the TF-Agents Q-network.")

    policy_saver.PolicySaver(agent.policy).save(os.path.join(args.out, "tfagents_policy"))
    print("saved", npz_path)
    print("Next:  python evaluate.py --policy results/policy_tfagents.npz "
          "--log results/training_log_tfagents.csv --out results_tfagents")


if __name__ == "__main__":
    main()
