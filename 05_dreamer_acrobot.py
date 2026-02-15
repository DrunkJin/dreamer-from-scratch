#!/usr/bin/env python3
"""
05 — DreamerV3 on Acrobot-v1 (Sparse Reward Challenge)
========================================================
Acrobot is a significantly harder test for DreamerV3:
- 6-dim observation: [cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot]
- 3 discrete actions: apply +1, 0, or -1 torque to the joint
- SPARSE REWARD: -1 every step until the tip reaches the target height
- Must learn complex swing-up dynamics with delayed reward

Expected: episode length < 100 (solved) within 200-400 episodes.
Random policy averages ~500 steps.
"""

import gymnasium as gym
import numpy as np
import torch

from dreamer import DreamerAgent, SequenceReplayBuffer, set_seed, get_device
from dreamer import plot_training_curves


def train_dreamer_acrobot():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    env = gym.make('Acrobot-v1')
    obs_dim = env.observation_space.shape[0]  # 6
    action_dim = env.action_space.n            # 3

    agent = DreamerAgent(
        obs_dim=obs_dim, action_dim=action_dim, discrete=True, device=device,
        gru_dim=256, stoch_dim=16, stoch_classes=16,
        horizon=15, gamma=0.99, lam=0.95,
        lr_world=3e-4, lr_actor=1e-4, lr_critic=1e-4,
        entropy_coef=1e-3,  # Higher entropy for exploration
        kl_coef=0.5,
    )

    buffer = SequenceReplayBuffer(max_episodes=500)

    # ── Hyperparameters ──
    num_episodes = 100
    prefill_steps = 2000
    warmup_steps = 200
    train_every = 5
    batch_size = 16
    seq_len = 32

    # ── Prefill ──
    print(f"Prefilling buffer with {prefill_steps} random steps...")
    obs, _ = env.reset()
    steps = 0
    while steps < prefill_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs.astype(np.float32),
                   np.array([action], dtype=np.float32),
                   float(reward), done)
        obs = next_obs if not done else env.reset()[0]
        steps += 1
    print(f"  Buffer: {len(buffer)} episodes, {buffer.total_steps} steps")

    # ── World model warmup ──
    print(f"\nWarming up world model for {warmup_steps} steps...")
    for step in range(warmup_steps):
        batch = buffer.sample(batch_size, seq_len, device)
        losses = agent.train_step(batch, world_model_only=True)
        if (step + 1) % 100 == 0:
            print(f"  Warmup {step+1}/{warmup_steps} | "
                  f"world={losses['world_total']:.4f}")

    # ── Training ──
    episode_returns = []
    episode_lengths = []
    loss_history = {'world_total': [], 'actor': [], 'critic': [], 'kl': []}
    global_step = 0

    print(f"\nTraining DreamerV3 on Acrobot-v1 for {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        agent.reset_state()
        ep_return = 0
        ep_len = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs.astype(np.float32),
                       np.array([action], dtype=np.float32),
                       float(reward), done)

            global_step += 1
            if global_step % train_every == 0 and len(buffer) > 0:
                batch = buffer.sample(batch_size, seq_len, device)
                losses = agent.train_step(batch)
                for k in loss_history:
                    if k in losses:
                        loss_history[k].append(losses[k])

            obs = next_obs
            ep_return += reward
            ep_len += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

        if ep % 20 == 0 or ep == 1:
            recent_len = np.mean(episode_lengths[-20:])
            print(f"  Episode {ep:3d} | Return: {ep_return:6.1f} | "
                  f"Length: {ep_len:4d} | Avg20 Len: {recent_len:6.1f}")

        if len(episode_lengths) >= 30 and np.mean(episode_lengths[-30:]) < 100:
            print(f"\n  ★ Solved! 30-ep avg length: {np.mean(episode_lengths[-30:]):.1f}")
            break

    env.close()

    plot_training_curves(
        rewards=episode_returns,
        losses=loss_history,
        title='DreamerV3 Acrobot-v1',
        save_path='assets/05_acrobot_training.png'
    )

    torch.save(agent.state_dict(), 'assets/dreamer_acrobot.pt')
    print(f"\nFinal 20-ep avg length: {np.mean(episode_lengths[-20:]):.1f}")
    print(f"Model saved to assets/dreamer_acrobot.pt")


if __name__ == '__main__':
    train_dreamer_acrobot()
